# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Mooncake remote artifact provider prototype.

The tests exercise the whole provider surface against an in-memory object
store: publication ordering, probe validation, byte-exact streaming into a real
filesystem store, the store's own manifest/lease machinery, and the two
corruption paths (sealed READY marker, payload bytes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    ComponentIdentity,
    CoordinationScope,
    FailureCode,
    HostWeightError,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    ProducerIdentity,
    ProductionMetadata,
    RemoteImportPolicy,
    RemoteOnMiss,
    RuntimeMode,
    RuntimeWeightLayout,
    StorageDomainPolicy,
    StoreStatus,
    WeightArtifactIdentity,
    WeightRepresentation,
    WeightSourceIdentity,
)
from vllm_omni.host_weight_runtime.filesystem import detect_storage_class
from vllm_omni.host_weight_runtime.filesystem.writer import FilesystemArtifactWriter
from vllm_omni.host_weight_runtime.manifest import PublicationMarker, WeightManifest
from vllm_omni.host_weight_runtime.protocols import (
    ProductionSourceMode,
    TensorWriteSpec,
    WeightProductionSpec,
)
from vllm_omni.host_weight_runtime.remote import (
    InMemoryArtifactBackend,
    MooncakeArtifactProvider,
    import_from_remote,
    publish_artifact_from_directory,
)

pytestmark = pytest.mark.cpu

MIB = 1 << 20
ARTIFACT_MIB = 64
_LAYER_SHAPE = (2048, 4096)  # 16 MiB in bf16
_EMBED_SHAPE = (4096, 4096)  # 32 MiB in bf16


def _identity(component: str = "transformer") -> WeightArtifactIdentity:
    return WeightArtifactIdentity(
        schema_version=1,
        source=WeightSourceIdentity("test-org/test-model", "0123456789abcdef", "source-files-sha256"),
        component=ComponentIdentity(component, "diffusion.dit"),
        representation=WeightRepresentation("runtime-bf16", "torch.bfloat16"),
        layout=RuntimeWeightLayout("final-module-layout"),
        adaptation=AdaptationIdentity(),
        producer=ProducerIdentity(
            "test.final-layout",
            "1",
            "implementation-sha256",
            "test-producer-v1",
            "test-restorer-v1",
        ),
    )


class _SourceProducer:
    """Emits a 64 MiB two-file artifact so byte identity is worth checking."""

    def __init__(self, identity: WeightArtifactIdentity) -> None:
        self._spec = WeightProductionSpec(
            producer_id="test.final-layout",
            outputs=(identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: object) -> ProductionMetadata:
        transformer_specs = (
            TensorWriteSpec("layers.0.weight", _LAYER_SHAPE, torch.bfloat16),
            TensorWriteSpec("layers.1.weight", _LAYER_SHAPE, torch.bfloat16),
        )
        with writer.open_tensor_file("transformer.safetensors", transformer_specs) as output:  # type: ignore[attr-defined]
            output.write_tensor("layers.0.weight", torch.full(_LAYER_SHAPE, 0.5, dtype=torch.bfloat16))
            output.write_tensor("layers.1.weight", torch.full(_LAYER_SHAPE, 1.5, dtype=torch.bfloat16))
        embed_specs = (TensorWriteSpec("embed.weight", _EMBED_SHAPE, torch.bfloat16),)
        with writer.open_tensor_file("embed.safetensors", embed_specs) as output:  # type: ignore[attr-defined]
            output.write_tensor("embed.weight", torch.full(_EMBED_SHAPE, 2.5, dtype=torch.bfloat16))
        return ProductionMetadata(
            producer_schema="test-producer-v1",
            restorer_schema="test-restorer-v1",
            format_metadata=CanonicalJson.from_value({"source": "mooncake-poc"}),
        )


@pytest.fixture(scope="module")
def source_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, WeightArtifactIdentity, WeightManifest]:
    entry_dir = tmp_path_factory.mktemp("source-artifact")
    identity = _identity()
    writer = FilesystemArtifactWriter(entry_dir, capacity_check=lambda reserved, total: None)
    metadata = _SourceProducer(identity).produce(writer)
    manifest, _marker = writer.finalize(identity, metadata)
    return entry_dir, identity, manifest


def _runtime(root: Path, *, providers: tuple[str, ...] = ("mooncake",)) -> HostWeightRuntime:
    domain = StorageDomainPolicy(root=root, storage_class=detect_storage_class(root.parent))
    config = HostWeightRuntimeConfig(
        mode=RuntimeMode.REQUIRED,
        domain=domain,
        remote=RemoteImportPolicy(on_local_miss=RemoteOnMiss.REQUIRE, providers=providers),
    )
    return HostWeightRuntime.from_config(config)


def _published(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest],
) -> tuple[InMemoryArtifactBackend, WeightArtifactIdentity, WeightManifest]:
    entry_dir, identity, manifest = source_artifact
    backend = InMemoryArtifactBackend()
    publish_artifact_from_directory(backend, entry_dir)
    return backend, identity, manifest


def test_import_streams_artifact_and_measures_throughput(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    backend, identity, manifest = _published(source_artifact)
    provider = MooncakeArtifactProvider(backend)
    runtime = _runtime(tmp_path / "store")

    result = import_from_remote(runtime, identity, [provider])

    assert result.status is StoreStatus.BUILT
    assert result.lease is not None
    try:
        published = result.lease.manifest
        assert published.canonical_bytes == manifest.canonical_bytes
        assert published.artifact_content_sha256 == manifest.artifact_content_sha256
        assert published.producer_schema == "test-producer-v1"
        assert published.restorer_schema == "test-restorer-v1"
        assert published.format_metadata == CanonicalJson.from_value({"source": "mooncake-poc"})
    finally:
        result.lease.close()

    stats = provider.last_fetch_stats
    assert stats is not None
    assert stats.files == 2
    assert stats.tensors == 3
    assert manifest.total_tensor_bytes == ARTIFACT_MIB * MIB
    # Payload objects carry the safetensors header on top of the tensor bytes.
    assert stats.payload_bytes == sum(entry.size for entry in manifest.files)
    assert stats.payload_bytes > manifest.total_tensor_bytes
    assert stats.seconds > 0.0
    assert stats.mb_per_s > 0.0
    assert stats.manifest_sha256
    assert stats.content_sha256 == manifest.artifact_content_sha256


def test_second_import_hits_local_store_without_remote_reads(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    backend, identity, _manifest = _published(source_artifact)
    provider = MooncakeArtifactProvider(backend)
    runtime = _runtime(tmp_path / "store")

    first = import_from_remote(runtime, identity, [provider])
    assert first.status is StoreStatus.BUILT
    first_stats = provider.last_fetch_stats
    if first.lease is not None:
        first.lease.close()
    remote_reads = backend.gets

    second = import_from_remote(runtime, identity, [provider])

    assert second.status is StoreStatus.HIT
    if second.lease is not None:
        second.lease.close()
    # A probe still reads the manifest and READY marker, but no payload is fetched,
    # so the reported numbers stay those of the first (real) import.
    assert backend.gets == remote_reads + 2
    assert provider.last_fetch_stats is first_stats


def test_unpublished_artifact_reports_miss(tmp_path: Path) -> None:
    identity = _identity()
    backend = InMemoryArtifactBackend()
    provider = MooncakeArtifactProvider(backend)
    runtime = _runtime(tmp_path / "store")

    result = import_from_remote(runtime, identity, [provider])

    assert result.status is StoreStatus.MISS
    assert result.lease is None
    assert provider.last_fetch_stats is None


def test_provider_filter_skips_untouched_backend(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    backend, identity, _manifest = _published(source_artifact)
    provider = MooncakeArtifactProvider(backend)
    runtime = _runtime(tmp_path / "store", providers=("some-other-pool",))

    result = import_from_remote(runtime, identity, [provider])

    assert result.status is StoreStatus.MISS
    assert backend.gets == 0


def test_unsealed_ready_marker_is_rejected(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    backend, identity, manifest = _published(source_artifact)
    provider = MooncakeArtifactProvider(backend)
    forged = PublicationMarker(
        schema_version=1,
        publication_id="forged",
        manifest_sha256="0" * 64,
        artifact_content_sha256=manifest.artifact_content_sha256,
        integrity_mode="full_checksum",
    )
    backend.put(provider.ready_key(identity.key), json.dumps(forged.to_dict()).encode("utf-8"))

    with pytest.raises(HostWeightError) as caught:
        provider.probe(identity)

    assert caught.value.failure.code is FailureCode.REMOTE_ARTIFACT_INVALID


def test_corrupted_payload_is_rejected_before_publish(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    backend, identity, manifest = _published(source_artifact)
    provider = MooncakeArtifactProvider(backend)
    runtime = _runtime(tmp_path / "store")
    payload_key = provider.payload_key(identity.key, manifest.files[0].file_name)
    payload = bytearray(backend.get(payload_key) or b"")
    payload[-1] ^= 0xFF
    backend.put(payload_key, bytes(payload))

    result = import_from_remote(runtime, identity, [provider])

    assert result.status is StoreStatus.FAILED
    assert result.failure is not None
    assert result.failure.code is FailureCode.REMOTE_ARTIFACT_INVALID


def test_publish_refuses_tampered_directory(
    source_artifact: tuple[Path, WeightArtifactIdentity, WeightManifest], tmp_path: Path
) -> None:
    entry_dir, identity, manifest = source_artifact
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for path in entry_dir.iterdir():
        (tampered / path.name).write_bytes(path.read_bytes())
    target = tampered / manifest.files[0].file_name
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="does not match the manifest"):
        publish_artifact_from_directory(InMemoryArtifactBackend(), tampered, identity=identity)
