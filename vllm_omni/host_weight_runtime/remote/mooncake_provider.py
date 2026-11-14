# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prototype remote artifact provider backed by a Mooncake object store.

The Host Weight Runtime (HWR) resolves one *exact* runtime-ready artifact from
three possible sources: a local store hit, a local producer, or -- the leg this
module adds -- a remote artifact pool.  Nothing here changes the store's
contracts: :meth:`MooncakeArtifactProvider.fetch` writes through the public
:class:`~vllm_omni.host_weight_runtime.protocols.ArtifactWriter` interface, so a
remote import is validated, hashed, quarantined and leased by exactly the same
machinery as a locally produced artifact.

Object layout inside Mooncake (one object per payload file keeps the remote
read granularity at "one tensor file", not "one whole artifact")::

    {prefix}/{artifact_key}/manifest.json    canonical WeightManifest bytes
    {prefix}/{artifact_key}/READY.json       PublicationMarker, written last
    {prefix}/{artifact_key}/{file_name}      safetensors payload, byte-identical
                                             to the locally published file

`READY.json` is the completion marker: a probe that sees a manifest without a
marker treats the publication as incomplete instead of trusting it.

Wiring point inside ``HostWeightRuntime.resolve`` (currently the remote branch
emits ``UNSUPPORTED`` without executing anything): replace that block with::

    remote = import_from_remote(self, identity, providers, deadline=deadline)
    attempts.append(self._store_attempt(ResolutionStage.REMOTE, remote, elapsed, action))
    if remote.lease is not None:
        return self._finish(ResolutionReport(..., outcome=ResolutionOutcome.REMOTE_IMPORT, ...),
                            lease=remote.lease)

``attempts`` then reads ``lookup -> remote -> (production)`` and
``ResolutionOutcome.REMOTE_IMPORT`` (already defined, currently unused) becomes
reachable.  ``tests/host_weight_runtime/test_runtime_resolution.py::
test_remote_required_is_explicitly_unsupported_and_skips_local_production``
asserts today's ``unsupported`` behaviour and must be updated with that change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from ..errors import FailureCode, HostWeightError, HostWeightFailure, ResolutionStage
from ..identity import WeightArtifactIdentity
from ..manifest import PublicationMarker, TensorKind, TensorManifestEntry, WeightManifest
from ..outcomes import StoreResult, StoreStatus
from ..protocols import (
    ArtifactWriter,
    BuildRequest,
    CoordinationScope,
    LookupPhase,
    ProductionMetadata,
    ProductionSourceMode,
    RemoteArtifactDescriptor,
    TensorWriteSpec,
    WeightProducer,
    WeightProductionSpec,
)

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
READY_FILE = "READY.json"
DEFAULT_OBJECT_PREFIX = "hwr/v1"
DEFAULT_PROVIDER_ID = "mooncake"
_HASH_CHUNK_BYTES = 8 << 20


def _failure(
    stage: ResolutionStage,
    code: FailureCode,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> HostWeightFailure:
    from ..identity import CanonicalJson  # noqa: PLC0415 - keeps module import graph flat

    return HostWeightFailure(
        stage=stage,
        code=code,
        retryable=retryable,
        message=message,
        details=CanonicalJson.from_value(details or {}),
    )


def _remote_error(
    code: FailureCode,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> HostWeightError:
    return HostWeightError(_failure(ResolutionStage.REMOTE, code, message, retryable=retryable, details=details))


# --------------------------------------------------------------------------- backends


class MooncakeBackendError(RuntimeError):
    """The object store is unreachable or rejected an operation."""


class MooncakeBackendTimeout(MooncakeBackendError):
    """The object store did not answer inside the caller's budget."""


@runtime_checkable
class ArtifactObjectBackend(Protocol):
    """Minimal byte-object interface the provider needs.

    Deliberately *not* the Mooncake client API: keeping the seam this narrow is
    what lets the CPU tests exercise the whole provider with an in-memory object
    store while a real deployment plugs ``MooncakeDistributedStoreBackend`` in.
    """

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, value: bytes) -> None: ...

    def remove(self, key: str) -> None: ...


class InMemoryArtifactBackend:
    """Deterministic stub used by tests and by master-less environments."""

    def __init__(self, *, latency_seconds: float = 0.0) -> None:
        self._objects: dict[str, bytes] = {}
        self._latency_seconds = latency_seconds
        self.gets = 0
        self.hits = 0
        self.puts = 0
        self.bytes_get = 0
        self.bytes_put = 0

    def get(self, key: str) -> bytes | None:
        self.gets += 1
        value = self._objects.get(key)
        if value is None:
            return None
        self.hits += 1
        self.bytes_get += len(value)
        return value

    def put(self, key: str, value: bytes) -> None:
        self.puts += 1
        self.bytes_put += len(value)
        self._objects[key] = bytes(value)

    def remove(self, key: str) -> None:
        self._objects.pop(key, None)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._objects))

    def __len__(self) -> int:
        return len(self._objects)


class MooncakeDistributedStoreBackend:
    """Adapter over ``mooncake.store.MooncakeDistributedStore``.

    Only the object operations are wrapped; the provider never sees a
    ``replica_num`` or a segment layout.  ``store`` can be injected for tests
    that own a client, otherwise the client is created and ``setup`` is called
    with the connector's positional contract
    (``host, metadata, segment, localbuf, protocol, rdma_devices, master``).
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        metadata_server: str = "http://127.0.0.1:8080/metadata",
        master_server: str = "127.0.0.1:50051",
        segment_bytes: int = 512 << 20,
        local_buffer_bytes: int = 64 << 20,
        protocol: str = "tcp",
        rdma_devices: str = "",
        store: Any | None = None,
    ) -> None:
        self._store = store if store is not None else self._connect(
            host,
            metadata_server,
            master_server,
            segment_bytes,
            local_buffer_bytes,
            protocol,
            rdma_devices,
        )
        self._replicate_config = self._make_replicate_config()

    @staticmethod
    def _connect(
        host: str,
        metadata_server: str,
        master_server: str,
        segment_bytes: int,
        local_buffer_bytes: int,
        protocol: str,
        rdma_devices: str,
    ) -> Any:
        try:
            from mooncake.store import MooncakeDistributedStore  # noqa: PLC0415 - optional dependency
        except Exception as exc:  # pragma: no cover - exercised only without the wheel
            raise MooncakeBackendError(f"mooncake python client is unavailable: {exc}") from exc
        client = MooncakeDistributedStore()
        status = client.setup(
            host,
            metadata_server,
            segment_bytes,
            local_buffer_bytes,
            protocol,
            rdma_devices,
            master_server,
        )
        if status != 0:
            raise MooncakeBackendError(f"MooncakeDistributedStore.setup returned {status}")
        return client

    def _make_replicate_config(self) -> Any:
        try:
            from mooncake.store import ReplicateConfig  # noqa: PLC0415
        except Exception:
            return None
        config = ReplicateConfig()
        config.replica_num = 1
        return config

    def get(self, key: str) -> bytes | None:
        try:
            if hasattr(self._store, "is_exist") and self._store.is_exist(key) != 1:
                return None
            value = self._store.get(key)
        except TimeoutError as exc:
            raise MooncakeBackendTimeout(f"Mooncake get timed out for {key!r}: {exc}") from exc
        except Exception as exc:
            raise MooncakeBackendError(f"Mooncake get failed for {key!r}: {exc}") from exc
        return bytes(value) if value else None

    def put(self, key: str, value: bytes) -> None:
        try:
            if self._replicate_config is None:
                status = self._store.put(key, value)
            else:
                status = self._store.put(key, value, self._replicate_config)
        except TimeoutError as exc:
            raise MooncakeBackendTimeout(f"Mooncake put timed out for {key!r}: {exc}") from exc
        except Exception as exc:
            raise MooncakeBackendError(f"Mooncake put failed for {key!r}: {exc}") from exc
        if status != 0:
            raise MooncakeBackendError(f"Mooncake put returned {status} for {key!r}")

    def remove(self, key: str) -> None:
        try:
            self._store.remove(key, True)
        except Exception as exc:
            raise MooncakeBackendError(f"Mooncake remove failed for {key!r}: {exc}") from exc

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()


# --------------------------------------------------------------------------- stats


@dataclass(frozen=True)
class RemoteFetchStats:
    """Measured cost of one import; the PoC reports these instead of guessing."""

    provider_id: str
    artifact_key: str
    files: int
    tensors: int
    payload_bytes: int
    seconds: float
    mb_per_s: float
    manifest_sha256: str
    content_sha256: str


# --------------------------------------------------------------------------- provider


class MooncakeArtifactProvider:
    """``RemoteArtifactProvider`` over a Mooncake-style object store."""

    def __init__(
        self,
        backend: ArtifactObjectBackend,
        *,
        provider_id: str = DEFAULT_PROVIDER_ID,
        prefix: str = DEFAULT_OBJECT_PREFIX,
    ) -> None:
        if not provider_id:
            raise ValueError("remote provider_id must be non-empty")
        self._backend = backend
        self._provider_id = provider_id
        self._prefix = prefix.strip("/")
        self.last_fetch_stats: RemoteFetchStats | None = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    # -- object keys ------------------------------------------------------
    def _base(self, artifact_key: str) -> str:
        if not artifact_key:
            raise ValueError("remote object keys require a non-empty artifact key")
        return f"{self._prefix}/{artifact_key}" if self._prefix else artifact_key

    def manifest_key(self, artifact_key: str) -> str:
        return f"{self._base(artifact_key)}/{MANIFEST_FILE}"

    def ready_key(self, artifact_key: str) -> str:
        return f"{self._base(artifact_key)}/{READY_FILE}"

    def payload_key(self, artifact_key: str, file_name: str) -> str:
        return f"{self._base(artifact_key)}/{file_name}"

    # -- backend helpers --------------------------------------------------
    def _get_optional(self, key: str) -> bytes | None:
        try:
            return self._backend.get(key)
        except MooncakeBackendTimeout as exc:
            raise _remote_error(FailureCode.REMOTE_TIMEOUT, str(exc), retryable=True) from exc
        except MooncakeBackendError as exc:
            raise _remote_error(FailureCode.REMOTE_UNAVAILABLE, str(exc), retryable=True) from exc

    def _get_required(self, key: str, *, artifact_key: str) -> bytes:
        value = self._get_optional(key)
        if value is None:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"published artifact {artifact_key} is missing object {key!r}",
                details={"object": key, "artifact_key": artifact_key},
            )
        return value

    # -- probe ------------------------------------------------------------
    def probe(self, identity: WeightArtifactIdentity) -> RemoteArtifactDescriptor | None:
        """Return a descriptor for one complete remote publication, else ``None``."""
        manifest_bytes = self._get_optional(self.manifest_key(identity.key))
        ready_bytes = self._get_optional(self.ready_key(identity.key))
        if manifest_bytes is None and ready_bytes is None:
            logger.debug("[HWR-remote] %s: no objects under %s", self._provider_id, self._base(identity.key))
            return None
        if manifest_bytes is None or ready_bytes is None:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote publication for {identity.key} is incomplete "
                f"(manifest={'present' if manifest_bytes else 'missing'}, "
                f"ready={'present' if ready_bytes else 'missing'})",
                details={"artifact_key": identity.key},
            )
        manifest, marker, manifest_sha256 = self._parse_publication(manifest_bytes, ready_bytes, identity)
        logger.info(
            "[HWR-remote] %s probe hit %s: files=%d tensors=%d bytes=%d content=%s",
            self._provider_id,
            identity.key[:16],
            len(manifest.files),
            len(manifest.tensors),
            manifest.total_tensor_bytes,
            marker.artifact_content_sha256[:16],
        )
        return RemoteArtifactDescriptor(
            identity=identity,
            provider_id=self._provider_id,
            manifest_sha256=manifest_sha256,
            content_sha256=manifest.artifact_content_sha256,
        )

    def _parse_publication(
        self,
        manifest_bytes: bytes,
        ready_bytes: bytes,
        identity: WeightArtifactIdentity,
    ) -> tuple[WeightManifest, PublicationMarker, str]:
        try:
            manifest = WeightManifest.from_dict(json.loads(manifest_bytes))
            marker = PublicationMarker.from_dict(json.loads(ready_bytes))
        except Exception as exc:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote manifest for {identity.key} is not a valid publication: {exc}",
                details={"artifact_key": identity.key},
            ) from exc
        if manifest.canonical_bytes != manifest_bytes:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote manifest for {identity.key} is not canonically encoded",
                details={"artifact_key": identity.key},
            )
        if manifest.artifact_key != identity.key:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote artifact_key {manifest.artifact_key[:16]} does not match identity {identity.key[:16]}",
                details={"artifact_key": identity.key},
            )
        if manifest.identity.canonical_bytes != identity.canonical_bytes:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote artifact identity for {identity.key[:16]} does not match the requested identity",
                details={"artifact_key": identity.key},
            )
        if (
            manifest.producer_schema != identity.producer.manifest_schema
            or manifest.restorer_schema != identity.producer.restorer_schema
        ):
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote schemas ({manifest.producer_schema}, {manifest.restorer_schema}) do not match "
                f"identity ({identity.producer.manifest_schema}, {identity.producer.restorer_schema})",
                details={"artifact_key": identity.key},
            )
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if marker.manifest_sha256 != manifest_sha256:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"READY marker of {identity.key[:16]} does not seal the stored manifest",
                details={"artifact_key": identity.key},
            )
        if marker.artifact_content_sha256 != manifest.artifact_content_sha256:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"READY marker of {identity.key[:16]} disagrees with the manifest content digest",
                details={"artifact_key": identity.key},
            )
        return manifest, marker, manifest_sha256

    # -- fetch ------------------------------------------------------------
    def fetch(self, descriptor: RemoteArtifactDescriptor, writer: ArtifactWriter) -> ProductionMetadata:
        """Stream every declared payload file into ``writer`` and re-verify it."""
        identity = descriptor.identity
        artifact_key = identity.key
        started = time.monotonic()
        self.last_fetch_stats = None  # a failed fetch must not leave stale numbers behind
        manifest_bytes = self._get_required(self.manifest_key(artifact_key), artifact_key=artifact_key)
        ready_bytes = self._get_required(self.ready_key(artifact_key), artifact_key=artifact_key)
        manifest, marker, manifest_sha256 = self._parse_publication(manifest_bytes, ready_bytes, identity)
        if descriptor.manifest_sha256 and descriptor.manifest_sha256 != manifest_sha256:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote manifest of {artifact_key[:16]} changed between probe and fetch",
                details={"artifact_key": artifact_key},
            )
        if descriptor.content_sha256 and descriptor.content_sha256 != manifest.artifact_content_sha256:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote content digest of {artifact_key[:16]} changed between probe and fetch",
                details={"artifact_key": artifact_key},
            )

        entries_by_file: dict[str, list[TensorManifestEntry]] = defaultdict(list)
        for entry in manifest.tensors:
            entries_by_file[entry.file_name].append(entry)

        payload_bytes = 0
        for file_entry in manifest.files:
            payload = self._get_required(
                self.payload_key(artifact_key, file_entry.file_name), artifact_key=artifact_key
            )
            if len(payload) != file_entry.size:
                raise _remote_error(
                    FailureCode.REMOTE_ARTIFACT_INVALID,
                    f"remote payload {file_entry.file_name!r} is {len(payload)} bytes, manifest says {file_entry.size}",
                    details={"artifact_key": artifact_key, "file_name": file_entry.file_name},
                )
            if hashlib.sha256(payload).hexdigest() != file_entry.sha256:
                raise _remote_error(
                    FailureCode.REMOTE_ARTIFACT_INVALID,
                    f"remote payload {file_entry.file_name!r} does not match its manifest sha256",
                    details={"artifact_key": artifact_key, "file_name": file_entry.file_name},
                )
            loaded = self._load_safetensors(payload, file_entry.file_name, artifact_key)
            entries = sorted(entries_by_file.get(file_entry.file_name, ()), key=lambda item: item.name)
            if not entries:
                raise _remote_error(
                    FailureCode.REMOTE_ARTIFACT_INVALID,
                    f"remote payload {file_entry.file_name!r} declares no tensors",
                    details={"artifact_key": artifact_key, "file_name": file_entry.file_name},
                )
            specs = tuple(self._spec_for(entry) for entry in entries)
            with writer.open_tensor_file(file_entry.file_name, specs, ordered=True) as output:
                for entry in entries:
                    tensor = loaded.pop(entry.name, None)
                    if tensor is None:
                        raise _remote_error(
                            FailureCode.REMOTE_ARTIFACT_INVALID,
                            f"remote payload {file_entry.file_name!r} lacks tensor {entry.name!r}",
                            details={"artifact_key": artifact_key, "tensor": entry.name},
                        )
                    self._verify_tensor(entry, tensor, artifact_key)
                    output.write_tensor(entry.name, tensor)
            if loaded:
                raise _remote_error(
                    FailureCode.REMOTE_ARTIFACT_INVALID,
                    f"remote payload {file_entry.file_name!r} carries undeclared tensors: {sorted(loaded)[:4]}",
                    details={"artifact_key": artifact_key, "file_name": file_entry.file_name},
                )
            payload_bytes += len(payload)

        elapsed = time.monotonic() - started
        self.last_fetch_stats = RemoteFetchStats(
            provider_id=self._provider_id,
            artifact_key=artifact_key,
            files=len(manifest.files),
            tensors=len(manifest.tensors),
            payload_bytes=payload_bytes,
            seconds=elapsed,
            mb_per_s=(payload_bytes / (1 << 20) / elapsed) if elapsed > 0 else 0.0,
            manifest_sha256=manifest_sha256,
            content_sha256=marker.artifact_content_sha256,
        )
        logger.info(
            "[HWR-remote] %s imported %s: files=%d tensors=%d bytes=%d in %.2fs (%.1f MiB/s)",
            self._provider_id,
            artifact_key[:16],
            len(manifest.files),
            len(manifest.tensors),
            payload_bytes,
            elapsed,
            self.last_fetch_stats.mb_per_s,
        )
        return ProductionMetadata(
            producer_schema=manifest.producer_schema,
            restorer_schema=manifest.restorer_schema,
            format_metadata=manifest.format_metadata,
        )

    @staticmethod
    def _load_safetensors(payload: bytes, file_name: str, artifact_key: str) -> dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load as load_safetensors  # noqa: PLC0415 - lazy optional import
        except Exception as exc:  # pragma: no cover - safetensors is a hard dependency in practice
            raise _remote_error(
                FailureCode.REMOTE_UNAVAILABLE,
                f"safetensors is required to decode remote payloads: {exc}",
                retryable=False,
            ) from exc
        try:
            return load_safetensors(payload)
        except Exception as exc:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"remote payload {file_name!r} of {artifact_key[:16]} is not readable safetensors: {exc}",
                details={"artifact_key": artifact_key, "file_name": file_name},
            ) from exc

    @staticmethod
    def _spec_for(entry: TensorManifestEntry) -> TensorWriteSpec:
        return TensorWriteSpec(
            name=entry.name,
            shape=entry.shape,
            dtype=getattr(torch, entry.dtype.split(".")[-1]),
            kind=TensorKind(entry.kind),
            role=entry.role,
        )

    @staticmethod
    def _verify_tensor(entry: TensorManifestEntry, tensor: torch.Tensor, artifact_key: str) -> None:
        if tuple(tensor.shape) != tuple(entry.shape):
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"tensor {entry.name!r} shape {tuple(tensor.shape)} != manifest {tuple(entry.shape)}",
                details={"artifact_key": artifact_key, "tensor": entry.name},
            )
        if str(tensor.dtype) != entry.dtype:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"tensor {entry.name!r} dtype {tensor.dtype} != manifest {entry.dtype}",
                details={"artifact_key": artifact_key, "tensor": entry.name},
            )
        tensor = tensor.contiguous()
        if _tensor_sha256(tensor) != entry.sha256:
            raise _remote_error(
                FailureCode.REMOTE_ARTIFACT_INVALID,
                f"tensor {entry.name!r} bytes do not match its manifest sha256",
                details={"artifact_key": artifact_key, "tensor": entry.name},
            )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    for offset in range(0, raw.numel(), _HASH_CHUNK_BYTES):
        digest.update(raw[offset : offset + _HASH_CHUNK_BYTES].numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- publish


def publish_artifact_from_directory(
    backend: ArtifactObjectBackend,
    entry_dir: Path,
    *,
    prefix: str = DEFAULT_OBJECT_PREFIX,
    provider_id: str = DEFAULT_PROVIDER_ID,
    identity: WeightArtifactIdentity | None = None,
) -> RemoteArtifactDescriptor:
    """Upload one finished local artifact directory (payloads, manifest, READY)."""
    entry_dir = Path(entry_dir)
    manifest_path = entry_dir / MANIFEST_FILE
    ready_path = entry_dir / READY_FILE
    manifest_bytes = manifest_path.read_bytes()
    ready_bytes = ready_path.read_bytes()
    manifest = WeightManifest.from_dict(json.loads(manifest_bytes))
    if manifest.canonical_bytes != manifest_bytes:
        raise ValueError(f"{manifest_path} is not canonically encoded; refusing to publish")
    marker = PublicationMarker.from_dict(json.loads(ready_bytes))
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if marker.manifest_sha256 != manifest_sha256 or marker.artifact_content_sha256 != manifest.artifact_content_sha256:
        raise ValueError(f"{ready_path} does not seal {manifest_path}; refusing to publish")
    if identity is not None and identity.key != manifest.artifact_key:
        raise ValueError("publishing identity does not match the artifact manifest")

    base = f"{prefix.strip('/')}/{manifest.artifact_key}" if prefix.strip("/") else manifest.artifact_key
    for file_entry in manifest.files:
        path = entry_dir / file_entry.file_name
        payload = path.read_bytes()
        if len(payload) != file_entry.size or hashlib.sha256(payload).hexdigest() != file_entry.sha256:
            raise ValueError(f"{path} does not match the manifest file entry; refusing to publish")
        backend.put(f"{base}/{file_entry.file_name}", payload)
    backend.put(f"{base}/{MANIFEST_FILE}", manifest_bytes)
    backend.put(f"{base}/{READY_FILE}", ready_bytes)  # completion marker goes last
    logger.info(
        "[HWR-remote] published %s: files=%d bytes=%d",
        manifest.artifact_key[:16],
        len(manifest.files),
        manifest.total_tensor_bytes,
    )
    return RemoteArtifactDescriptor(
        identity=identity or manifest.identity,
        provider_id=provider_id,
        manifest_sha256=manifest_sha256,
        content_sha256=manifest.artifact_content_sha256,
    )


# --------------------------------------------------------------------------- wiring


class RemoteImportProducer:
    """``WeightProducer`` adapter: a remote fetch looks like any local build."""

    def __init__(
        self,
        provider: MooncakeArtifactProvider,
        descriptor: RemoteArtifactDescriptor,
        identity: WeightArtifactIdentity,
        *,
        producer_id: str | None = None,
    ) -> None:
        if descriptor.identity.key != identity.key:
            raise ValueError("remote descriptor does not describe the requested identity")
        self._provider = provider
        self._descriptor = descriptor
        self._spec = WeightProductionSpec(
            producer_id=producer_id or identity.producer.producer_id,
            outputs=(identity,),
            source_mode=ProductionSourceMode.CHECKPOINT_STREAM,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
            lookup_phase=LookupPhase.PRE_LOAD_SAFE,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        return self._provider.fetch(self._descriptor, writer)


def import_from_remote(
    runtime: Any,
    identity: WeightArtifactIdentity,
    providers: Sequence[MooncakeArtifactProvider],
    *,
    deadline: float | None = None,
) -> StoreResult:
    """Probe ``providers`` in order and import the first complete publication.

    Returns the store's own :class:`StoreResult`, so the caller decides policy
    (``PREFERRED`` vs ``REQUIRED``) exactly as it already does for producers.
    """
    store = getattr(runtime, "store", None)
    if store is None:
        return StoreResult(
            StoreStatus.FAILED,
            failure=_failure(
                ResolutionStage.DOMAIN,
                FailureCode.DOMAIN_UNAVAILABLE,
                "remote import requires an initialized host weight store",
            ),
        )
    allowed = set(runtime.config.remote.providers) if getattr(runtime, "config", None) is not None else set()
    if deadline is None:
        wait = runtime.config.wait
        deadline = time.monotonic() + float(wait.coordination_timeout_seconds)
    validation = runtime.config.integrity.local_lookup
    candidates = [provider for provider in providers if not allowed or provider.provider_id in allowed]
    if not candidates:
        return StoreResult(StoreStatus.MISS)
    for provider in candidates:
        descriptor = provider.probe(identity)
        if descriptor is None:
            continue
        producer = RemoteImportProducer(provider, descriptor, identity)
        result = store.get_or_build(
            BuildRequest(identity=identity),
            producer,
            validation=validation,
            deadline=deadline,
        )
        logger.info(
            "[HWR-remote] %s import of %s finished with %s",
            provider.provider_id,
            identity.key[:16],
            result.status.value,
        )
        return result
    logger.info("[HWR-remote] no provider published %s; treating as a miss", identity.key[:16])
    return StoreResult(StoreStatus.MISS)


__all__ = [
    "ArtifactObjectBackend",
    "DEFAULT_OBJECT_PREFIX",
    "DEFAULT_PROVIDER_ID",
    "InMemoryArtifactBackend",
    "MANIFEST_FILE",
    "MooncakeArtifactProvider",
    "MooncakeBackendError",
    "MooncakeBackendTimeout",
    "MooncakeDistributedStoreBackend",
    "READY_FILE",
    "RemoteFetchStats",
    "RemoteImportProducer",
    "import_from_remote",
    "publish_artifact_from_directory",
]
