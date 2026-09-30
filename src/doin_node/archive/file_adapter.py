"""Content-addressed file shadow.

data_lake_service.LakeBackend declares no byte-upload capability. Its
write_metrics operation stores a consumer report (SQL gov_report / gov_metric
/ gov_dataset rows, or the HTTP JSON reply from POST /api/v1/metrics), not
the resource body. data_gov.resource_registration.ResourceRegistry.append
writes one catalog JSON row of declared facts and content_sha256 by exclusive
create. Registering a resource does not copy resource bytes.

This adapter is the stage-1 shadow store for those bytes. It is a disposable
local directory, not a lake deployment and not multi-node.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from doin_core.archive.body import ArchiveEnvelope, ArchiveRefusal, digest_bytes

LAKE_WRITE_FACT = (
    "data_lake_service.LakeBackend declares no byte-upload capability. "
    "write_metrics stores a consumer report, not resource bytes. "
    "data_gov.resource_registration.ResourceRegistry.append writes one catalog "
    "JSON row and does not copy resource bytes. This file adapter is the "
    "disposable shadow store for the bytes."
)


@dataclass(frozen=True)
class ArchiveReceipt:
    """Read-back receipt. ``verified`` is true only after the hashes match."""

    body_digest: str
    manifest_digest: str
    header_hash: str | None
    anchor: str
    already_stored: bool
    verified: bool


class FileArchiveAdapter:
    """Temporary directory of content-addressed body, section and manifest bytes."""

    LABEL = "DISPOSABLE_FILE_NOT_LAKE"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, envelope: ArchiveEnvelope) -> ArchiveReceipt:
        if digest_bytes(envelope.body) != envelope.body_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        if digest_bytes(envelope.manifest) != envelope.manifest_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        self._store("objects", envelope.body_digest, envelope.body)
        self._store("manifests", envelope.manifest_digest, envelope.manifest)
        for section in envelope.sections:
            self._store("objects", section.digest, section.content)
        existed = self._index_path(envelope.body_digest).is_file()
        if existed:
            self._require_same_index(envelope)
        else:
            self._write_index(envelope)
        receipt = ArchiveReceipt(
            body_digest=envelope.body_digest,
            manifest_digest=envelope.manifest_digest,
            header_hash=envelope.header_hash,
            anchor=envelope.anchor,
            already_stored=existed,
            verified=False,
        )
        self.read_verified(receipt)
        return ArchiveReceipt(
            body_digest=receipt.body_digest,
            manifest_digest=receipt.manifest_digest,
            header_hash=receipt.header_hash,
            anchor=receipt.anchor,
            already_stored=existed,
            verified=True,
        )

    def read_body(self, digest: str) -> bytes:
        return self._read_exact(self.root / "objects" / _hex64(digest), digest)

    def read_manifest(self, digest: str) -> bytes:
        return self._read_exact(self.root / "manifests" / _hex64(digest), digest)

    def read_verified(self, receipt: ArchiveReceipt) -> tuple[bytes, bytes]:
        manifest = self.read_manifest(receipt.manifest_digest)
        parsed = json.loads(manifest)
        if parsed.get("manifest_digest") is not None:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        if parsed.get("body_digest") != receipt.body_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        body = self.read_body(receipt.body_digest)
        for section in parsed.get("sections") or []:
            content = self.read_body(section["digest"])
            if len(content) != section["size"]:
                raise ArchiveRefusal("DIGEST_MISMATCH")
        return body, manifest

    def record_count(self) -> int:
        index = self.root / "index"
        if not index.is_dir():
            return 0
        return sum(1 for path in index.glob("*.json") if path.is_file())

    def _store(self, directory: str, digest: str, content: bytes) -> None:
        path = self.root / directory / _hex64(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = self._read_exact(path, digest)
            if existing != content:
                raise ArchiveRefusal("DIGEST_MISMATCH")
            return
        _write_exclusive(path, content)
        stored = self._read_exact(path, digest)
        if stored != content:
            raise ArchiveRefusal("DIGEST_MISMATCH")

    def _index_path(self, body_digest: str) -> Path:
        return self.root / "index" / f"{_hex64(body_digest)}.json"

    def _require_same_index(self, envelope: ArchiveEnvelope) -> None:
        path = self._index_path(envelope.body_digest)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        if (
            stored.get("body_digest") != envelope.body_digest
            or stored.get("manifest_digest") != envelope.manifest_digest
        ):
            raise ArchiveRefusal("CONFLICT")

    def _write_index(self, envelope: ArchiveEnvelope) -> None:
        payload = {
            "schema": "doin.file_receipt.v1",
            "label": self.LABEL,
            "body_digest": envelope.body_digest,
            "manifest_digest": envelope.manifest_digest,
            "header_hash": envelope.header_hash,
            "anchor": envelope.anchor,
        }
        _write_exclusive(
            self._index_path(envelope.body_digest),
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )

    @staticmethod
    def _read_exact(path: Path, expected: str) -> bytes:
        if not path.exists():
            raise ArchiveRefusal("MISSING")
        if not path.is_file():
            raise ArchiveRefusal("UNREADABLE")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        if digest_bytes(data) != expected:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return data


def _hex64(digest: str) -> str:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ArchiveRefusal("NOT_A_SHA256")
    return digest


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        handle = os.open(path, flags, 0o644)
    except FileExistsError:
        return
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
