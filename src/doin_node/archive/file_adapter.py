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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doin_core.archive.body import (
    ArchiveEnvelope,
    ArchiveRefusal,
    VerifiedArchive,
    digest_bytes,
    verify_archive_bytes,
    verify_envelope,
)

LAKE_WRITE_FACT = (
    "data_lake_service.LakeBackend write_metrics stores a consumer report, not "
    "resource bytes. data_gov.resource_registration.ResourceRegistry.append writes "
    "one catalog JSON row and does not copy resource bytes. A governed byte store "
    "is a separate capability and is not a deployed lake. This file adapter is a "
    "disposable directory, not a lake."
)
MANIFEST_IDENTITY_RULE = (
    "BODY_DIGEST identifies the block body bytes and does not include candidates. "
    "MANIFEST_DIGEST identifies one binding of section digests, sizes and inventory. "
    "A later manifest for the same body is a successor: it is stored beside the first "
    "and does not overwrite it. Putting the same bytes again is a retry. "
    "A contradiction, the same digest with different bytes or an index that disagrees, "
    "is CONFLICT and is not a retry."
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
        verified = verify_envelope(envelope)
        self._store("objects", verified.body_digest, verified.body)
        self._store("manifests", verified.manifest_digest, verified.manifest)
        seen: set[str] = set()
        for section in verified.sections:
            if section.digest in seen:
                continue
            seen.add(section.digest)
            self._store("objects", section.digest, section.content)
        created = self._write_index(verified)
        if not created:
            self._require_same_index(verified)
        reloaded = self.load_verified(verified.manifest_digest)
        if reloaded.body != verified.body or reloaded.manifest != verified.manifest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return ArchiveReceipt(
            body_digest=verified.body_digest,
            manifest_digest=verified.manifest_digest,
            header_hash=verified.header_hash,
            anchor=verified.anchor,
            already_stored=not created,
            verified=True,
        )

    def put_governed(self, envelope: ArchiveEnvelope, store: Any, *, grant: str) -> ArchiveReceipt:
        """Write verified bytes through a governed store and read them back.

        An empty grant is refused before any write. This directory is not a lake,
        and a successful read-back is not a deployed lake.
        """
        if not isinstance(grant, str) or not grant:
            raise ArchiveRefusal("GRANT_REQUIRED")
        verified = verify_envelope(envelope)
        written: list[tuple[bytes, str]] = [
            (verified.body, verified.body_digest),
            (verified.manifest, verified.manifest_digest),
        ]
        for section in verified.sections:
            written.append((section.content, section.digest))
        seen: set[str] = set()
        for content, digest in written:
            if digest in seen:
                continue
            seen.add(digest)
            stored = store.put_bytes(content, grant=grant)
            if stored != digest:
                raise ArchiveRefusal("DIGEST_MISMATCH")
        sections = {
            section.digest: store.get_bytes(section.digest, grant=grant)
            for section in verified.sections
        }
        again = verify_archive_bytes(
            body=store.get_bytes(verified.body_digest, grant=grant),
            manifest=store.get_bytes(verified.manifest_digest, grant=grant),
            sections=sections,
        )
        if (
            again.body_digest != verified.body_digest
            or again.manifest_digest != verified.manifest_digest
            or again.chain_verified
        ):
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return ArchiveReceipt(
            body_digest=again.body_digest,
            manifest_digest=again.manifest_digest,
            header_hash=again.header_hash,
            anchor=again.anchor,
            already_stored=False,
            verified=True,
        )

    def read_body(self, digest: str) -> bytes:
        return self._read_exact(self.root / "objects" / _hex64(digest), digest)

    def read_manifest(self, digest: str) -> bytes:
        return self._read_exact(self.root / "manifests" / _hex64(digest), digest)

    def read_verified(self, receipt: ArchiveReceipt) -> tuple[bytes, bytes]:
        verified = self.load_verified(receipt.manifest_digest)
        if (
            verified.body_digest != receipt.body_digest
            or verified.header_hash != receipt.header_hash
            or verified.anchor != receipt.anchor
            or verified.chain_verified
        ):
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return verified.body, verified.manifest

    def load_verified(self, manifest_digest: str) -> VerifiedArchive:
        """Reconstruct one manifest from stored bytes. The envelope is not an input."""
        manifest = self.read_manifest(manifest_digest)
        try:
            parsed = json.loads(manifest)
        except ValueError as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        body_digest = parsed.get("body_digest") if isinstance(parsed, dict) else None
        if not isinstance(body_digest, str):
            raise ArchiveRefusal("DIGEST_MISMATCH")
        body = self.read_body(body_digest)
        sections: dict[str, bytes] = {}
        for section in parsed.get("sections") or []:
            if not isinstance(section, dict) or not isinstance(section.get("digest"), str):
                raise ArchiveRefusal("SECTION")
            sections[section["digest"]] = self.read_body(section["digest"])
        verified = verify_archive_bytes(body=body, manifest=manifest, sections=sections)
        if verified.manifest_digest != manifest_digest or verified.chain_verified:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return verified

    def recover(self) -> dict[str, Any]:
        """Report hash failures. A truncated file is not marked verified and is not deleted."""
        problems: list[dict[str, Any]] = []
        checked = 0
        for directory in ("objects", "manifests"):
            root = self.root / directory
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                checked += 1
                try:
                    data = path.read_bytes()
                except OSError:
                    problems.append({"name": path.name, "verified": False, "reason": "UNREADABLE"})
                    continue
                if digest_bytes(data) != path.name:
                    problems.append(
                        {
                            "name": path.name,
                            "verified": False,
                            "reason": "TRUNCATED_OR_MISMATCH",
                        }
                    )
        index = self.root / "index"
        if index.is_dir():
            for path in sorted(index.glob("*.json")):
                checked += 1
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    digest = payload.get("manifest_digest") if isinstance(payload, dict) else None
                    if digest != path.stem:
                        raise ArchiveRefusal("INDEX_NAME")
                    self.load_verified(digest)
                except (OSError, ValueError, ArchiveRefusal) as exc:
                    problems.append(
                        {"name": path.name, "verified": False, "reason": type(exc).__name__}
                    )
        return {
            "label": self.LABEL,
            "checked": checked,
            "verified": checked > 0 and not problems,
            "problems": problems,
        }

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

    def _index_path(self, manifest_digest: str) -> Path:
        return self.root / "index" / f"{_hex64(manifest_digest)}.json"

    def _require_same_index(self, verified: VerifiedArchive) -> None:
        path = self._index_path(verified.manifest_digest)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        if (
            stored.get("body_digest") != verified.body_digest
            or stored.get("manifest_digest") != verified.manifest_digest
            or stored.get("header_hash") != verified.header_hash
            or stored.get("anchor") != verified.anchor
        ):
            raise ArchiveRefusal("CONFLICT")

    def _write_index(self, verified: VerifiedArchive) -> bool:
        payload = {
            "schema": "doin.file_receipt.v1",
            "label": self.LABEL,
            "body_digest": verified.body_digest,
            "manifest_digest": verified.manifest_digest,
            "header_hash": verified.header_hash,
            "anchor": verified.anchor,
            "rule": "MANIFEST_IDENTITY",
        }
        return _write_exclusive(
            self._index_path(verified.manifest_digest),
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


def _write_exclusive(path: Path, content: bytes) -> bool:
    """Publish ``content`` atomically. A partial file never appears at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".part-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
