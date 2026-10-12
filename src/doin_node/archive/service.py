"""Disposable publication through a governed byte store.

The store is ``GovernedByteStore``: content-addressed bytes, a caller grant,
and a hash check. Registration does not authorize a put or a get. This module
does not open Postgres, does not add an HTTP route, and does not claim either
the reference directory or the byte store is a lake.

Order: durable write, then a pending reference, then a hash read, then an
accepted reference, then projection. A refused write stops before the pending
reference. The byte store and the warehouse do not share a transaction, so a
refusal does not delete digests that another manifest may already share.
"""

from __future__ import annotations

import json
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
from doin_node.archive.file_adapter import FileArchiveAdapter, _hex64, _write_exclusive
from doin_node.archive.warehouse import DisposableWarehouse, project_from_reference

BYTE_STORE_LABEL = "DISPOSABLE_BYTE_STORE_NOT_DEPLOYED_LAKE"
_REFERENCE_SCHEMA = "doin.service_reference.v1"
_STORE_REFUSALS = frozenset(
    {
        "BYTES_REQUIRED",
        "GRANT_REFUSED",
        "GRANT_REQUIRED",
        "HASH_MISMATCH",
        "MISSING",
        "NOT_A_SHA256",
        "POSTGRES_REFUSED",
    }
)


class DisposableServiceAdapter:
    """Publish one envelope through a disposable byte store, then project it.

    The reference directory keeps receipts only. Body bytes stay in the store.
    ``chain_verified`` is not set by a successful read.
    """

    def __init__(self, store: Any, reference_root: str | Path, *, grant: str) -> None:
        if not isinstance(grant, str) or not grant:
            raise ArchiveRefusal("GRANT_REQUIRED")
        described = _described(store)
        if described.get("label") != BYTE_STORE_LABEL:
            raise ArchiveRefusal("BYTE_STORE_REQUIRED")
        if described.get("deployed_lake") is not False:
            raise ArchiveRefusal("DEPLOYED_LAKE_REFUSED")
        if described.get("registration_authorizes_delivery") is not False:
            raise ArchiveRefusal("REGISTRATION_DOES_NOT_AUTHORIZE")
        if not callable(getattr(store, "put_bytes", None)):
            raise ArchiveRefusal("BYTE_STORE_REQUIRED")
        if not callable(getattr(store, "get_bytes", None)):
            raise ArchiveRefusal("BYTE_STORE_REQUIRED")
        self.store = store
        self.root = Path(reference_root)
        self.grant = grant

    def publish_and_project(
        self,
        warehouse: DisposableWarehouse,
        envelope: ArchiveEnvelope,
        *,
        experiment_id: str,
    ) -> int:
        """Write, reference, hash-read, accept, then project.

        Projection uses ``project_from_reference``, so the inserted rows are
        parsed again from the bytes just read.
        """
        verified = verify_envelope(envelope)
        if verified.chain_verified:
            raise ArchiveRefusal("CHAIN_NOT_VERIFIED")
        self._durable_write(verified)
        self._write_reference("pending", verified, accepted=False)
        checked = self._verified_read(verified.manifest_digest)
        if (
            checked.manifest_digest != verified.manifest_digest
            or checked.body_digest != verified.body_digest
            or checked.header_hash != verified.header_hash
            or checked.anchor != verified.anchor
            or checked.chain_verified
        ):
            raise ArchiveRefusal("DIGEST_MISMATCH")
        self._write_reference("accepted", checked, accepted=True)
        return project_from_reference(
            warehouse,
            self,
            manifest_digest=verified.manifest_digest,
            experiment_id=experiment_id,
        )

    def load_verified(self, manifest_digest: str) -> VerifiedArchive:
        """Hash-read an accepted reference. A pending reference is not enough."""
        accepted = self._load_json("accepted", manifest_digest)
        if accepted.get("accepted") is not True:
            raise ArchiveRefusal("REFERENCE_NOT_ACCEPTED")
        verified = self._verified_read(manifest_digest)
        if (
            accepted.get("body_digest") != verified.body_digest
            or accepted.get("manifest_digest") != verified.manifest_digest
            or accepted.get("header_hash") != verified.header_hash
            or accepted.get("anchor") != verified.anchor
            or verified.chain_verified
        ):
            raise ArchiveRefusal("CONFLICT")
        return verified

    def _durable_write(self, verified: VerifiedArchive) -> None:
        """Put each distinct object. Return only when every digest matched.

        A later refusal leaves earlier objects in place and writes no reference.
        """
        seen: set[str] = set()
        for content, digest in _objects(verified):
            if digest in seen:
                continue
            seen.add(digest)
            if not isinstance(content, bytes) or digest_bytes(content) != digest:
                raise ArchiveRefusal("DIGEST_MISMATCH")
            stored = self._call("put_bytes", content, grant=self.grant)
            if stored != digest:
                raise ArchiveRefusal("DIGEST_MISMATCH")

    def _verified_read(self, manifest_digest: str) -> VerifiedArchive:
        pending = self._load_json("pending", manifest_digest)
        manifest = self._get(manifest_digest)
        if digest_bytes(manifest) != manifest_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        try:
            parsed = json.loads(manifest)
        except ValueError as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        body_digest = parsed.get("body_digest") if isinstance(parsed, dict) else None
        if not isinstance(body_digest, str):
            raise ArchiveRefusal("DIGEST_MISMATCH")
        body = self._get(body_digest)
        sections: dict[str, bytes] = {}
        for section in parsed.get("sections") or []:
            if not isinstance(section, dict) or not isinstance(section.get("digest"), str):
                raise ArchiveRefusal("SECTION")
            sections[section["digest"]] = self._get(section["digest"])
        verified = verify_archive_bytes(body=body, manifest=manifest, sections=sections)
        if verified.chain_verified or verified.manifest_digest != manifest_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        if (
            pending.get("body_digest") != verified.body_digest
            or pending.get("header_hash") != verified.header_hash
            or pending.get("anchor") != verified.anchor
            or pending.get("manifest_digest") != manifest_digest
        ):
            raise ArchiveRefusal("CONFLICT")
        return verified

    def _get(self, digest: str) -> bytes:
        blob = self._call("get_bytes", digest, grant=self.grant)
        if isinstance(blob, bytearray):
            blob = bytes(blob)
        if not isinstance(blob, bytes):
            raise ArchiveRefusal("DIGEST_REQUIRES_BYTES")
        if digest_bytes(blob) != digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        return blob

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self.store, method)(*args, **kwargs)
        except ArchiveRefusal:
            raise
        except Exception as exc:
            reason = str(exc)
            if reason in _STORE_REFUSALS:
                raise ArchiveRefusal(reason) from exc
            raise

    def _write_reference(
        self,
        kind: str,
        verified: VerifiedArchive,
        *,
        accepted: bool,
    ) -> None:
        if kind not in ("pending", "accepted"):
            raise ArchiveRefusal("REFERENCE")
        payload = {
            "accepted": accepted,
            "anchor": verified.anchor,
            "body_digest": verified.body_digest,
            "byte_store_label": BYTE_STORE_LABEL,
            "header_hash": verified.header_hash,
            "label": FileArchiveAdapter.LABEL,
            "manifest_digest": verified.manifest_digest,
            "schema": _REFERENCE_SCHEMA,
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if accepted and b'"accepted":true' not in content:
            raise ArchiveRefusal("REFERENCE")
        if not accepted and b'"accepted":false' not in content:
            raise ArchiveRefusal("REFERENCE")
        path = self.root / kind / f"{_hex64(verified.manifest_digest)}.json"
        if not _write_exclusive(path, content):
            try:
                stored = path.read_bytes()
            except OSError as exc:
                raise ArchiveRefusal("UNREADABLE") from exc
            if stored != content:
                raise ArchiveRefusal("CONFLICT")

    def _load_json(self, kind: str, manifest_digest: str) -> dict[str, Any]:
        path = self.root / kind / f"{_hex64(manifest_digest)}.json"
        if not path.is_file():
            raise ArchiveRefusal("MISSING")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArchiveRefusal("UNREADABLE") from exc
        if not isinstance(payload, dict):
            raise ArchiveRefusal("UNREADABLE")
        if payload.get("schema") != _REFERENCE_SCHEMA:
            raise ArchiveRefusal("UNREADABLE")
        if payload.get("manifest_digest") != manifest_digest:
            raise ArchiveRefusal("DIGEST_MISMATCH")
        if payload.get("label") != FileArchiveAdapter.LABEL:
            raise ArchiveRefusal("CONFLICT")
        if payload.get("byte_store_label") != BYTE_STORE_LABEL:
            raise ArchiveRefusal("CONFLICT")
        return payload


def _described(store: Any) -> dict[str, Any]:
    describe = getattr(store, "describe", None)
    if not callable(describe):
        raise ArchiveRefusal("BYTE_STORE_REQUIRED")
    described = describe()
    if not isinstance(described, dict):
        raise ArchiveRefusal("BYTE_STORE_REQUIRED")
    return described


def _objects(verified: VerifiedArchive) -> list[tuple[bytes, str]]:
    items = [
        (verified.body, verified.body_digest),
        (verified.manifest, verified.manifest_digest),
    ]
    for section in verified.sections:
        items.append((section.content, section.digest))
    return items
