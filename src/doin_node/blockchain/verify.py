"""One authoritative full-chain verifier (findings 202-203, WP2).

Executes ten ordered checks against a SQLite chain database and returns a
typed :class:`~doin_core.models.verification.ChainVerificationReport`:

 1. SQLite integrity / foreign-key checks;
 2. contiguous block indices from a deterministic genesis;
 3. exact configured genesis hash / chain ID;
 4. each row's block hash against its canonical header;
 5. each ``previous_hash`` against the preceding verified block;
 6. transaction count and contiguous ``tx_index`` values;
 7. every transaction-content hash;
 8. every Merkle root;
 9. snapshots against existing block index/hash;
10. metadata height/tip against verified rows.

Pruning semantics: a chain whose transaction bodies were pruned is NEVER
``fully_verified``. When (and only when) the recorded checkpoint
commitment and the fully retained suffix verify, the result is a typed
``verified_suffix_from_checkpoint``. Absent or contradictory pruning
provenance is ``refused`` / ``failed`` — never success.

The verifier opens the database read-only and never mutates it, so it can
be pointed at forensic *copies* of fleet databases:

    python -m doin_node.blockchain.verify --db <path> \\
        [--expect-chain-id <id>] [--expect-genesis <hash>]

Exit codes: 0 fully_verified, 10 verified_suffix_from_checkpoint,
2 failed, 3 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models.transaction import TX_ID_PATTERN, compute_transaction_id
from doin_core.models.verification import (
    ChainVerificationOutcome,
    ChainVerificationReport,
    CheckStatus,
    FailureCoordinate,
    VerificationCheck,
    VerifiedSuffixFromCheckpoint,
)
from doin_core.protocol.messages import PROTOCOL_VERSION

from doin_node.storage.chaindb import (
    META_CHAIN_ID,
    META_CHECKPOINT_HASH,
    META_CHECKPOINT_INDEX,
    META_GENESIS_HASH,
    META_HEIGHT,
    META_PRUNED_BEFORE,
    META_TIP_HASH,
)

GENESIS_PREVIOUS_HASH = "0" * 64
EMPTY_MERKLE_ROOT = "0" * 64
GENESIS_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()

#: Ordered (number, stable name) for the ten checks.
CHECK_NAMES: list[tuple[int, str]] = [
    (1, "sqlite_integrity"),
    (2, "contiguous_indices_from_deterministic_genesis"),
    (3, "configured_genesis_and_chain_id"),
    (4, "block_hashes_match_canonical_headers"),
    (5, "previous_hash_chain"),
    (6, "transaction_count_and_contiguous_tx_index"),
    (7, "transaction_content_hashes"),
    (8, "merkle_roots"),
    (9, "snapshots_match_blocks"),
    (10, "metadata_height_tip"),
]


class ChainStartupRefused(Exception):
    """Typed startup refusal: the chain database did not verify.

    Raised by the runtime BEFORE any network, sync, optimization,
    evaluation, dashboard or OLAP side effect. Carries the full typed
    report; the message stays payload-free.
    """

    def __init__(self, report: ChainVerificationReport) -> None:
        coord = report.first_failure
        where = (
            f" at block={coord.block_index} tx_index={coord.tx_index}"
            f" ({coord.check_name}: {coord.reason})"
            if coord
            else ""
        )
        super().__init__(
            f"chain verification {report.outcome.value}{where} — "
            f"db={report.db_path}"
        )
        self.report = report


def _header_hash_from_row(row: sqlite3.Row) -> str:
    """Recompute the canonical block-header hash from raw stored fields.

    Mirrors ``BlockHeader.compute_hash`` byte-for-byte: the stored
    timestamp string IS the canonical ``isoformat()`` output, so no model
    construction (and no parser) sits between the stored bytes and the
    hash.
    """
    payload = json.dumps(
        {
            "index": row["block_index"],
            "previous_hash": row["previous_hash"],
            "timestamp": row["timestamp"],
            "merkle_root": row["merkle_root"],
            "generator_id": row["generator_id"],
            "weighted_performance_sum": row["weighted_performance_sum"],
            "threshold": row["threshold"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class _Verifier:
    """Single verification run over one read-only SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        expected_chain_id: str | None,
        expected_genesis_hash: str | None,
    ) -> None:
        self.conn = conn
        self.report = ChainVerificationReport(
            outcome=ChainVerificationOutcome.FAILED,
            db_path=db_path,
            protocol_version=PROTOCOL_VERSION,
            expected_chain_id=expected_chain_id,
            expected_genesis_hash=expected_genesis_hash,
        )
        self.blocks: list[sqlite3.Row] = []
        self.txs_by_block: dict[int, list[sqlite3.Row]] = {}
        self.pruned_body_blocks: list[int] = []
        self.pruning_meta: dict[str, str | None] = {}
        self._stopped = False

    # ── plumbing ─────────────────────────────────────────────────

    def _meta(self, key: str) -> str | None:
        try:
            row = self.conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error:
            return None
        return row["value"] if row else None

    def _record(
        self,
        number: int,
        name: str,
        status: CheckStatus,
        detail: str = "",
        block_index: int | None = None,
        tx_index: int | None = None,
    ) -> bool:
        """Record a check result. Returns True when verification continues."""
        self.report.checks.append(
            VerificationCheck(
                number=number,
                name=name,
                status=status,
                detail=detail,
                first_failing_block=block_index,
                first_failing_tx=tx_index,
            )
        )
        if status in (CheckStatus.FAIL, CheckStatus.UNAVAILABLE):
            self.report.first_failure = FailureCoordinate(
                check_number=number,
                check_name=name,
                block_index=block_index,
                tx_index=tx_index,
                reason=detail,
            )
            self.report.outcome = (
                ChainVerificationOutcome.REFUSED
                if status is CheckStatus.UNAVAILABLE
                else ChainVerificationOutcome.FAILED
            )
            self._stopped = True
            return False
        return True

    def _skip_remaining(self, from_number: int) -> None:
        for number, name in CHECK_NAMES:
            if number >= from_number:
                self.report.checks.append(
                    VerificationCheck(
                        number=number,
                        name=name,
                        status=CheckStatus.SKIPPED,
                        detail="not executed: an earlier check failed",
                    )
                )

    # ── the ten ordered checks ───────────────────────────────────

    def run(self) -> ChainVerificationReport:
        checks = [
            self._check_1_sqlite,
            self._check_2_contiguous_from_genesis,
            self._check_3_configured_identity,
            self._check_4_block_hashes,
            self._check_5_previous_hash_chain,
            self._check_6_tx_count_and_index,
            self._check_7_tx_content_hashes,
            self._check_8_merkle_roots,
            self._check_9_snapshots,
            self._check_10_metadata_height_tip,
        ]
        for i, check in enumerate(checks, start=1):
            check()
            if self._stopped:
                self._skip_remaining(i + 1)
                break
        else:
            if self.pruned_body_blocks:
                pruned_before = int(self.pruning_meta[META_PRUNED_BEFORE])  # type: ignore[arg-type]
                checkpoint_index = int(self.pruning_meta[META_CHECKPOINT_INDEX])  # type: ignore[arg-type]
                tip = self.blocks[-1]
                self.report.outcome = (
                    ChainVerificationOutcome.VERIFIED_SUFFIX_FROM_CHECKPOINT
                )
                self.report.verified_suffix = VerifiedSuffixFromCheckpoint(
                    checkpoint_block_index=checkpoint_index,
                    checkpoint_block_hash=str(
                        self.pruning_meta[META_CHECKPOINT_HASH]
                    ),
                    suffix_start_index=pruned_before,
                    suffix_end_index=tip["block_index"],
                    suffix_tip_hash=tip["hash"],
                    pruned_body_blocks=len(self.pruned_body_blocks),
                )
            else:
                self.report.outcome = ChainVerificationOutcome.FULLY_VERIFIED

        if self.blocks:
            tip = self.blocks[-1]
            self.report.height = tip["block_index"] + 1
            self.report.tip_index = tip["block_index"]
            self.report.tip_hash = tip["hash"]
        self.report.chain_id = self._meta(META_CHAIN_ID) or ""
        self.report.genesis_hash = self._meta(META_GENESIS_HASH) or ""
        self.report.finished_at = datetime.now(timezone.utc)
        return self.report

    def _check_1_sqlite(self) -> None:
        number, name = CHECK_NAMES[0]
        try:
            rows = self.conn.execute("PRAGMA integrity_check").fetchall()
            results = [r[0] for r in rows]
            if results != ["ok"]:
                self._record(
                    number, name, CheckStatus.FAIL,
                    f"integrity_check: {'; '.join(results[:5])}",
                )
                return
            fk = self.conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk:
                first = fk[0]
                self._record(
                    number, name, CheckStatus.FAIL,
                    f"foreign_key_check: {len(fk)} violation(s), first in "
                    f"table {first[0]!r} rowid {first[1]}",
                )
                return
            for table in ("metadata", "blocks", "transactions"):
                present = self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if present is None:
                    self._record(
                        number, name, CheckStatus.FAIL,
                        f"required table {table!r} is missing",
                    )
                    return
        except sqlite3.Error as e:
            self._record(
                number, name, CheckStatus.FAIL, f"sqlite error: {e}"
            )
            return
        self._record(number, name, CheckStatus.PASS)

    def _check_2_contiguous_from_genesis(self) -> None:
        number, name = CHECK_NAMES[1]
        self.blocks = self.conn.execute(
            "SELECT * FROM blocks ORDER BY block_index"
        ).fetchall()
        if not self.blocks:
            self._record(
                number, name, CheckStatus.FAIL, "chain has no blocks"
            )
            return
        for position, row in enumerate(self.blocks):
            if row["block_index"] != position:
                self._record(
                    number, name, CheckStatus.FAIL,
                    f"block indices not contiguous: expected {position}, "
                    f"found {row['block_index']}",
                    block_index=row["block_index"],
                )
                return
        genesis = self.blocks[0]
        deterministic = (
            genesis["previous_hash"] == GENESIS_PREVIOUS_HASH
            and genesis["merkle_root"] == EMPTY_MERKLE_ROOT
            and genesis["timestamp"] == GENESIS_TIMESTAMP
            and genesis["weighted_performance_sum"] == 0.0
            and genesis["threshold"] == 0.0
            and genesis["tx_count"] == 0
        )
        if not deterministic:
            self._record(
                number, name, CheckStatus.FAIL,
                "genesis block is not the deterministic genesis "
                "(previous_hash/merkle_root/timestamp/zero fields)",
                block_index=0,
            )
            return
        if genesis["hash"] != _header_hash_from_row(genesis):
            self._record(
                number, name, CheckStatus.FAIL,
                "genesis stored hash does not match its canonical header",
                block_index=0,
            )
            return
        self._record(
            number, name, CheckStatus.PASS,
            f"{len(self.blocks)} contiguous blocks from deterministic genesis",
        )

    def _check_3_configured_identity(self) -> None:
        number, name = CHECK_NAMES[2]
        genesis_hash = self.blocks[0]["hash"]
        attested_genesis = self._meta(META_GENESIS_HASH)
        attested_chain_id = self._meta(META_CHAIN_ID)

        if attested_genesis is not None and attested_genesis != genesis_hash:
            self._record(
                number, name, CheckStatus.FAIL,
                "metadata genesis_hash attestation does not match the "
                "stored genesis block",
                block_index=0,
            )
            return
        expected_genesis = self.report.expected_genesis_hash
        if expected_genesis is not None and expected_genesis != genesis_hash:
            self._record(
                number, name, CheckStatus.FAIL,
                f"genesis hash {genesis_hash[:16]}… does not match the "
                f"configured genesis {expected_genesis[:16]}…",
                block_index=0,
            )
            return
        expected_chain_id = self.report.expected_chain_id
        if expected_chain_id is not None:
            if attested_chain_id is None:
                # Absence of required provenance is refusal, not success.
                self._record(
                    number, name, CheckStatus.UNAVAILABLE,
                    "a chain ID is expected but the database carries no "
                    "chain_id attestation",
                )
                return
            if attested_chain_id != expected_chain_id:
                self._record(
                    number, name, CheckStatus.FAIL,
                    f"chain_id {attested_chain_id!r} does not match the "
                    f"configured {expected_chain_id!r}",
                )
                return
        detail = f"genesis={genesis_hash[:16]}…"
        if attested_chain_id:
            detail += f" chain_id={attested_chain_id}"
        elif expected_chain_id is None:
            detail += " (no chain-id expectation or attestation)"
        self._record(number, name, CheckStatus.PASS, detail)

    def _check_4_block_hashes(self) -> None:
        number, name = CHECK_NAMES[3]
        for row in self.blocks:
            if row["hash"] != _header_hash_from_row(row):
                self._record(
                    number, name, CheckStatus.FAIL,
                    "stored block hash does not match its canonical header",
                    block_index=row["block_index"],
                )
                return
        self._record(number, name, CheckStatus.PASS)

    def _check_5_previous_hash_chain(self) -> None:
        number, name = CHECK_NAMES[4]
        for i in range(1, len(self.blocks)):
            if self.blocks[i]["previous_hash"] != self.blocks[i - 1]["hash"]:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "previous_hash does not match the preceding verified block",
                    block_index=self.blocks[i]["block_index"],
                )
                return
        self._record(number, name, CheckStatus.PASS)

    def _check_6_tx_count_and_index(self) -> None:
        number, name = CHECK_NAMES[5]
        tx_rows = self.conn.execute(
            "SELECT block_index, tx_index, tx_id, tx_type, domain_id, "
            "peer_id, payload, timestamp FROM transactions "
            "ORDER BY block_index, tx_index"
        ).fetchall()
        self.txs_by_block = {}
        for row in tx_rows:
            self.txs_by_block.setdefault(row["block_index"], []).append(row)

        self.pruning_meta = {
            META_PRUNED_BEFORE: self._meta(META_PRUNED_BEFORE),
            META_CHECKPOINT_INDEX: self._meta(META_CHECKPOINT_INDEX),
            META_CHECKPOINT_HASH: self._meta(META_CHECKPOINT_HASH),
        }
        pruned_before_raw = self.pruning_meta[META_PRUNED_BEFORE]
        pruned_before = int(pruned_before_raw) if pruned_before_raw else None

        known_blocks = {row["block_index"] for row in self.blocks}
        for block_index in self.txs_by_block:
            if block_index not in known_blocks:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "transaction rows reference a block that does not exist",
                    block_index=block_index,
                )
                return

        self.pruned_body_blocks = []
        for row in self.blocks:
            block_index = row["block_index"]
            declared = row["tx_count"]
            present = self.txs_by_block.get(block_index, [])
            if len(present) == declared:
                indices = [t["tx_index"] for t in present]
                if indices != list(range(declared)):
                    expected = next(
                        (i for i, v in enumerate(indices) if v != i), declared
                    )
                    self._record(
                        number, name, CheckStatus.FAIL,
                        "tx_index values are not contiguous "
                        f"(expected 0..{declared - 1}, found {indices})",
                        block_index=block_index,
                        tx_index=expected,
                    )
                    return
                continue
            if not present and declared > 0:
                # Bodies absent entirely — pruning, if and only if the
                # recorded provenance says so. Otherwise: corruption.
                if pruned_before is None:
                    self._record(
                        number, name, CheckStatus.FAIL,
                        f"block declares {declared} transaction(s) but no "
                        "rows are stored and no pruning provenance exists",
                        block_index=block_index,
                        tx_index=0,
                    )
                    return
                if block_index >= pruned_before:
                    self._record(
                        number, name, CheckStatus.FAIL,
                        f"block declares {declared} transaction(s) but no "
                        f"rows are stored, and it is inside the retained "
                        f"suffix (pruned_before_index={pruned_before})",
                        block_index=block_index,
                        tx_index=0,
                    )
                    return
                self.pruned_body_blocks.append(block_index)
                continue
            self._record(
                number, name, CheckStatus.FAIL,
                f"transaction count mismatch: header declares {declared}, "
                f"{len(present)} row(s) stored",
                block_index=block_index,
                tx_index=len(present),
            )
            return

        if self.pruned_body_blocks:
            # The checkpoint commitment itself must verify before any
            # pruned prefix is tolerated.
            checkpoint_index_raw = self.pruning_meta[META_CHECKPOINT_INDEX]
            checkpoint_hash = self.pruning_meta[META_CHECKPOINT_HASH]
            if checkpoint_index_raw is None or not checkpoint_hash:
                self._record(
                    number, name, CheckStatus.UNAVAILABLE,
                    "transaction bodies are pruned but the checkpoint "
                    "commitment metadata is absent",
                    block_index=self.pruned_body_blocks[0],
                )
                return
            checkpoint_index = int(checkpoint_index_raw)
            if checkpoint_index != pruned_before - 1:
                self._record(
                    number, name, CheckStatus.FAIL,
                    f"checkpoint_block_index {checkpoint_index} does not "
                    f"commit to pruned_before_index {pruned_before}",
                    block_index=checkpoint_index,
                )
                return
            if checkpoint_index >= len(self.blocks):
                self._record(
                    number, name, CheckStatus.FAIL,
                    "checkpoint block is not stored",
                    block_index=checkpoint_index,
                )
                return
            checkpoint_row = self.blocks[checkpoint_index]
            if checkpoint_row["hash"] != checkpoint_hash:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "checkpoint commitment hash does not match the stored "
                    "checkpoint block",
                    block_index=checkpoint_index,
                )
                return

        detail = ""
        if self.pruned_body_blocks:
            detail = (
                f"{len(self.pruned_body_blocks)} block(s) with pruned bodies "
                f"below checkpoint (pruned_before_index={pruned_before})"
            )
        self._record(number, name, CheckStatus.PASS, detail)

    def _check_7_tx_content_hashes(self) -> None:
        number, name = CHECK_NAMES[6]
        for block_index in sorted(self.txs_by_block):
            for tx in self.txs_by_block[block_index]:
                tx_id = tx["tx_id"]
                if not TX_ID_PATTERN.fullmatch(tx_id):
                    self._record(
                        number, name, CheckStatus.FAIL,
                        "transaction ID is not 64 lowercase hex characters",
                        block_index=block_index,
                        tx_index=tx["tx_index"],
                    )
                    return
                try:
                    payload = json.loads(tx["payload"])
                except json.JSONDecodeError:
                    self._record(
                        number, name, CheckStatus.FAIL,
                        "transaction payload is not valid JSON",
                        block_index=block_index,
                        tx_index=tx["tx_index"],
                    )
                    return
                expected = compute_transaction_id(
                    tx_type=tx["tx_type"],
                    domain_id=tx["domain_id"],
                    peer_id=tx["peer_id"],
                    payload=payload,
                    timestamp=tx["timestamp"],
                )
                if tx_id != expected:
                    self._record(
                        number, name, CheckStatus.FAIL,
                        "transaction content does not match its ID",
                        block_index=block_index,
                        tx_index=tx["tx_index"],
                    )
                    return
        self._record(number, name, CheckStatus.PASS)

    def _check_8_merkle_roots(self) -> None:
        number, name = CHECK_NAMES[7]
        pruned = set(self.pruned_body_blocks)
        for row in self.blocks:
            block_index = row["block_index"]
            if block_index in pruned:
                continue  # cannot recompute — tolerated only under checkpoint
            tx_ids = [t["tx_id"] for t in self.txs_by_block.get(block_index, [])]
            if compute_merkle_root(tx_ids) != row["merkle_root"]:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "Merkle root does not match the recomputed transaction "
                    "hashes",
                    block_index=block_index,
                )
                return
        detail = (
            f"{len(pruned)} pruned block(s) excluded from Merkle "
            "recomputation" if pruned else ""
        )
        self._record(number, name, CheckStatus.PASS, detail)

    def _check_9_snapshots(self) -> None:
        number, name = CHECK_NAMES[8]
        try:
            snapshots = self.conn.execute(
                "SELECT block_index, block_hash FROM state_snapshots "
                "ORDER BY block_index"
            ).fetchall()
        except sqlite3.Error:
            snapshots = []
        by_index = {row["block_index"]: row for row in self.blocks}
        for snap in snapshots:
            block = by_index.get(snap["block_index"])
            if block is None:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "snapshot references a block index that does not exist",
                    block_index=snap["block_index"],
                )
                return
            if block["hash"] != snap["block_hash"]:
                self._record(
                    number, name, CheckStatus.FAIL,
                    "snapshot block hash does not match the verified block",
                    block_index=snap["block_index"],
                )
                return
        self._record(
            number, name, CheckStatus.PASS,
            f"{len(snapshots)} snapshot(s) verified",
        )

    def _check_10_metadata_height_tip(self) -> None:
        number, name = CHECK_NAMES[9]
        tip = self.blocks[-1]
        claimed_height = self._meta(META_HEIGHT)
        claimed_tip = self._meta(META_TIP_HASH)
        if claimed_height is None and claimed_tip is None:
            self._record(
                number, name, CheckStatus.PASS,
                "no metadata height/tip claim recorded (legacy database); "
                f"derived height={tip['block_index'] + 1} from verified rows",
            )
            return
        if claimed_height is not None and int(claimed_height) != tip["block_index"] + 1:
            self._record(
                number, name, CheckStatus.FAIL,
                f"metadata height {claimed_height} does not match verified "
                f"height {tip['block_index'] + 1}",
                block_index=tip["block_index"],
            )
            return
        if claimed_tip is not None and claimed_tip != tip["hash"]:
            self._record(
                number, name, CheckStatus.FAIL,
                "metadata tip_hash does not match the verified tip",
                block_index=tip["block_index"],
            )
            return
        self._record(number, name, CheckStatus.PASS)


def _open_read_only(db_path: str | Path) -> sqlite3.Connection:
    """Open the database strictly for reading.

    Prefers SQLite's read-only URI mode; falls back to a plain connection
    (still used only for SELECT/PRAGMA reads) when the platform cannot
    honor mode=ro for a WAL database. The verifier never writes.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"chain database not found: {path}")
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, isolation_level=None
        )
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def verify_chain_db(
    db_path: str | Path,
    *,
    expected_chain_id: str | None = None,
    expected_genesis_hash: str | None = None,
) -> ChainVerificationReport:
    """Run the ten ordered checks against a chain database (read-only)."""
    conn = _open_read_only(db_path)
    try:
        return _Verifier(
            conn, str(db_path), expected_chain_id, expected_genesis_hash
        ).run()
    finally:
        conn.close()


# ── CLI ──────────────────────────────────────────────────────────────

_EXIT_CODES = {
    ChainVerificationOutcome.FULLY_VERIFIED: 0,
    ChainVerificationOutcome.VERIFIED_SUFFIX_FROM_CHECKPOINT: 10,
    ChainVerificationOutcome.FAILED: 2,
    ChainVerificationOutcome.REFUSED: 3,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m doin_node.blockchain.verify",
        description="Authoritative DOIN chain verifier (read-only).",
    )
    parser.add_argument("--db", required=True, help="Path to a chain SQLite file (a copy for forensics)")
    parser.add_argument("--expect-chain-id", default=None, help="Refuse unless the DB attests exactly this chain ID")
    parser.add_argument("--expect-genesis", default=None, help="Fail unless the genesis hash equals this value")
    args = parser.parse_args(argv)

    try:
        report = verify_chain_db(
            args.db,
            expected_chain_id=args.expect_chain_id,
            expected_genesis_hash=args.expect_genesis,
        )
    except FileNotFoundError as e:
        print(json.dumps({"outcome": "refused", "error": str(e)}))
        return _EXIT_CODES[ChainVerificationOutcome.REFUSED]

    print(report.model_dump_json(indent=2))
    summary = f"outcome={report.outcome.value} height={report.height}"
    if report.first_failure:
        summary += (
            f" first_failure=check{report.first_failure.check_number}"
            f"/{report.first_failure.check_name}"
            f" block={report.first_failure.block_index}"
            f" tx={report.first_failure.tx_index}"
        )
    print(summary, file=sys.stderr)
    return _EXIT_CODES[report.outcome]


if __name__ == "__main__":
    raise SystemExit(main())
