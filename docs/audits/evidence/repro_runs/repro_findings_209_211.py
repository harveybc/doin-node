"""Findings 209-211 reproduction: metadata pair, append binding, chain identity.

Runs the auditor's counterexamples against a given doin-core/doin-node source
tree (env vars DOIN_CORE_SRC / DOIN_NODE_SRC) using TEMPORARY SQLite DBs.
Never touches any real chain database.

Probes:
  209  (a) delete metadata tip_hash only   -> verify outcome / check 10
       (b) delete metadata height only     -> verify outcome / check 10
       (c) height = 'not-an-int'           -> typed report or raw exception?
       (d) corrupted (height, tip_hash) pair -> verify outcome / check 10
  210  full verify -> tamper a HISTORICAL tx row via a SECOND connection
       -> append the next valid block: ACCEPTED (before) or typed refusal
       (after); then a subsequent full verify.
  211  materialize a real fleet shared-population example without
       chain_id/genesis_hash: accepted (before) or typed refusal (after);
       materialize the canonical identity template (after only).

Output: one JSON object on stdout.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CORE_SRC = os.environ["DOIN_CORE_SRC"]
NODE_SRC = os.environ["DOIN_NODE_SRC"]
sys.path.insert(0, NODE_SRC)
sys.path.insert(0, CORE_SRC)

# Purge any pre-imported/installed copies so the env vars win.
for mod in list(sys.modules):
    if mod.startswith(("doin_core", "doin_node")):
        del sys.modules[mod]

from doin_core.crypto.hashing import compute_merkle_root  # noqa: E402
from doin_core.models.block import Block, BlockHeader  # noqa: E402
from doin_core.models.transaction import Transaction, TransactionType  # noqa: E402
from doin_node.blockchain.verify import verify_chain_db  # noqa: E402
from doin_node.storage.chaindb import ChainDB  # noqa: E402

result: dict[str, object] = {
    "run": os.environ.get("RUN_LABEL", "unlabeled"),
    "doin_core_src": CORE_SRC,
    "doin_node_src": NODE_SRC,
    "utc": datetime.now(timezone.utc).isoformat(),
}

TS = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _tx(n: int) -> Transaction:
    return Transaction(
        tx_type=TransactionType.OPTIMAE_ANNOUNCED,
        domain_id="repro-domain",
        peer_id=f"peer-{n}",
        payload={"n": n},
        timestamp=TS,
    )


def _block_on(prev: Block, txs: list[Transaction]) -> Block:
    header = BlockHeader(
        index=prev.header.index + 1,
        previous_hash=prev.hash,
        timestamp=TS,
        merkle_root=compute_merkle_root([tx.id for tx in txs]),
        generator_id="repro-generator",
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    return Block(header=header, transactions=txs)


def _build_chain(db_path: Path, n_blocks: int = 2) -> None:
    db = ChainDB(db_path)
    db.open()
    prev = db.initialize("genesis")
    for i in range(1, n_blocks + 1):
        block = _block_on(prev, [_tx(i * 100), _tx(i * 100 + 1)])
        db.append_block(block)
        prev = block
    db.close()


def _second_connection_exec(db_path: Path, stmt: str, params: tuple = ()) -> None:
    """Mutate the DB through a SECOND sqlite3 connection (as the auditor did)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(stmt, params)
    conn.commit()
    conn.close()


def _verify_probe(db_path: Path) -> dict[str, object]:
    """Run the verifier; report either the typed outcome or a raw exception."""
    try:
        report = verify_chain_db(db_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "raised": f"{type(exc).__name__}: {exc}",
            "typed_report": False,
        }
    check10 = next((c for c in report.checks if c.number == 10), None)
    return {
        "typed_report": True,
        "outcome": report.outcome.value,
        "check_10_status": check10.status.value if check10 else None,
        "check_10_detail": check10.detail if check10 else None,
        "first_failure": (
            report.first_failure.model_dump() if report.first_failure else None
        ),
    }


with tempfile.TemporaryDirectory(prefix="findings209-211-") as tmp:
    base = Path(tmp) / "base.sqlite"
    _build_chain(base)

    # ── 209: metadata (height, tip_hash) pair ────────────────────────
    f209: dict[str, object] = {}
    for label, stmts in {
        "missing_tip_hash_only": [
            ("DELETE FROM metadata WHERE key = 'tip_hash'", ()),
        ],
        "missing_height_only": [
            ("DELETE FROM metadata WHERE key = 'height'", ()),
        ],
        "non_integer_height": [
            ("UPDATE metadata SET value = 'not-an-int' WHERE key = 'height'", ()),
        ],
        "corrupted_metadata_pair": [
            ("UPDATE metadata SET value = '999' WHERE key = 'height'", ()),
            ("UPDATE metadata SET value = ? WHERE key = 'tip_hash'", ("ab" * 32,)),
        ],
    }.items():
        variant = Path(tmp) / f"209-{label}.sqlite"
        shutil.copyfile(base, variant)
        for stmt, params in stmts:
            _second_connection_exec(variant, stmt, params)
        f209[label] = _verify_probe(variant)
    result["finding_209"] = f209

    # ── 210: verified startup, second-connection tamper, next append ─
    f210: dict[str, object] = {}
    db_path = Path(tmp) / "210.sqlite"
    shutil.copyfile(base, db_path)

    db = ChainDB(db_path)
    db.open()
    report = verify_chain_db(db_path)
    f210["startup_verify_outcome"] = report.outcome.value

    # Bind the append cursor exactly as the verified startup path does
    # (API exists only in the corrected tree).
    if hasattr(db, "bind_verified_cursor"):
        db.bind_verified_cursor(report.height, report.tip_hash or "")
        f210["append_cursor_bound"] = True
    else:
        f210["append_cursor_bound"] = False

    # Tamper a HISTORICAL transaction row through a SECOND connection.
    _second_connection_exec(
        db_path,
        "UPDATE transactions SET payload = ? WHERE block_index = 1 AND tx_index = 0",
        (json.dumps({"n": 999999, "note": "TAMPERED"}),),
    )
    f210["history_tampered_via_second_connection"] = True

    # Append the NEXT valid block on the (untampered) tip.
    tip = db.get_block(db.height - 1)
    next_block = _block_on(tip, [_tx(900)])
    try:
        db.append_block(next_block)
        f210["append_after_history_tamper"] = "ACCEPTED"
        f210["height_after_append"] = db.height
    except Exception as exc:  # noqa: BLE001
        f210["append_after_history_tamper"] = "REFUSED"
        f210["append_refusal"] = f"{type(exc).__name__}: {exc}"
        f210["append_refusal_typed"] = type(exc).__name__ not in (
            "Exception", "RuntimeError", "ValueError",
        )
    db.close()

    subsequent = verify_chain_db(db_path)
    f210["subsequent_full_verify"] = subsequent.outcome.value
    result["finding_210"] = f210

    # ── 211: fleet shared-population config without chain identity ───
    f211: dict[str, object] = {}
    from doin_node.cli import load_config  # noqa: E402

    fleet_example = (
        Path(NODE_SRC).parent
        / "examples" / "predictor_omega_node_cnn_direction_neat.json"
    )
    f211["fleet_example"] = fleet_example.name
    try:
        config = load_config(str(fleet_example), {})
        f211["materialized_without_identity"] = "ACCEPTED"
        f211["config_chain_id"] = config.chain_id
        f211["config_genesis_hash"] = config.genesis_hash
    except Exception as exc:  # noqa: BLE001
        f211["materialized_without_identity"] = "REFUSED"
        f211["refusal"] = f"{type(exc).__name__}: {exc}"

    template = (
        Path(NODE_SRC).parent
        / "examples" / "fleet_shared_population_identity_template.json"
    )
    if template.exists():
        try:
            config = load_config(str(template), {})
            f211["identity_template"] = {
                "materialized": "ACCEPTED",
                "chain_id": config.chain_id,
                "genesis_hash": config.genesis_hash,
            }
        except Exception as exc:  # noqa: BLE001
            f211["identity_template"] = {
                "materialized": "REFUSED",
                "refusal": f"{type(exc).__name__}: {exc}",
            }
    else:
        f211["identity_template"] = "not present in this tree"
    result["finding_211"] = f211

print(json.dumps(result, indent=2))
