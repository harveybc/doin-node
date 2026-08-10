"""Finding 201 reproduction: forged transaction ID + post-append tamper.

Runs the auditor's counterexample against a given doin-core/doin-node source
tree (env vars DOIN_CORE_SRC / DOIN_NODE_SRC) using a TEMPORARY SQLite DB.
Never touches any real chain database.

Steps:
  1. genesis
  2. construct a transaction whose ID is an arbitrary 64-hex string unrelated
     to its body; build block 1 with a Merkle root over that forged ID; append
  3. tamper the persisted JSON payload directly in SQLite
  4. reload block 1 (tamper detected?)
  5. append block 2 on top (accepted?)

Output: one JSON object on stdout.
"""
from __future__ import annotations

import json
import os
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
from doin_node.storage.chaindb import ChainDB  # noqa: E402

result: dict[str, object] = {
    "run": os.environ.get("RUN_LABEL", "unlabeled"),
    "doin_core_src": CORE_SRC,
    "doin_node_src": NODE_SRC,
    "utc": datetime.now(timezone.utc).isoformat(),
}

FORGED_ID = "f" * 64  # arbitrary, unrelated to any body

with tempfile.TemporaryDirectory(prefix="finding201-") as tmp:
    db = ChainDB(Path(tmp) / "chain.sqlite")
    db.open()
    genesis = db.initialize("repro-genesis")

    # -- Step 2: forged-ID transaction ------------------------------------
    tx = None
    try:
        tx = Transaction(
            id=FORGED_ID,
            tx_type=TransactionType.OPTIMAE_ANNOUNCED,
            domain_id="repro-domain",
            peer_id="honest-peer",
            payload={"fitness": 0.99, "note": "original body"},
            timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )
        result["forged_id_constructed"] = True
        result["forged_id_differs_from_content_hash"] = tx.id != tx.compute_id()
    except Exception as exc:  # noqa: BLE001
        result["forged_id_constructed"] = False
        result["forged_id_construction_refusal"] = type(exc).__name__

    if tx is None:
        # Adversary bypasses construction: model_construct also refuses
        # (it runs model_post_init), so the remaining bypass is direct
        # post-construction attribute mutation.
        try:
            Transaction.model_construct(
                id=FORGED_ID,
                tx_type=TransactionType.OPTIMAE_ANNOUNCED,
                domain_id="repro-domain",
                peer_id="honest-peer",
                payload={"fitness": 0.99, "note": "original body"},
                timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc),
            )
            result["model_construct_bypass_refused"] = False
        except Exception as exc:  # noqa: BLE001
            result["model_construct_bypass_refused"] = True
            result["model_construct_refusal"] = type(exc).__name__
        tx = Transaction(
            tx_type=TransactionType.OPTIMAE_ANNOUNCED,
            domain_id="repro-domain",
            peer_id="honest-peer",
            payload={"fitness": 0.99, "note": "original body"},
            timestamp=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )
        tx.id = FORGED_ID  # attribute-mutation forgery
        result["forged_tx_built_via_attribute_mutation"] = True

    appended_forged = False
    if tx is not None:
        header1 = BlockHeader(
            index=1,
            previous_hash=genesis.hash,
            timestamp=datetime(2026, 8, 10, 0, 1, tzinfo=timezone.utc),
            merkle_root=compute_merkle_root([tx.id]),
            generator_id="repro-generator",
            weighted_performance_sum=1.0,
            threshold=0.5,
        )
        block1 = Block(header=header1, transactions=[tx])
        try:
            db.append_block(block1)
            appended_forged = True
        except Exception as exc:  # noqa: BLE001
            result["forged_append_error"] = f"{type(exc).__name__}: {exc}"
    result["forged_id_appended"] = appended_forged

    # -- Step 3: tamper persisted payload ---------------------------------
    tampered = False
    if appended_forged:
        db._conn.execute(  # direct SQLite tamper, as the auditor did
            "UPDATE transactions SET payload = ? WHERE tx_id = ?",
            (json.dumps({"fitness": 0.01, "note": "TAMPERED body"}), FORGED_ID),
        )
        tampered = True
    result["payload_tampered_in_sqlite"] = tampered

    # -- Step 4: reload block 1 -------------------------------------------
    if tampered:
        try:
            loaded = db.get_block(1)
            body = loaded.transactions[0].payload if loaded.transactions else None
            result["tamper_not_detected_on_load"] = True
            result["loaded_payload_is_tampered"] = (
                body is not None and body.get("note") == "TAMPERED body"
            )
        except Exception as exc:  # noqa: BLE001
            result["tamper_not_detected_on_load"] = False
            result["load_refusal"] = f"{type(exc).__name__}: {exc}"

    # -- Step 5: next block on top ----------------------------------------
    next_accepted = False
    next_error = ""
    if appended_forged:
        tx2 = Transaction(
            tx_type=TransactionType.DOMAIN_REGISTERED,
            domain_id="repro-domain",
            peer_id="honest-peer-2",
            payload={"ok": True},
            timestamp=datetime(2026, 8, 10, 0, 2, tzinfo=timezone.utc),
        )
        header2 = BlockHeader(
            index=2,
            previous_hash=db.tip_hash,
            timestamp=datetime(2026, 8, 10, 0, 3, tzinfo=timezone.utc),
            merkle_root=compute_merkle_root([tx2.id]),
            generator_id="repro-generator",
            weighted_performance_sum=1.0,
            threshold=0.5,
        )
        block2 = Block(header=header2, transactions=[tx2])
        try:
            db.append_block(block2)
            next_accepted = True
        except Exception as exc:  # noqa: BLE001
            next_error = f"{type(exc).__name__}: {exc}"
    result["next_block_accepted_after_tamper"] = next_accepted
    if next_error:
        result["next_block_refusal"] = next_error

    # -- Honest-chain tamper probe ----------------------------------------
    # Independent of the forged-ID path: append an HONEST block on the
    # current tip, tamper its persisted payload, and check whether any
    # load detects it. (Full-history verification at append/startup is
    # WP2 scope — finding 202; WP1's guarantee is that tampered content
    # can never be read back or re-validated without a typed refusal.)
    honest_tx = Transaction(
        tx_type=TransactionType.EVALUATION_SERVED,
        domain_id="repro-domain",
        peer_id="honest-peer-3",
        payload={"note": "honest body"},
        timestamp=datetime(2026, 8, 10, 0, 4, tzinfo=timezone.utc),
    )
    honest_index = db.height
    honest_header = BlockHeader(
        index=honest_index,
        previous_hash=db.tip_hash,
        timestamp=datetime(2026, 8, 10, 0, 5, tzinfo=timezone.utc),
        merkle_root=compute_merkle_root([honest_tx.id]),
        generator_id="repro-generator",
        weighted_performance_sum=1.0,
        threshold=0.5,
    )
    db.append_block(Block(header=honest_header, transactions=[honest_tx]))
    db._conn.execute(
        "UPDATE transactions SET payload = ? WHERE tx_id = ?",
        (json.dumps({"note": "TAMPERED honest body"}), honest_tx.id),
    )
    try:
        db.get_block(honest_index)
        result["honest_block_tamper_detected_on_load"] = False
    except Exception as exc:  # noqa: BLE001
        result["honest_block_tamper_detected_on_load"] = True
        result["honest_block_tamper_refusal"] = f"{type(exc).__name__}: {exc}"

    result["height"] = db.height
    db.close()

result["attack_succeeded"] = bool(
    result.get("forged_id_appended") and result.get("next_block_accepted_after_tamper")
)
print(json.dumps(result, indent=2))
