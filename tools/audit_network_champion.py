"""Read-only audit of chain equality and accepted champion progression.

Usage: python tools/audit_network_champion.py --node node-a=http://host:port
       --node node-b=http://host:port --domain your-domain --output report.json

No training, node changes, secrets or private-key access. The report omits
endpoint addresses. Exit 0 means the checked properties passed; 2 means a
violation; 3 means the sample changed or could not be obtained. This does not
certify model scores: independent replay of the configured evaluation is a
separate requirement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from doin_core.crypto.hashing import compute_merkle_root
from doin_core.models import Block


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def fetch_node(named_endpoint):
    label, url = named_endpoint.split("=", 1)

    def get(path):
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=20) as response:
            return json.load(response)

    try:
        before = get("/chain/status")
        blocks = []
        for start in range(0, before["chain_height"], 50):
            end = min(start + 49, before["chain_height"] - 1)
            blocks.extend(get(f"/chain/blocks?from={start}&to={end}")["blocks"])
        after = get("/chain/status")
        status = get("/status")
        return label, dict(stable=before == after, chain=after, blocks=blocks,
                           domains=status.get("domains", {}))
    except Exception as exc:
        return label, {"error_type": type(exc).__name__}


def audit_blocks(raw_blocks, domain, higher_is_better=False):
    errors, rows = [], []
    seen, best = {}, None
    new_bests = 0
    previous_hash = "0" * 64
    for index, raw in enumerate(raw_blocks):
        block = Block.model_validate(raw)
        if (block.header.index != index or block.header.previous_hash != previous_hash
                or block.hash != block.header.compute_hash()):
            errors.append({"block": index, "kind": "BLOCK_LINK_OR_HASH"})
        if index and block.header.merkle_root != compute_merkle_root(
                [tx.id for tx in block.transactions]):
            errors.append({"block": index, "kind": "MERKLE_MISMATCH"})
        previous_hash = block.hash
        accepted = {}
        for tx in block.transactions:
            if (tx.domain_id != domain or tx.tx_type.value != "optimae_accepted"
                    or not tx.payload.get("optimae_id")):
                continue
            if tx.id != tx.compute_id():
                errors.append({"block": index, "kind": "ACCEPTANCE_BODY_HASH_MISMATCH"})
            key = tx.payload["optimae_id"]
            item = accepted.setdefault(key, {"optimae_id": key, "peer_id": tx.peer_id})
            for metric in ("verified_performance", "performance"):
                if tx.payload.get(metric) is not None:
                    value = tx.payload[metric]
                    if "performance" in item and item["performance"] != value:
                        errors.append({"block": index, "kind": "ACCEPTANCE_SCORE_CONFLICT"})
                    item["performance"] = value
            ref = tx.payload.get("candidate_archive_reference")
            if ref:
                item["body_digest"] = ref.get("body_digest")
        for key, item in accepted.items():
            if key in seen:
                errors.append({"block": index, "kind": "REPEATED_OPTIMUM",
                               "first_block": seen[key]})
            seen.setdefault(key, index)
            if item["peer_id"] != block.header.generator_id:
                errors.append({"block": index, "kind": "GENERATOR_NOT_DISCOVERER"})
            value = item.get("performance")
            if value is None:
                errors.append({"block": index, "kind": "UNRESOLVED_ACCEPTANCE_SCORE"})
                continue
            improves = best is None or (value > best if higher_is_better else value < best)
            if improves:
                best, new_bests = value, new_bests + 1
            else:
                errors.append({"block": index, "kind": "NO_NEW_NETWORK_BEST",
                               "performance": value, "best_before": best})
            rows.append(dict(block=index, **item, improves=improves, best_after=best))
    return {"errors": errors, "acceptances": rows, "unique_optimae": len(seen),
            "new_best_observations_including_first": new_bests, "best": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({node.split("=", 1)[0] for node in args.node}) != len(args.node):
        parser.error("node labels must be unique")
    with ThreadPoolExecutor(max_workers=min(4, len(args.node))) as pool:
        fetched = dict(pool.map(fetch_node, args.node))
    report = {"schema": "doin.network_champion_audit.v1",
              "observed_at": datetime.now(timezone.utc).isoformat(),
              "domain": args.domain, "nodes": {}}
    complete = True
    for label, data in fetched.items():
        if "error_type" in data:
            report["nodes"][label] = data
            complete = False
            continue
        complete &= data["stable"] and len(data["blocks"]) == data["chain"]["chain_height"]
        report["nodes"][label] = {
            "stable": data["stable"], "height": data["chain"]["chain_height"],
            "tip_hash": data["chain"]["tip_hash"],
            "network_id": data["chain"].get("network_id"),
            "versions": data["chain"].get("component_versions"),
            "all_blocks_sha256": digest(data["blocks"]),
            "runtime_champion": data["domains"].get(args.domain, {}).get("best_performance"),
        }
    if complete:
        values = list(fetched.values())
        report["all_blocks_equal"] = all(x["blocks"] == values[0]["blocks"] for x in values)
        report["analysis"] = audit_blocks(values[0]["blocks"], args.domain, args.higher_is_better)
        report["verdict"] = "PASS" if report["all_blocks_equal"] and not report["analysis"]["errors"] else "FAIL"
    else:
        report["verdict"] = "INCONCLUSIVE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "analysis"}, indent=2))
    if "analysis" in report:
        print(json.dumps({k:v for k,v in report["analysis"].items() if k != "acceptances"}, indent=2))
    return {"PASS": 0, "FAIL": 2, "INCONCLUSIVE": 3}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
