"""Private file-based worker; executed by the pinned predictor interpreter.

Modes:
  train:  <checkout> <request.json> <response.json>
          Real predictor evaluator. When the pinned checkout provides
          tools/modular_heartbeat.py and the request names heartbeat_path, the
          evaluation runs under its unbuffered heartbeat (<= 60 s cadence).
  verify <checkout> <accepted.json> <validation.npz> <verification.json>
          Independent checkpoint rescoring (no fit) by tools/modular_checkpoint_scorer.
"""

import json
import sys
from pathlib import Path


def train(checkout, request_path, response_path):
    request = json.loads(Path(request_path).read_text())
    heartbeat = request.get("heartbeat_path")
    if heartbeat and (Path(checkout) / "tools" / "modular_heartbeat.py").is_file():
        from tools.modular_heartbeat import run_request

        run_request(request_path, response_path, heartbeat,
                    interval=float(request.get("heartbeat_interval", 30.0)))
        return
    from tools.modular_candidate_evaluator import evaluate_candidate

    result = evaluate_candidate(
        request["config"], request["train_path"],
        request["validation_path"], request["output_dir"],
    )
    Path(response_path).write_text(json.dumps(result, allow_nan=False) + "\n")


def verify(checkout, accepted_path, validation_path, output_path):
    from tools.modular_checkpoint_scorer import verify as score

    score(accepted_path, validation_path, output_path)


def main():
    args = sys.argv[1:]
    if args and args[0] == "verify":
        sys.path.insert(0, args[1])
        verify(*args[1:])
        return
    sys.path.insert(0, args[0])
    train(*args)


if __name__ == "__main__":
    main()
