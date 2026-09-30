"""Private file-based worker; executed by the pinned predictor interpreter."""

import json
import sys
from pathlib import Path


def main():
    checkout, request_path, response_path = sys.argv[1:]
    sys.path.insert(0, checkout)
    from tools.modular_candidate_evaluator import evaluate_candidate

    request = json.loads(Path(request_path).read_text())
    result = evaluate_candidate(
        request["config"], request["train_path"],
        request["validation_path"], request["output_dir"],
    )
    Path(response_path).write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
