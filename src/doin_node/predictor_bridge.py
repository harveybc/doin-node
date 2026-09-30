"""Local-only DOIN plugin adapter to predictor's measured candidate evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess

from doin_core.plugins.base import InferencePlugin, OptimizationPlugin


class CandidateEvaluationError(RuntimeError):
    """No usable measured result was produced."""


class DuplicateCandidateError(CandidateEvaluationError):
    """This candidate is already reserved, completed, or failed in this run."""


class NoImprovementError(CandidateEvaluationError):
    """The measured candidate did not improve the incumbent."""


class VerificationError(CandidateEvaluationError):
    """Independent checkpoint rescoring did not confirm the receipt."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


class PredictorCandidateBridge(OptimizationPlugin, InferencePlugin):
    """One caller-proposed candidate per step, not a replacement search engine.

    configure receives execution settings and candidate_config. evaluate_candidate
    accepts flat top-level overrides (nested values replace, never deep-merge).
    evaluate NEVER retrains: it locates the accepted receipt for the candidate's
    identity (under output_dir or receipt_roots) and independently rescores its
    saved checkpoint in a fresh pinned process (verify_checkpoint). Optional
    settings: cuda_visible_devices (default "" = CPU), cpu_threads (1..64),
    heartbeat_interval (<= 60 s), receipt_roots. No live dispatch is implemented.
    """

    def configure(self, config):
        self.config = json.loads(canonical(config))
        for key in ("predictor_checkout", "predictor_python", "train_path",
                    "validation_path", "output_dir"):
            value = Path(config[key]).expanduser()
            if not value.is_absolute():
                raise ValueError(f"{key} must be absolute")
            self.config[key] = str(value.resolve())
        self.base = self.config["candidate_config"]
        if not isinstance(self.base, dict):
            raise ValueError("candidate_config must be an object")
        self.timeout = config.get("timeout_seconds", 300)
        if not finite_number(self.timeout) or not 0 < self.timeout <= 86400:
            raise ValueError("timeout_seconds must be finite and in (0, 86400]")
        self.devices = self.config.get("cuda_visible_devices", "")
        if not isinstance(self.devices, str):
            raise ValueError("cuda_visible_devices must be a string")
        self.threads = self.config.get("cpu_threads", 1)
        if type(self.threads) is not int or not 1 <= self.threads <= 64:
            raise ValueError("cpu_threads must be an integer in [1, 64]")
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30.0)
        if not finite_number(self.heartbeat_interval) or not 0 < self.heartbeat_interval <= 60:
            raise ValueError("heartbeat_interval must be in (0, 60] seconds")
        roots = self.config.get("receipt_roots", [self.config["output_dir"]])
        if not isinstance(roots, list) or not all(isinstance(r, str) and Path(r).is_absolute() for r in roots):
            raise ValueError("receipt_roots must be absolute paths")
        self.receipt_roots = [str(Path(r).resolve()) for r in roots]
        self._check_pin()
        self.last_result = None
        self.last_parameters = None
        self.get_domain_metadata()

    def _check_pin(self):
        revision = self.config["predictor_revision"]
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError("predictor_revision must be a full commit SHA")
        actual = subprocess.run(
            ["git", "-C", self.config["predictor_checkout"], "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if actual != revision:
            raise ValueError("predictor checkout does not match pinned revision")
        checkout = self.config["predictor_checkout"]
        tracked = subprocess.run(
            ["git", "-C", checkout, "ls-files", "--error-unmatch",
             "tools/modular_candidate_evaluator.py"],
            capture_output=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", checkout, "diff", "--quiet", "HEAD", "--"],
            capture_output=True, timeout=10,
        )
        if tracked.returncode or dirty.returncode:
            raise ValueError("predictor evaluator must be committed and tracked checkout clean")

    def _objective(self, candidate):
        nested = candidate.get("modular_experiment", {}).get("objective")
        objective = candidate.get("objective", nested)
        if nested is not None and objective != nested:
            raise ValueError("conflicting objectives")
        if (not isinstance(objective, dict) or
                set(objective) != {"metric", "split", "higher_is_better", "unit"} or
                objective["split"] != "validation" or
                type(objective["higher_is_better"]) is not bool or
                not all(isinstance(objective[k], str) and objective[k]
                        for k in ("metric", "unit"))):
            raise ValueError("explicit validation objective required")
        return objective

    def get_domain_metadata(self):
        objective = self._objective(self.base)
        return {"performance_metric": objective["metric"],
                "higher_is_better": objective["higher_is_better"]}

    def evaluate_candidate(self, parameters=None):
        self.last_result = None
        self.last_parameters = None
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("candidate parameters must be an object")
        candidate, objective, config_hash, hashes, identity = self._identity(parameters)
        run = Path(self.config["output_dir"]) / identity
        run.parent.mkdir(parents=True, exist_ok=True)
        try:
            run.mkdir()
        except FileExistsError as exc:
            raise DuplicateCandidateError(identity) from exc
        request = {"config": candidate,
                   "train_path": self.config["train_path"],
                   "validation_path": self.config["validation_path"],
                   "output_dir": str(run / "artifacts"),
                   "heartbeat_path": str(run / "heartbeat.json"),
                   "heartbeat_interval": self.heartbeat_interval}
        (run / "request.json").write_text(canonical(request) + "\n")
        response = run / "response.json"
        try:
            code = self._worker([self.config["predictor_checkout"], str(run / "request.json"),
                                 str(response)], run / "worker.log")
            if code != 0:
                raise CandidateEvaluationError(f"predictor exited {code}; see {run / 'worker.log'}")
            result = json.loads(response.read_text())
            canonical(result)  # Reject nonfinite values anywhere in the receipt.
            self._validate(result, objective, config_hash, hashes, run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CandidateEvaluationError(f"invalid predictor result in {run}: {exc}") from exc
        result["bridge"] = {"candidate_id": identity,
                            "predictor_revision": self.config["predictor_revision"],
                            "predictor_python": self.config["predictor_python"]}
        (run / "accepted.json").write_text(canonical(result) + "\n")
        self.last_result = result
        self.last_parameters = candidate
        return result

    def _identity(self, parameters):
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("candidate parameters must be an object")
        candidate = json.loads(canonical({**self.base, **(parameters or {})}))
        objective = self._objective(candidate)
        if objective != self._objective(self.base):
            raise ValueError("candidate cannot change the configured objective")
        self._check_pin()
        hashes = {key: digest_file(self.config[key + "_path"])
                  for key in ("train", "validation")}
        config_hash = hashlib.sha256(canonical(candidate).encode()).hexdigest()
        identity = hashlib.sha256(canonical({
            "config": config_hash, "data": hashes,
            "revision": self.config["predictor_revision"],
            "interpreter": self.config["predictor_python"],
        }).encode()).hexdigest()
        return candidate, objective, config_hash, hashes, identity

    def _worker(self, args, log_path):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        threads = str(self.threads)
        env.update(CUDA_VISIBLE_DEVICES=self.devices, OMP_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   TF_NUM_INTRAOP_THREADS=threads, TF_NUM_INTEROP_THREADS="1",
                   PYTHONUNBUFFERED="1")
        command = [self.config["predictor_python"], "-I", "-u",
                   str(Path(__file__).with_name("predictor_worker.py")), *args]
        with Path(log_path).open("w") as log:
            process = subprocess.Popen(
                command, cwd=self.config["predictor_checkout"], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                return process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise CandidateEvaluationError("predictor timed out") from exc

    def verify_checkpoint(self, accepted_path):
        """Independent rescoring of an accepted receipt's saved checkpoint; never trains."""
        accepted_path = Path(accepted_path).resolve()
        accepted = json.loads(accepted_path.read_text())
        canonical(accepted)
        self._check_pin()
        if accepted.get("bridge", {}).get("predictor_revision") != self.config["predictor_revision"]:
            raise VerificationError("receipt was produced by a different pinned predictor revision")
        if accepted["digests"]["validation_sha256"] != digest_file(self.config["validation_path"]):
            raise VerificationError("configured validation bytes differ from the receipt")
        receipt_sha = digest_file(accepted_path)
        out = Path(self.config["output_dir"]) / ("verify-" + receipt_sha[:32])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.mkdir()
        except FileExistsError as exc:
            raise DuplicateCandidateError(f"verification already attempted: {out}") from exc
        target = out / "verification.json"
        code = self._worker(["verify", self.config["predictor_checkout"], str(accepted_path),
                             self.config["validation_path"], str(target)], out / "worker.log")
        if code != 0 or not target.is_file():
            raise VerificationError(f"scorer exited {code}; see {out / 'worker.log'}")
        result = json.loads(target.read_text())
        canonical(result)
        if result.get("receipt_sha256") != receipt_sha:
            raise VerificationError("verification bound to different receipt bytes")
        result["bridge"] = {"verification_id": out.name, "accepted_receipt": str(accepted_path),
                            "predictor_revision": self.config["predictor_revision"]}
        target.write_text(canonical(result) + "\n")
        return result

    def _validate(self, result, objective, config_hash, hashes, run):
        if (result["schema_version"] != "modular.candidate.evaluation.v1" or
                result["status"] != "completed" or result.get("dry_run", False)):
            raise ValueError("measured completed receipt required; dry-run rejected")
        training = result["training"]
        for key in ("observed_updates", "selected_epoch"):
            if type(training[key]) is not int or training[key] <= 0:
                raise ValueError("measured training updates and selected epoch required")
        if result["data"]["test_used"] is not False or result["reload_parity"]["passed"] is not True:
            raise ValueError("validation-only result and model reload parity required")
        returned = dict(result["objective"])
        value = returned.pop("value")
        if (returned != objective or not finite_number(value) or
                not finite_number(result["metrics"][objective["metric"]]) or
                value != result["metrics"][objective["metric"]]):
            raise ValueError("finite measured objective must match configured validation metric")
        digests = result["digests"]
        if digests["config_sha256"] != config_hash or any(
            digests[key + "_sha256"] != value for key, value in hashes.items()
        ):
            raise ValueError("candidate/data receipt digest mismatch")
        artifact = Path(result["artifacts"]["best_model"]).resolve()
        if not artifact.is_relative_to(run / "artifacts") or not artifact.is_file():
            raise ValueError("model artifact must exist within candidate output")
        if digest_file(artifact) != digests["model_sha256"]:
            raise ValueError("model artifact digest mismatch")

    def optimize(self, current_best_params, current_best_performance):
        # Incumbents are comparison context, never silently substituted for proposals.
        if current_best_performance is not None and not finite_number(current_best_performance):
            raise ValueError("incumbent performance must be finite")
        result = self.evaluate_candidate()
        value = result["objective"]["value"]
        if current_best_performance is not None:
            higher = self.get_domain_metadata()["higher_is_better"]
            improved = value > current_best_performance if higher else value < current_best_performance
            if not improved:
                raise NoImprovementError("candidate did not improve incumbent")
        return json.loads(canonical(self.last_parameters)), value

    def evaluate(self, parameters, data=None):
        """Independent verification by checkpoint rescoring. Never retrains."""
        if data is not None:
            raise ValueError("only configured identified train/validation paths are supported")
        *_, identity = self._identity(parameters)
        found = [Path(root) / identity / "accepted.json" for root in self.receipt_roots]
        found = [p for p in found if p.is_file()]
        if not found:
            raise CandidateEvaluationError(
                "no accepted receipt/checkpoint for this candidate identity; evaluate does not retrain")
        result = self.verify_checkpoint(found[0])
        if result["verdict"] != "VERIFIED":
            raise VerificationError("checkpoint rescoring refuted the receipt: " + "; ".join(result["problems"]))
        return result["objective"]["rescored_value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Bridge execution JSON")
    parser.add_argument("--candidate", help="Optional flat candidate overrides JSON")
    parser.add_argument("--verify", help="Accepted receipt to rescore from its checkpoint (no training)")
    args = parser.parse_args()
    bridge = PredictorCandidateBridge()
    bridge.configure(json.loads(Path(args.config).read_text()))
    if args.verify:
        result = bridge.verify_checkpoint(args.verify)
        print(canonical({"verdict": result["verdict"], "objective": result["objective"],
                         "problems": result["problems"]}))
        raise SystemExit(0 if result["verdict"] == "VERIFIED" else 3)
    overrides = json.loads(Path(args.candidate).read_text()) if args.candidate else None
    result = bridge.evaluate_candidate(overrides)
    print(canonical({"parameters": bridge.last_parameters,
                     "performance": result["objective"]["value"], "result": result}))


if __name__ == "__main__":
    main()
