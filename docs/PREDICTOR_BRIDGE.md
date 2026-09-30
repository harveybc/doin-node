# Offline predictor candidate bridge

Scope: implement the existing DOIN optimization/inference ABCs as a local
adapter to `tools.modular_candidate_evaluator.evaluate_candidate`. No node,
network, chain, search algorithm, or synthetic training substitute is involved.
The caller proposes candidates; predictor owns actual training and data gates.

## Test design and traceability

| ID | Requirement | Planned evidence |
| --- | --- | --- |
| B1 | Candidate config survives JSON and plugin roundtrip | subprocess receipt and returned parameters |
| B2 | Only measured receipts become fitness | status, updates, objective/metric, artifact/digest checks; dry-run refusal |
| B3 | Isolated pinned interpreter, CPU and bounded failures | subprocess errors, timeout, nonfinite results, revision mismatch |
| B4 | Duplicate identity cannot overwrite a result | canonical config/data/revision hash and atomic directory reservation |
| B5 | Offline optimizer-facing command | CLI subprocess test; no UnifiedNode import |

Discovery through unit-test design completed before implementation. Tests use
explicit protocol doubles only, never scientific evidence. Alpha acceptance
requires the parent to run the real engine on identified temporal splits.

## Local command

Create a bridge execution JSON with these keys (all paths absolute):

```json
{
  "predictor_checkout": "/path/to/pinned/predictor-worktree",
  "predictor_python": "/path/to/predictor-env/bin/python",
  "predictor_revision": "<full 40-character committed predictor HEAD>",
  "train_path": "/path/to/identified/train.npz",
  "validation_path": "/path/to/identified/validation.npz",
  "output_dir": "/path/to/new/local-optimizer-run",
  "timeout_seconds": 300,
  "candidate_config": {
    "objective": {"metric": "MAE", "split": "validation", "higher_is_better": false, "unit": "z_train"}
  }
}
```

`candidate_config` must contain the **complete real engine configuration**, not
just the objective shown above: branch/temporal settings plus `window`,
`sample_hours`, `feature_names`, `horizons`, `target_feature_indices`, and
bounded `evaluator` settings. Use the engine owner's config. Inputs must satisfy
the predictor evaluator's identified temporal NPZ contract, not CSV fixtures.
The evaluator file must be committed and tracked changes must be clean.

From this DOIN worktree, with a DOIN-capable interpreter and the modular core:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src:../doin-core-modular-handoff-20260930/src \
  /path/to/doin-env/bin/python -m doin_node.predictor_bridge \
  --config /path/to/bridge.json --candidate /path/to/candidate-overrides.json
```

`--candidate` is optional. Overrides replace whole top-level values; objective
changes are forbidden. Stdout is JSON containing effective `parameters`,
measured `performance`, and the evaluator `result`. Training logs go to
`output_dir/<candidate_id>/worker.log`; request, raw response and accepted
receipt are separate JSON files. Nonzero exit means no usable fitness.

Install in an isolated DOIN environment to discover `predictor_candidate` in
both `doin.optimization` and `doin.inference`, and the
`doin-predictor-candidate` console command. No installation into a live
environment was performed. This explicit adapter is the limited exception to
the repository's usual external-plugin layout requested for this handoff.

## Plugin API and limits

- `configure(execution_config)` pins checkout/interpreter and base candidate.
- `evaluate_candidate(overrides=None)` returns the validated receipt.
- `optimize(current_best_params, current_best_performance)` measures the configured
  proposal and returns `(effective_parameters, objective_value)` only on improvement.
  Incumbent parameters are not proposals. No search loop is invented here.
- `evaluate(parameters, data=None)` retrains via the same evaluator; it is **not**
  independent checkpoint inference, a synthetic-data adapter, or consensus proof.
  Use a separate output root to repeat a candidate for verification.
- Duplicate canonical config/data/revision/interpreter identities are refused
  across instances, including failed reservations. Retry in a new output root.
- The worker uses `-I`, explicit checkout import, CPU visibility, one-thread
  numeric settings and a wall timeout that kills its own process group only.
  This is interpreter isolation, not an untrusted-code security sandbox.

## Handoff evidence

Bounded verification: 74 passed in 2.49 seconds, zero failures, using
`python -m pytest tests/test_predictor_bridge.py tests/test_modular_handoff.py
tests/test_cli_config.py -q` with CPU visibility and one-thread limits.
`git diff --check` passed. The full repository suite was not run.

`tests/test_predictor_bridge.py` maps B1-B5 to subprocess protocol tests;
`tests/test_modular_handoff.py` and `tests/test_cli_config.py` protect existing
config handoff. These establish bridge behavior only. No real engine, dataset,
checkpoint, live service or chain was run or changed. Parent must supply and
commit the engine checkout, execute the command on real identified inputs, and
retain its accepted receipt before claiming alpha training acceptance.
