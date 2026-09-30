# Modular DOIN handoff (offline contract lane)

The versioned `modular.experiment.v1` payload lives in `doin-core` and is
carried under `domains[].optimization_config.modular_experiment`. `doin-node`
validates it while loading JSON, copies it to the inference plugin config, and
sets `optimization_metric` in both subtrees. Conflicting direction or metric
settings fail before plugin loading. Existing domains without this key retain
their prior config behavior.

The architecture requires at least two named branches with temporal encoders,
sequence fusion, and a temporal core. `R0` forbids a detector checkpoint;
`R1` and `R2` require a reference, with frozen versus fine-tunable behavior
left to the actual training plugin. The objective is a named validation metric,
direction and unit. Data identity includes a chronological train/validation
boundary. The payload is a candidate specification, never a fitness claim.

Run without a node, plugin, GPU, or chain:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:../doin-core-modular-handoff-20260930/src \
  python -m doin_node.modular_dry_run examples/modular_handoff_dry_run.json
```

The example uses placeholder data identity and `optimize: false` /
`evaluate: false`. It proves serialization, validation and config handoff only.
Real training still requires a predictor-side modular trainer and evaluator
that consume this schema, enforce R1 freezing/R2 fine-tuning and data lineage,
produce measured validation metrics, and support independent DOIN verification.

## Requirement trace

| ID | Requirement | Structural evidence | Behavioral evidence |
|---|---|---|---|
| M1 | Typed serializable temporal/branched candidate | `doin_core.models.modular_experiment` | `test_json_round_trip_preserves_branch_and_objective`, invalid-shape cases |
| M2 | Explicit R0/R1/R2 checkpoint semantics | `ModularExperiment.checkpoint_matches_regime` | regime tests |
| M3 | One objective to optimizer and evaluator | `load_config` materialization | `test_modular_config_handoff_to_both_plugin_subtrees`, conflict tests |
| M4 | No false optimization evidence or live runtime | `modular_dry_run` does not construct a node | `test_offline_dry_run_reports_no_training`, local CLI dry-run |

Method state: discovery, requirements, acceptance/system/component/integration/
unit test design, implementation, and offline system verification completed in
this lane. Alpha scientific acceptance and release are deferred until the real
training/evaluation dependency above is implemented and measured. Next action:
predictor owner implements the schema consumer and trains on an identified
dataset with independent validation; DOIN then verifies reported metrics.
