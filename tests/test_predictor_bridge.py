"""Protocol doubles exercise IPC only; these are NOT training evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import EntryPoint

import pytest

from doin_core.plugins.base import InferencePlugin, OptimizationPlugin
from doin_node.predictor_bridge import (
    CandidateEvaluationError, DuplicateCandidateError, NoImprovementError,
    PredictorCandidateBridge,
)


DOUBLE = '''
import hashlib, json, os, time
from pathlib import Path
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate_candidate(config, train_path, validation_path, output_dir):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['OMP_NUM_THREADS'] == '1'
    mode = config.get('test_mode')
    if mode == 'error':
        raise RuntimeError('deliberate protocol double failure')
    if mode == 'timeout':
        time.sleep(10)
    out = Path(output_dir)
    out.mkdir()
    model = out / 'best.keras'
    model.write_bytes(b'PROTOCOL DOUBLE, NOT A TRAINED MODEL')
    value = float('nan') if mode == 'nan' else 0.25
    result = dict(
        schema_version='modular.candidate.evaluation.v1', status='completed',
        objective=dict(config['objective'], value=value), metrics={'MAE': value},
        training={'observed_updates': 2, 'selected_epoch': 1},
        data={'test_used': False}, reload_parity={'passed': True},
        artifacts={'best_model': str(model)},
        digests=dict(config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            train_sha256=digest(train_path), validation_sha256=digest(validation_path),
            model_sha256=digest(model)))
    if mode == 'dry_run': result['status'] = 'config_validated_no_training'
    if mode == 'unmeasured': result['training']['observed_updates'] = 0
    if mode == 'metric_mismatch': result['metrics']['MAE'] = 0.5
    if mode == 'bad_digest': result['digests']['config_sha256'] = 'wrong'
    if mode == 'no_artifact': model.unlink()
    if mode == 'missing': return {'objective': 0.25}
    return result
'''


@pytest.fixture
def config(tmp_path):
    checkout = tmp_path / 'predictor checkout'
    (checkout / 'tools').mkdir(parents=True)
    (checkout / 'tools' / '__init__.py').write_text('')
    (checkout / 'tools' / 'modular_candidate_evaluator.py').write_text(DOUBLE)
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                  "commit", "-qm", "protocol double"]):
        subprocess.run(['git', '-C', str(checkout), *args], check=True)
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    train, val = tmp_path / 'train.npz', tmp_path / 'validation.npz'
    train.write_bytes(b'protocol train input')
    val.write_bytes(b'protocol validation input')
    return dict(predictor_checkout=str(checkout), predictor_python=sys.executable,
                predictor_revision=revision, train_path=str(train), validation_path=str(val),
                output_dir=str(tmp_path / 'runs'), timeout_seconds=5,
                candidate_config=dict(window=16, evaluator={'max_epochs': 2},
                    objective=dict(metric='MAE', split='validation', higher_is_better=False, unit='z_train')))


def bridge(config):
    plugin = PredictorCandidateBridge()
    plugin.configure(config)
    return plugin


def test_configured_parameters_roundtrip(config):
    plugin = bridge(config)
    params, value = plugin.optimize({'window': 999}, None)
    assert params == config['candidate_config']
    assert value == 0.25
    assert plugin.get_domain_metadata() == dict(performance_metric='MAE', higher_is_better=False)
    request = next(Path(config['output_dir']).glob('*/request.json'))
    assert json.loads(request.read_text())['config'] == params
    assert isinstance(plugin, (OptimizationPlugin, InferencePlugin))


def test_overrides_and_independent_plugin_evaluation(config):
    plugin = bridge(config)
    result = plugin.evaluate_candidate({'window': 32, 'evaluator': {'seed': 17}})
    params = plugin.last_parameters
    assert params['window'] == 32 and params['evaluator'] == {'seed': 17}
    assert config['candidate_config']['window'] == 16
    config['output_dir'] += '-verification'
    verifier = bridge(config)
    assert verifier.evaluate(params) == result['objective']['value']
    with pytest.raises(ValueError, match='identified'):
        verifier.evaluate(params, {'unidentified': True})


@pytest.mark.parametrize('mode', ['error', 'nan', 'dry_run', 'unmeasured',
                                 'metric_mismatch', 'bad_digest', 'no_artifact', 'missing'])
def test_no_fitness_from_invalid_result(config, mode):
    plugin = bridge(config)
    with pytest.raises(CandidateEvaluationError):
        plugin.evaluate_candidate({'test_mode': mode})
    assert plugin.last_result is None
    assert not list(Path(config['output_dir']).glob('*/accepted.json'))


def test_timeout(config):
    config['timeout_seconds'] = 0.1
    with pytest.raises(CandidateEvaluationError, match='timed out'):
        bridge(config).evaluate_candidate({'test_mode': 'timeout'})


def test_duplicate_identity_persists_across_instances(config):
    first = bridge(config).evaluate_candidate()
    config['candidate_config'] = dict(reversed(list(config['candidate_config'].items())))
    with pytest.raises(DuplicateCandidateError):
        bridge(config).evaluate_candidate()
    changed = bridge(config).evaluate_candidate({'window': 17})
    assert first['bridge']['candidate_id'] != changed['bridge']['candidate_id']
    Path(config['train_path']).write_bytes(b'changed input identity')
    changed_data = bridge(config).evaluate_candidate()
    assert first['bridge']['candidate_id'] != changed_data['bridge']['candidate_id']


def test_pin_and_nonfinite_inputs_refused(config):
    config['predictor_revision'] = '0' * 40
    with pytest.raises(ValueError, match='pinned'):
        bridge(config)
    config['timeout_seconds'] = float('nan')
    with pytest.raises(ValueError):
        bridge(config)


def test_no_improvement(config):
    with pytest.raises(NoImprovementError):
        bridge(config).optimize(None, 0.1)


def test_missing_interpreter_and_dirty_checkout(config):
    config['predictor_python'] += '-missing'
    with pytest.raises(CandidateEvaluationError):
        bridge(config).evaluate_candidate()
    evaluator = Path(config['predictor_checkout']) / 'tools/modular_candidate_evaluator.py'
    evaluator.write_text(DOUBLE + '\n# dirty\n')
    with pytest.raises(ValueError, match='clean'):
        bridge(config)


def test_cli_and_entry_points(config, tmp_path):
    tomllib = pytest.importorskip('tomllib')
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / 'pyproject.toml').read_text())
    for group, base in [('doin.optimization', OptimizationPlugin), ('doin.inference', InferencePlugin)]:
        target = metadata['project']['entry-points'][group]['predictor_candidate']
        assert issubclass(EntryPoint(name='predictor_candidate', value=target, group=group).load(), base)
    path = tmp_path / 'bridge.json'
    path.write_text(json.dumps(config))
    command = [sys.executable, '-m', 'doin_node.predictor_bridge', '--config', str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, env=os.environ.copy())
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt['performance'] == 0.25
    assert receipt['parameters'] == config['candidate_config']
