"""Small synthetic mechanism/provenance checks, never performance evidence."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from ptin_sim.jev_hmarl_extension import JevHMARL, JevHMARLPolicy, HMARLConfig
from ptin_sim.closed_loop.communication_study.physical_research_policies import CandidateHMARL
from ptin_sim.closed_loop.communication_study.jev_iql import OutcomeCalibration


@pytest.fixture
def example():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    config = HMARLConfig(3, 33, units=2, per_unit_dim=15, identity_dim=1, hidden_dim=16)
    calibration = OutcomeCalibration('jev-fixture', 1., (0., 0., 0., 0.), 'a' * 64)
    batch = dict(states=torch.randn(2, 3), candidates=torch.randn(2, 3, 33),
                 valid=torch.tensor([[True, True, True], [True, True, False]]),
                 chosen=torch.tensor([0, 1]))
    annotation = dict(model='jev-fixture', split='training', candidate_indices=[0, 1],
                      probabilities=[.05, .95], abstain_probability=0., abstained=False)
    batch['teacher_annotations'] = [annotation, None]
    return config, calibration, batch, torch.tensor([-1., -2.])


def test_zero_coefficient_reproduces_frozen_baseline_update(example):
    cfg, cal, batch, returns = example
    base = CandidateHMARL(cfg)
    model = JevHMARL(cfg, calibration=cal, teacher_weight=0.)
    model.load_state_dict(base.state_dict())
    base.update(batch, returns, available_only=True)
    result = model.update(batch, returns, available_only=True)
    assert result['teacher_loss'] == 0 and result['teacher_admitted'] == 0
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key


def test_teacher_changes_actor_and_mixer_gradients_not_value(example):
    cfg, cal, batch, _ = example
    model = JevHMARL(cfg, calibration=cal)
    scores = model.scores(batch['states'], batch['candidates'])
    loss, count = model.preference_objective(scores, batch['valid'], batch['teacher_annotations'])
    loss.backward()
    assert count == 1
    assert sum(p.grad.abs().sum() for p in model.local.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.mix.parameters()) > 0
    assert all(p.grad is None for p in model.value.parameters())
    # Exact KL direction: the teacher-preferred candidate is pushed upward.
    logits = torch.zeros(2, 3, requires_grad=True)
    loss, _ = model.preference_objective(logits, batch['valid'], batch['teacher_annotations'])
    loss.backward()
    assert logits.grad[0, 1] < 0 < logits.grad[0, 0]
    assert logits.grad[0, 2] == 0


def test_packed_and_padded_updates_agree(example):
    cfg, cal, batch, returns = example
    a = JevHMARL(cfg, calibration=cal)
    b = JevHMARL(cfg, calibration=cal)
    b.load_state_dict(a.state_dict())
    packed = dict(batch, candidates=batch['candidates'][batch['valid']])
    x = a.update(batch, returns, available_only=True)
    y = b.update(packed, returns, available_only=True)
    assert x == pytest.approx(y, abs=1e-6)
    for key, value in a.state_dict().items():
        assert torch.allclose(value, b.state_dict()[key], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('field,value', [
    ('split', 'validation'), ('model', 'wrong'), ('candidate_indices', [0, 0]),
    ('candidate_indices', [0, 3]), ('candidate_indices', [0, 1.5]),
    ('probabilities', [.5, float('nan')]), ('abstain_probability', 2.)])
def test_invalid_teacher_data_rejected(example, field, value):
    cfg, cal, batch, _ = example
    batch = deepcopy(batch)
    batch['teacher_annotations'][0][field] = value
    model = JevHMARL(cfg, calibration=cal)
    with pytest.raises(ValueError):
        model.preference_objective(model.scores(batch['states'], batch['candidates']),
                                   batch['valid'], batch['teacher_annotations'])


@pytest.mark.parametrize('why', ['unsupported', 'abstained', 'all_abstain_mass', 'missing'])
def test_unsupported_teacher_has_zero_effect(example, why):
    cfg, cal, batch, _ = example
    if why == 'unsupported':
        cal = OutcomeCalibration(cal.model, 1., (None,) * 4, 'a' * 64)
    if why == 'abstained':
        batch['teacher_annotations'][0]['abstained'] = True
    if why == 'all_abstain_mass':
        batch['teacher_annotations'][0]['abstain_probability'] = 1.
    if why == 'missing':
        batch['teacher_annotations'] = [None, None]
    model = JevHMARL(cfg, calibration=cal)
    loss, count = model.preference_objective(model.scores(batch['states'], batch['candidates']),
                                             batch['valid'], batch['teacher_annotations'])
    assert float(loss) == 0 and count == 0


def test_uncalibrated_control_uses_same_labels_when_risk_bins_unsupported(example):
    cfg, cal, batch, _ = example
    cal = OutcomeCalibration(cal.model, 1., (None,) * 4, 'a' * 64)
    model = JevHMARL(cfg, calibration=cal, use_outcome_calibration=False)
    loss, count = model.preference_objective(model.scores(batch['states'], batch['candidates']),
                                             batch['valid'], batch['teacher_annotations'])
    assert count == 1 and loss > 0


def test_teacher_option_permutation_keeps_the_same_gradient(example):
    cfg, cal, batch, _ = example
    model = JevHMARL(cfg, calibration=cal)
    a = torch.zeros(2, 3, requires_grad=True)
    loss, _ = model.preference_objective(a, batch['valid'], batch['teacher_annotations'])
    loss.backward()
    batch['teacher_annotations'][0].update(candidate_indices=[1, 0], probabilities=[.95, .05])
    b = torch.zeros(2, 3, requires_grad=True)
    loss, _ = model.preference_objective(b, batch['valid'], batch['teacher_annotations'])
    loss.backward()
    assert torch.equal(a.grad, b.grad)


def test_checkpoint_reload_and_local_selector(example, tmp_path):
    cfg, cal, batch, returns = example
    model = JevHMARL(cfg, calibration=cal)
    model.update(batch, returns)
    evidence = dict(dataset_sha256='b' * 64, encoder_sha256='c' * 64,
                    annotations_sha256='d' * 64, requests_sha256='e' * 64,
                    calibration_sha256='a' * 64, reward_metric='critical_ens_kwh',
                    contains_test_or_validation_data=False)
    path = tmp_path / 'synthetic.pt'
    model.save(path, evidence=evidence)
    restored, _ = JevHMARL.load(path)
    assert torch.equal(model.scores(batch['states'], batch['candidates']),
                       restored.scores(batch['states'], batch['candidates']))
    # Optimizer state is retained too; the next update stays identical.
    model.update(batch, returns)
    restored.update(batch, returns)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    encoder = SimpleNamespace(sha256='c' * 64,
                              encode=lambda obs, joints: (batch['states'][0], batch['candidates'][0]))
    policy = JevHMARLPolicy(path, encoder, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           dataset_sha256='b' * 64)
    choices = [[{'kind': 'wait'}], [{'kind': 'supply'}], [{'kind': 'repair'}]]
    chosen = policy.select({}, choices)
    assert chosen in choices and not policy.last_decision_trace['execution_uses_cloud_api']
    chosen[0]['kind'] = 'changed'
    assert all(j[0]['kind'] != 'changed' for j in choices)
    with pytest.raises(ValueError, match='identity'):
        CandidateHMARL.load(path)
