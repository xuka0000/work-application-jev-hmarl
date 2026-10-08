"""Frozen decision-teacher preferences inside the factorized MARL update.

This isolated development module does not change the frozen baseline tree.
It optimizes device-factor scores and their joint mixer, not Jev's weights.
Execution uses the learned scores and contains no provider client. It inherits
the baseline's centralized received observation; it does not claim CTDE.
"""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .closed_loop.candidate_batch import packed_scores
from .closed_loop.communication_study.physical_research_policies import (
    CandidateHMARL, HMARLConfig, _masked_logits,
)
from .closed_loop.communication_study.jev_iql import OutcomeCalibration, _probabilities


class JevHMARL(CandidateHMARL):
    algorithm = 'outcome-calibrated decision-teacher factorized MARL development v1'

    def __init__(self, config, *, calibration, teacher_weight=.1,
                 use_outcome_calibration=True):
        if not isinstance(calibration, OutcomeCalibration):
            raise ValueError('Physical outcome calibration required')
        if not math.isfinite(teacher_weight) or teacher_weight < 0:
            raise ValueError('Finite nonnegative teacher coefficient required')
        super().__init__(config)
        self.algorithm = type(self).algorithm
        self.calibration = calibration
        self.teacher_weight = teacher_weight
        self.use_outcome_calibration = use_outcome_calibration

    def preference_objective(self, scores, valid, annotations):
        """KL on the exact annotated subset, averaged over the full batch.

        Invalid/stale transition binding is rejected by load_evidence before
        training. No claim is made that high model confidence proves low
        physical cost. Unsupported calibration bins contribute zero gradient.
        """
        zero = scores[valid].sum() * 0.
        if self.teacher_weight == 0:
            return zero, 0
        if len(annotations) != len(scores):
            raise ValueError('Teacher batch size differs')
        losses = []
        for row, ann in enumerate(annotations):
            if ann is None:
                continue
            if ann.get('split') != 'training' or ann.get('model') != self.calibration.model:
                raise ValueError('Teacher model or training split differs')
            raw_indices = ann.get('candidate_indices', [])
            if any(type(x) is not int for x in raw_indices):
                raise ValueError('Candidate indices must be integers')
            indices = torch.tensor(raw_indices, dtype=torch.long, device=scores.device)
            if (indices.ndim != 1 or len(indices) < 2
                    or len(set(raw_indices)) != len(indices)
                    or (indices < 0).any() or (indices >= scores.shape[1]).any()
                    or not valid[row, indices].all()):
                raise ValueError('Teacher choices must belong to this valid catalogue')
            abstain = float(ann.get('abstain_probability', 0.))
            if not math.isfinite(abstain) or not 0 <= abstain <= 1:
                raise ValueError('Invalid abstention probability')
            if ann.get('abstained', False):
                continue
            probabilities = ann['probabilities']
            target = _probabilities(probabilities, self.calibration.temperature
                                    if self.use_outcome_calibration else 1.)
            if len(target) != len(indices):
                raise ValueError('Teacher probability width differs')
            weight = (self.calibration.weight(probabilities)
                      if self.use_outcome_calibration else 1.) * (1. - abstain)
            if weight <= 0:
                continue
            target = torch.as_tensor(target, device=scores.device, dtype=scores.dtype)
            losses.append(weight * F.kl_div(
                F.log_softmax(scores[row, indices], dim=0), target, reduction='sum'))
        return (torch.stack(losses).sum() / len(scores), len(losses)) if losses else (zero, 0)

    def update(self, batch, returns, *, available_only=False):
        states, actions, valid = batch['states'], batch['candidates'], batch['valid']
        chosen = batch['chosen']
        returns = returns.to(states)
        if actions.ndim == 2:
            scores = packed_scores(states, actions, valid, self.scores)
        elif available_only:
            rows, cols = valid.nonzero(as_tuple=True)
            scores = torch.full(valid.shape, -torch.inf, dtype=states.dtype, device=states.device)
            scores[rows, cols] = self.scores(states[rows], actions[rows, cols][:, None, :])[:, 0]
        else:
            scores = self.scores(states, actions)
        scores = _masked_logits(scores, valid)
        if not valid.gather(1, chosen[:, None]).all():
            raise ValueError('Recorded action must be valid')
        selected = scores.gather(1, chosen[:, None]).squeeze(1)
        values = self.value(states).squeeze(1)
        advantage = (returns - values.detach()).clamp(-10., 10.)
        weights = torch.exp(advantage.clamp(max=3.)).detach()
        actor = (F.cross_entropy(scores, chosen, reduction='none') * weights).mean()
        critic = F.smooth_l1_loss(selected, returns)
        value = F.smooth_l1_loss(values, returns)
        teacher, admitted = self.preference_objective(
            scores, valid, batch.get('teacher_annotations', [None] * len(states)))
        loss = (critic + self.config.value_weight * value + self.config.behavior_weight * actor
                + self.teacher_weight * teacher)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        metrics = {k: float(v.detach()) for k, v in dict(
            loss=loss, critic_loss=critic, value_loss=value,
            actor_loss=actor, teacher_loss=teacher).items()}
        return dict(metrics, teacher_admitted=admitted)

    def save(self, path, *, evidence):
        required = ('dataset_sha256', 'encoder_sha256', 'annotations_sha256',
                    'requests_sha256', 'calibration_sha256')
        if (evidence.get('contains_test_or_validation_data') is not False
                or evidence.get('reward_metric') != 'critical_ens_kwh'
                or any(not evidence.get(k) for k in required)
                or evidence.get('calibration_sha256') != self.calibration.evidence_sha256):
            raise ValueError('Bound training corpus, annotations and physical calibration required')
        torch.save(dict(algorithm=self.algorithm, config=asdict(self.config),
                        calibration=asdict(self.calibration), teacher_weight=self.teacher_weight,
                        use_outcome_calibration=self.use_outcome_calibration,
                        state_dict=self.state_dict(), optimizer_state=self.optimizer.state_dict(),
                        updates=self.updates, evidence=evidence), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('algorithm') != cls.algorithm:
            raise ValueError('Decision-teacher MARL identity mismatch')
        model = cls(HMARLConfig(**data['config']),
                    calibration=OutcomeCalibration(**data['calibration']),
                    teacher_weight=data['teacher_weight'],
                    use_outcome_calibration=data['use_outcome_calibration'])
        model.load_state_dict(data['state_dict'])
        model.optimizer.load_state_dict(data['optimizer_state'])
        model.updates = data['updates']
        return model, data['evidence']


class JevHMARLPolicy:
    """Development evaluator adapter. Formal admission remains a separate gate."""
    def __init__(self, checkpoint, encoder, *, checkpoint_sha256, dataset_sha256):
        checkpoint = Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256:
            raise ValueError('Checkpoint digest differs')
        self.model, evidence = JevHMARL.load(checkpoint)
        if (evidence.get('dataset_sha256') != dataset_sha256
                or evidence.get('encoder_sha256') != encoder.sha256
                or evidence.get('contains_test_or_validation_data') is not False
                or evidence.get('reward_metric') != 'critical_ens_kwh'):
            raise ValueError('Training/encoder provenance differs')
        self.model.eval()
        self.encoder = encoder
        self.evidence = evidence
        self.last_decision_trace = {}

    def select(self, observation, joints):
        if not joints:
            raise ValueError('Nonempty admitted joint catalogue required')
        state, actions = self.encoder.encode(observation, joints)
        with torch.no_grad():
            scores = self.model.scores(torch.as_tensor(state)[None, :],
                                       torch.as_tensor(actions)[None, :, :])[0]
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite learned decision scores')
        index = int(scores.argmax())
        self.last_decision_trace = dict(selected_index=index, candidate_count=len(joints),
            teacher_model=self.model.calibration.model, execution_uses_cloud_api=False,
            training_updates=self.model.updates, formal_eligible=False)
        return deepcopy(joints[index])
