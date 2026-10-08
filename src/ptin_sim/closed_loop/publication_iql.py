"""IQL update equations with a categorical candidate policy.

Source: Kostrikov et al., ICLR 2022, and the authors' implicit_q_learning
actor.py, critic.py and learner.py. The categorical distribution replaces
their continuous policy. This is an explicit action-space adaptation, not
an assertion that historical PTIN rows contain the required action sets.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from .candidate_batch import packed_scores, packed_selected, validate_packed


@dataclass(frozen=True)
class IQLConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    discount: float = .99
    expectile: float = .7
    inverse_temperature: float = 3.
    target_rate: float = .005
    max_advantage_weight: float = 100.


def _network(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def expectile_loss(residual, expectile):
    return (torch.where(residual > 0, expectile, 1-expectile)*residual.square()).mean()


def masked_log_probabilities(logits, valid):
    if valid.dtype != torch.bool or logits.shape != valid.shape or logits.ndim != 2:
        raise ValueError('Each candidate needs one Boolean availability flag')
    if not valid.any(dim=1).all():
        raise ValueError('Every decision state needs at least one available action')
    if not torch.isfinite(logits[valid]).all():
        raise ValueError('Available-action logits must be finite')
    return torch.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=1)


def advantage_weighted_policy_loss(logits, valid, chosen, advantages, config):
    log_probs = masked_log_probabilities(logits, valid)
    if not valid.gather(1, chosen[:, None]).all():
        raise ValueError('The recorded executed action is unavailable in its state')
    # Clamp before exponentiation to avoid overflow without changing min(exp(A),100).
    weights = (advantages.detach()*config.inverse_temperature).clamp(
        max=math.log(config.max_advantage_weight)).exp()
    return -(weights*log_probs.gather(1, chosen[:, None]).squeeze(1)).mean()


def bellman_target(rewards, terminated, next_values, discount):
    return rewards + discount*(1-terminated)*next_values


class CandidateIQL(nn.Module):
    """Train on explicit same-state candidate sets and actually executed actions.

    A batch contains states [B,S], candidates [B,A,F], valid [B,A],
    chosen [B], rewards [B], next_states [B,S], terminated [B].
    Inputs must be normalized using training-only statistics before fitting.
    Terminated denotes an MDP terminal state, not a time-limit truncation.
    """
    def __init__(self, config):
        super().__init__()
        if not 0 < config.expectile < 1 or not 0 < config.target_rate <= 1:
            raise ValueError('Invalid expectile or target update rate')
        if not 0 <= config.discount <= 1 or config.max_advantage_weight <= 0:
            raise ValueError('Invalid discount or advantage cap')
        self.config = config
        width = config.state_dim + config.action_dim
        self.q1 = _network(width, config.hidden_dim)
        self.q2 = _network(width, config.hidden_dim)
        self.target_q1 = deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).requires_grad_(False)
        self.value = _network(config.state_dim, config.hidden_dim)
        self.actor = _network(width, config.hidden_dim)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.learning_rate)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.updates = 0

    def candidate_logits(self, states, candidates):
        expanded = states[:, None, :].expand(-1, candidates.shape[1], -1)
        return self.actor(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def probabilities(self, states, candidates, valid):
        return masked_log_probabilities(self.candidate_logits(states, candidates), valid).exp()

    @torch.no_grad()
    def update_targets(self):
        for target, current in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
            for target_parameter, parameter in zip(target.parameters(), current.parameters()):
                target_parameter.lerp_(parameter, self.config.target_rate)

    def update(self, batch):
        states, candidates, valid, chosen = (batch[k] for k in ('states', 'candidates', 'valid', 'chosen'))
        count = len(states)
        if states.shape != (count, self.config.state_dim) or batch['next_states'].shape != states.shape:
            raise ValueError('State dimensions do not match the declared observation schema')
        packed = candidates.ndim == 2
        if packed:
            validate_packed(candidates, valid, count=count, features=self.config.action_dim)
        elif candidates.ndim != 3 or candidates.shape[0] != count or candidates.shape[2] != self.config.action_dim:
            raise ValueError('Candidate dimensions do not match the declared action schema')
        if (not packed and valid.shape != candidates.shape[:2]) or chosen.shape != (count,) or chosen.dtype != torch.long:
            raise ValueError('Malformed candidate masks or executed action indices')
        if not ((chosen >= 0) & (chosen < valid.shape[1])).all():
            raise ValueError('Executed action index outside the candidate catalogue')
        for key in ('states', 'candidates', 'next_states', 'rewards', 'terminated'):
            if not torch.isfinite(batch[key]).all():
                raise ValueError(f'Non-finite batch field {key}')
        if batch['rewards'].shape != (count,) or batch['terminated'].shape != (count,):
            raise ValueError('Rewards and terminal flags must have one entry per transition')
        if not ((batch['terminated'] == 0) | (batch['terminated'] == 1)).all():
            raise ValueError('Terminal flags must be Boolean values')
        if valid.dtype != torch.bool or not valid.any(dim=1).all() or not valid.gather(1, chosen[:, None]).all():
            raise ValueError('Each executed action must belong to a nonempty available-action set')
        selected = (packed_selected(candidates, valid, chosen) if packed else
                    candidates[torch.arange(count, device=chosen.device), chosen])
        inputs = torch.cat((states, selected), dim=1)
        with torch.no_grad():
            target_min = torch.minimum(self.target_q1(inputs), self.target_q2(inputs)).squeeze(-1)
        # Match the authors' update order: V, actor, twin Q, target critics.
        self.v_optimizer.zero_grad()
        v_loss = expectile_loss(target_min-self.value(states).squeeze(-1), self.config.expectile)
        v_loss.backward()
        self.v_optimizer.step()
        with torch.no_grad():
            advantages = target_min-self.value(states).squeeze(-1)
        self.actor_optimizer.zero_grad()
        actor_loss = self.actor_objective(batch, advantages)
        actor_loss.backward()
        self.actor_optimizer.step()
        with torch.no_grad():
            target = bellman_target(batch['rewards'], batch['terminated'],
                self.value(batch['next_states']).squeeze(-1), self.config.discount)
        self.q_optimizer.zero_grad()
        q_loss = ((self.q1(inputs).squeeze(-1)-target).square()
                  + (self.q2(inputs).squeeze(-1)-target).square()).mean()
        q_loss.backward()
        self.q_optimizer.step()
        self.update_targets()
        self.updates += 1
        return dict(value_loss=float(v_loss.detach()), actor_loss=float(actor_loss.detach()),
                    critic_loss=float(q_loss.detach()), updates=self.updates)

    def actor_objective(self, batch, advantages):
        """Extension point retaining exactly one actor step per IQL update."""
        logits = (packed_scores(batch['states'], batch['candidates'], batch['valid'], self.candidate_logits)
                  if batch['candidates'].ndim == 2 else self.candidate_logits(batch['states'], batch['candidates']))
        return advantage_weighted_policy_loss(logits,
            batch['valid'], batch['chosen'], advantages, self.config)

    def save(self, path, *, evidence):
        if not evidence.get('dataset_sha256') or not evidence.get('observation_schema'):
            raise ValueError('Checkpoint needs dataset hash and observation schema')
        torch.save(dict(config=asdict(self.config), state_dict=self.state_dict(), updates=self.updates,
            algorithm='IQL with categorical candidate policy', evidence=evidence,
            optimizer_states=dict(q=self.q_optimizer.state_dict(), v=self.v_optimizer.state_dict(),
                                  actor=self.actor_optimizer.state_dict())), path)

    @classmethod
    def load(cls, path):
        record = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(IQLConfig(**record['config']))
        model.load_state_dict(record['state_dict'])
        model.updates = record['updates']
        model.q_optimizer.load_state_dict(record['optimizer_states']['q'])
        model.v_optimizer.load_state_dict(record['optimizer_states']['v'])
        model.actor_optimizer.load_state_dict(record['optimizer_states']['actor'])
        return model, record['evidence']
