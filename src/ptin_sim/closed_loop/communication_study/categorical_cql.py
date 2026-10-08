"""Discrete CQL(H) on explicit, variable-size joint candidate catalogues.

Kumar et al., NeurIPS 2020, https://arxiv.org/abs/2006.04779.
The conservative log-sum-exp minus dataset-action Q term augments a
one-step Bellman loss. Double-DQN selection and Polyak targets are declared
implementation choices. This is the discrete candidate adaptation, not a
continuous-action CQL-SAC reproduction.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from ..candidate_batch import packed_scores


@dataclass(frozen=True)
class CQLConfig:
    state_dim:int
    action_dim:int
    hidden_dim:int=64
    learning_rate:float=3e-4
    discount:float=.99
    conservative_weight:float=1.
    target_rate:float=.005


def conservative_penalty(values,valid,chosen):
    if values.ndim!=2 or valid.shape!=values.shape or valid.dtype!=torch.bool:
        raise ValueError('Boolean candidate mask required')
    if chosen.shape!=(len(values),) or chosen.dtype!=torch.long:
        raise ValueError('One executed action index per row required')
    if not ((chosen>=0)&(chosen<values.shape[1])).all() or not valid.gather(1,chosen[:,None]).all():
        raise ValueError('Recorded action is unavailable')
    if not torch.isfinite(values[valid]).all():raise ValueError('Nonfinite valid Q values')
    return (torch.logsumexp(values.masked_fill(~valid,-torch.inf),dim=1)
            -values.gather(1,chosen[:,None]).squeeze(1)).mean()


class CandidateCQL(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if (min(config.state_dim,config.action_dim,config.hidden_dim)<=0
                or not all(math.isfinite(x) for x in (config.learning_rate,config.discount,
                       config.conservative_weight,config.target_rate))
                or config.learning_rate<=0 or not 0<=config.discount<=1
                or config.conservative_weight<0 or not 0<config.target_rate<=1):
            raise ValueError('Invalid CQL configuration')
        self.q=nn.Sequential(nn.Linear(config.state_dim+config.action_dim,config.hidden_dim),nn.ReLU(),
            nn.Linear(config.hidden_dim,config.hidden_dim),nn.ReLU(),nn.Linear(config.hidden_dim,1))
        self.target_q=deepcopy(self.q).requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.q.parameters(),lr=config.learning_rate)
        self.updates=0

    def values(self,states,candidates,*,target=False):
        if (states.ndim!=2 or states.shape[1]!=self.config.state_dim or candidates.ndim!=3
                or candidates.shape[0]!=len(states) or candidates.shape[2]!=self.config.action_dim):
            raise ValueError('State/action schema mismatch')
        if not torch.isfinite(states).all() or not torch.isfinite(candidates).all():
            raise ValueError('Nonfinite state/action input')
        inputs=torch.cat((states[:,None,:].expand(-1,candidates.shape[1],-1),candidates),dim=-1)
        return (self.target_q if target else self.q)(inputs).squeeze(-1)

    def update(self,batch):
        def evaluate(*,next_state=False,target=False):
            prefix='next_' if next_state else ''
            states,actions,valid=(batch[prefix+k] for k in ('states','candidates','valid'))
            score=lambda s,a:self.values(s,a,target=target)
            return packed_scores(states,actions,valid,score) if actions.ndim==2 else score(states,actions)
        values=evaluate()
        conservative=conservative_penalty(values,batch['valid'],batch['chosen'])
        n=len(values);rewards=batch['rewards'];terminal=batch['terminated']
        if (rewards.shape!=(n,) or terminal.shape!=(n,) or not torch.isfinite(rewards).all()
                or not ((terminal==0)|(terminal==1)).all()):raise ValueError('Invalid rewards/termination')
        with torch.no_grad():
            next_values=evaluate(next_state=True)
            mask=batch['next_valid']
            if mask.dtype!=torch.bool or mask.shape!=next_values.shape or not mask.any(dim=1).all():
                raise ValueError('Next-state candidates are required, including terminal wait')
            selected=next_values.masked_fill(~mask,-torch.inf).argmax(dim=1)
            target_values=evaluate(next_state=True,target=True)
            target=rewards+self.config.discount*(1-terminal)*target_values.gather(1,selected[:,None]).squeeze(1)
        chosen_values=values.gather(1,batch['chosen'][:,None]).squeeze(1)
        bellman=(chosen_values-target).square().mean()
        loss=bellman+self.config.conservative_weight*conservative
        self.optimizer.zero_grad();loss.backward();self.optimizer.step()
        with torch.no_grad():
            for target_param,param in zip(self.target_q.parameters(),self.q.parameters()):
                target_param.lerp_(param,self.config.target_rate)
        self.updates+=1
        return dict(loss=float(loss.detach()),bellman_loss=float(bellman.detach()),
                    conservative_loss=float(conservative.detach()),updates=self.updates)

    def save(self,path,*,evidence):
        if not evidence.get('dataset_sha256') or not evidence.get('encoder_sha256'):
            raise ValueError('Dataset and encoder identities required')
        torch.save(dict(algorithm='discrete_candidate_CQL_H_double_DQN',config=asdict(self.config),
            state_dict=self.state_dict(),optimizer_state=self.optimizer.state_dict(),
            updates=self.updates,evidence=evidence),path)

    @classmethod
    def load(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(CQLConfig(**saved['config']));model.load_state_dict(saved['state_dict'])
        model.optimizer.load_state_dict(saved['optimizer_state']);model.updates=saved['updates']
        return model,saved['evidence']
