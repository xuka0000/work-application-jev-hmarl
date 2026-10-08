"""Distinct learned and planning policies for the complete physical study."""
from __future__ import annotations

from dataclasses import asdict,dataclass
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .physical_actions import action_id,received_state
from .physical_encoding import DEVICE_ACTION_FEATURES


def schema_sha256(state_features,action_features,*,method):
    raw=json.dumps(dict(method=method,state_features=list(state_features),
                        action_features=list(action_features)),sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _masked_logits(values,valid):
    if values.shape!=valid.shape or valid.dtype!=torch.bool or not valid.any(dim=1).all():
        raise ValueError('Every batch row needs at least one real candidate')
    return values.masked_fill(~valid,-torch.inf)


@dataclass(frozen=True)
class SafeGPINNConfig:
    state_dim:int
    action_dim:int
    graph_state_dim:int=4
    graph_action_dim:int=13
    hidden_dim:int=128
    learning_rate:float=3e-4
    graph_weight:float=.35
    residual_weight:float=.20
    behavior_weight:float=.25
    pair_weight:float=.10
    value_weight:float=.50


class CandidateSafeGPINN(nn.Module):
    """Graph residual candidate evaluator downstream of a causal hard mask."""
    algorithm='SAFE-GPINN physical graph residual candidate evaluator v1'
    def __init__(self,config:SafeGPINNConfig,*,variant='full'):
        super().__init__();self.config=config;self.variant=variant;h=config.hidden_dim
        if config.graph_state_dim>=config.state_dim or config.graph_action_dim>=config.action_dim:
            raise ValueError('Graph dimensions must be strict feature subsets')
        base_dim=(config.state_dim-config.graph_state_dim)+(config.action_dim-config.graph_action_dim)
        self.base=nn.Sequential(nn.Linear(base_dim,h),nn.ReLU(),
            nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
        self.graph=nn.Sequential(nn.Linear(config.graph_state_dim+config.graph_action_dim,h//2),nn.Tanh(),
            nn.Linear(h//2,1))
        self.residual=nn.Sequential(nn.Linear(config.graph_state_dim+config.graph_action_dim,h//2),nn.ReLU(),
            nn.Linear(h//2,1),nn.Softplus())
        self.value=nn.Sequential(nn.Linear(config.state_dim,h),nn.ReLU(),nn.Linear(h,1))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=config.learning_rate)
        self.updates=0

    def components(self,states,actions):
        if states.ndim!=2 or actions.ndim!=3 or states.shape[0]!=actions.shape[0]:
            raise ValueError('Expected batched states and variable candidates')
        expanded=states[:,:-self.config.graph_state_dim][:,None,:].expand(-1,actions.shape[1],-1)
        base_actions=actions[:,:,:-self.config.graph_action_dim]
        base=self.base(torch.cat((expanded,base_actions),dim=-1)).squeeze(-1)
        gs=states[:,-self.config.graph_state_dim:]
        ga=actions[:,:,-self.config.graph_action_dim:]
        gx=gs[:,None,:].expand(-1,actions.shape[1],-1)
        graph=self.graph(torch.cat((gx,ga),dim=-1)).squeeze(-1)
        residual=self.residual(torch.cat((gx,ga),dim=-1)).squeeze(-1)
        return base,graph,residual

    def scores(self,states,actions):
        base,graph,residual=self.components(states,actions)
        if 'no_graph' in self.variant:graph=torch.zeros_like(graph)
        if 'no_residual' in self.variant:residual=torch.zeros_like(residual)
        return base+self.config.graph_weight*graph-self.config.residual_weight*residual

    def objective(self,batch,returns):
        states,actions,valid=batch['states'],batch['candidates'],batch['valid']
        chosen=batch['chosen'];returns=returns.to(states)
        scores=_masked_logits(self.scores(states,actions),valid)
        selected=scores.gather(1,chosen[:,None]).squeeze(1)
        values=self.value(states).squeeze(1)
        advantage=(returns-values.detach()).clamp(-10.,10.)
        weights=torch.exp(advantage.clamp(max=3.)).detach()
        behavior=(F.cross_entropy(scores,chosen,reduction='none')*weights).mean()
        q_loss=F.smooth_l1_loss(selected,returns)
        value_loss=F.smooth_l1_loss(values,returns)
        base,graph,residual=self.components(states,actions)
        ga=actions[:,:,-self.config.graph_action_dim:]
        # Causal physical-inconsistency proxy: stale/long work and insufficient
        # energy increase the residual; repair/close/load evidence reduces it.
        duration=ga[...,12];age=ga[...,11];energy=ga[...,10]
        useful=ga[...,4]+ga[...,6]+ga[...,7]+ga[...,9]
        target_residual=(age+duration+(1.-energy).clamp_min(0.)-.25*useful).clamp_min(0.)
        residual_loss=F.smooth_l1_loss(residual[valid],target_residual[valid])
        # Pair only high-return behavior actions against the best currently
        # scored alternative. This is an AWR ranking term, not an oracle label.
        alternate=scores.clone();alternate.scatter_(1,chosen[:,None],-torch.inf)
        alt=alternate.max(dim=1).values
        pair=F.softplus(alt-selected)
        pair_loss=(pair*(advantage>0).float()).sum()/(advantage>0).sum().clamp_min(1)
        loss=q_loss+self.config.value_weight*value_loss+self.config.behavior_weight*behavior
        if 'no_residual' not in self.variant:loss=loss+self.config.residual_weight*residual_loss
        loss=loss+self.config.pair_weight*pair_loss
        return loss,dict(loss=loss,q_loss=q_loss,value_loss=value_loss,
                        behavior_loss=behavior,residual_loss=residual_loss,pair_loss=pair_loss)

    def update(self,batch,returns):
        loss,components=self.objective(batch,returns)
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),10.);self.optimizer.step();self.updates+=1
        return {k:float(v.detach()) for k,v in components.items()}

    def save(self,path,*,evidence):
        torch.save(dict(algorithm=self.algorithm,variant=self.variant,config=asdict(self.config),
            updates=self.updates,state_dict=self.state_dict(),evidence=evidence),path)

    @classmethod
    def load(cls,source):
        data=torch.load(source,map_location='cpu',weights_only=True)
        if data.get('algorithm')!=cls.algorithm:raise ValueError('SAFE-GPINN checkpoint identity mismatch')
        model=cls(SafeGPINNConfig(**data['config']),variant=data['variant'])
        model.load_state_dict(data['state_dict']);model.updates=data['updates'];return model,data


@dataclass(frozen=True)
class HMARLConfig:
    state_dim:int
    action_dim:int
    units:int=9
    per_unit_dim:int=15
    hidden_dim:int=96
    learning_rate:float=3e-4
    behavior_weight:float=.25
    value_weight:float=.50
    identity_dim:int=0


class CandidateHMARL(nn.Module):
    """Factorized device actors with a nonnegative centralized mixer."""
    algorithm='physical hierarchical MARL factorized actor critic v1'
    def __init__(self,config:HMARLConfig):
        super().__init__();self.config=config;h=config.hidden_dim
        if config.identity_dim<0 or config.action_dim!=config.units*(config.per_unit_dim+config.identity_dim)+1:
            raise ValueError('HMARL action slots do not match the physical device partition')
        if config.identity_dim:self.algorithm='physical hierarchical MARL target-aware factorized actor critic v2'
        self.device_id=nn.Embedding(config.units,8)
        self.local=nn.Sequential(nn.Linear(config.state_dim+config.per_unit_dim+config.identity_dim+8,h),nn.ReLU(),
            nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,1))
        self.mix=nn.Sequential(nn.Linear(config.state_dim,h//2),nn.ReLU(),nn.Linear(h//2,config.units))
        self.bias=nn.Sequential(nn.Linear(config.state_dim,h//2),nn.ReLU(),nn.Linear(h//2,1))
        self.value=nn.Sequential(nn.Linear(config.state_dim,h),nn.ReLU(),nn.Linear(h,1))
        self.optimizer=torch.optim.Adam(self.parameters(),lr=config.learning_rate);self.updates=0

    def scores(self,states,actions):
        batch,width,_=actions.shape;c=self.config
        slots=actions[...,:c.units*c.per_unit_dim].reshape(batch,width,c.units,c.per_unit_dim)
        if c.identity_dim:
            identities=actions[...,c.units*c.per_unit_dim+1:].reshape(batch,width,c.units,c.identity_dim)
            slots=torch.cat((slots,identities),dim=-1)
        s=states[:,None,None,:].expand(-1,width,c.units,-1)
        ids=self.device_id(torch.arange(c.units,device=states.device))[None,None,:,:].expand(batch,width,-1,-1)
        local=self.local(torch.cat((s,slots,ids),dim=-1)).squeeze(-1)
        weights=F.softplus(self.mix(states))[:,None,:]
        return (weights*local).sum(dim=-1)+self.bias(states)

    def update(self,batch,returns,*,available_only=False):
        states,actions,valid=batch['states'],batch['candidates'],batch['valid']
        chosen=batch['chosen'];returns=returns.to(states)
        if actions.ndim==2:
            from ..candidate_batch import packed_scores
            scores=packed_scores(states,actions,valid,self.scores)
        elif available_only:
            rows,columns=valid.nonzero(as_tuple=True)
            scores=torch.full(valid.shape,-torch.inf,dtype=states.dtype,device=states.device)
            scores[rows,columns]=self.scores(states[rows],actions[rows,columns][:,None,:])[:,0]
        else:scores=self.scores(states,actions)
        scores=_masked_logits(scores,valid)
        selected=scores.gather(1,chosen[:,None]).squeeze(1);values=self.value(states).squeeze(1)
        advantage=(returns-values.detach()).clamp(-10.,10.);weights=torch.exp(advantage.clamp(max=3.)).detach()
        actor=(F.cross_entropy(scores,chosen,reduction='none')*weights).mean()
        critic=F.smooth_l1_loss(selected,returns);value=F.smooth_l1_loss(values,returns)
        loss=critic+self.config.value_weight*value+self.config.behavior_weight*actor
        self.optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.parameters(),10.)
        self.optimizer.step();self.updates+=1
        return {k:float(v.detach()) for k,v in dict(loss=loss,critic_loss=critic,value_loss=value,
                                                     actor_loss=actor).items()}

    def save(self,path,*,evidence):
        torch.save(dict(algorithm=self.algorithm,config=asdict(self.config),updates=self.updates,
                        state_dict=self.state_dict(),evidence=evidence),path)

    @classmethod
    def load(cls,source):
        data=torch.load(source,map_location='cpu',weights_only=True)
        model=cls(HMARLConfig(**data['config']))
        if data.get('algorithm')!=model.algorithm:raise ValueError('HMARL checkpoint identity mismatch')
        model.load_state_dict(data['state_dict'])
        model.updates=data['updates'];return model,data


def causal_safe_mask(actions,encoder,*,variant):
    valid=torch.ones(actions.shape[:-1],dtype=torch.bool,device=actions.device)
    if 'no_shield' in variant:return valid
    graph=actions[...,-13:]
    active=graph[...,5]>0
    energy=graph[...,10];age=graph[...,11];duration=graph[...,12]
    max_age=float(encoder.resources['observation_max_age_s'])/encoder.horizon
    valid &= (~active) | ((energy>0.) & (age<=max_age+1e-7) & (duration<=1.+1e-7))
    # The first candidate is the explicit wait fallback and must remain valid.
    valid[...,0]=True
    return valid


class PhysicalLearnedResearchPolicy:
    def __init__(self,checkpoint,encoder,*,method,variant,checkpoint_sha256,
                 training_dataset_sha256,expected_updates):
        raw=Path(checkpoint).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=checkpoint_sha256:raise ValueError('Checkpoint digest mismatch')
        load=io.BytesIO(raw)
        if method=='safe_gpinn':model,data=CandidateSafeGPINN.load(load)
        elif method in {'jev_distilled','llm_distilled'}:
            from .distilled_policy import DistilledCandidatePolicy
            from .decision_teacher import JEV_MODEL,LLM_MODEL
            model,data=DistilledCandidatePolicy.load(load)
            expected_teacher=JEV_MODEL if method=='jev_distilled' else LLM_MODEL
            if model.calibration.model!=expected_teacher:
                raise ValueError('Student teacher identity mismatch')
        elif method=='hmarl':model,data=CandidateHMARL.load(load)
        else:raise ValueError('Unknown learned research policy')
        evidence=data['evidence'];expected_schema=schema_sha256(encoder.state_features,encoder.action_features,method=method)
        if (evidence.get('dataset_sha256')!=training_dataset_sha256
                or evidence.get('feature_schema_sha256')!=expected_schema
                or data['updates']!=expected_updates):
            raise ValueError('Checkpoint dataset/schema/update provenance mismatch')
        if method in {'safe_gpinn','jev_distilled','llm_distilled'} and data['variant']!=variant:
            raise ValueError('SAFE-GPINN component variant checkpoint mismatch')
        self.model=model.eval().requires_grad_(False);self.encoder=encoder
        self.method,self.variant=method,variant
        self.identity=dict(method=method,variant=variant,checkpoint_sha256=checkpoint_sha256,
            dataset_sha256=training_dataset_sha256,feature_schema_sha256=expected_schema,
            updates=expected_updates,algorithm=data['algorithm'])
        self.last_decision_trace=None

    @torch.inference_mode()
    def select(self,observation,joints):
        state,actions=self.encoder.encode(observation,joints)
        scores=self.model.scores(state[None],actions[None])[0]
        if self.method in {'safe_gpinn','jev_distilled','llm_distilled'}:
            valid=causal_safe_mask(actions[None],self.encoder,variant=self.variant)[0]
            scores=scores.masked_fill(~valid,-torch.inf)
        else:valid=torch.ones(len(joints),dtype=torch.bool)
        index=int(scores.argmax())
        finite=scores[valid];top=torch.topk(finite,min(3,len(finite))).values.tolist()
        self.last_decision_trace=dict(**self.identity,selected_index=index,candidate_count=len(joints),
            admitted_count=int(valid.sum()),selected_score=float(scores[index]),top_scores=top,
            score_gap=float(top[0]-top[1]) if len(top)>1 else None,
            selected_joint_sha256=[action_id(a) for a in joints[index]],
            catalogue_sha256=hashlib.sha256(json.dumps([[action_id(a) for a in row] for row in joints],
                separators=(',',':')).encode()).hexdigest())
        return deepcopy(joints[index])


class PhysicalPlanningPolicy:
    """Causal local adaptations of RH-MPC and two-stage PTIN assignment."""
    def __init__(self,scenario,resources,*,method):
        if method not in {'ewmac_rh_mpc','ptin_two_stage_adapted'}:raise ValueError('Unknown planner')
        self.scenario,self.resources,self.method=scenario,dict(resources),method
        self.edges={e.edge_id:e for e in scenario.pdn_edges};self.nodes={n.node_id:n for n in scenario.pdn_nodes}
        self.last_decision_trace=None

    def _action_value(self,action,known):
        if action['kind']=='wait':return 0.
        target=action.get('target');node=self.nodes[self.edges[target].to_node] if target in self.edges else None
        load=(node.active_power_kw*(2. if node.criticality=='critical' else 1.)) if node else 0.
        duration=float(action.get('duration_s',action.get('route',{}).get('estimated_duration_s',300.)))
        distance=float(action.get('route',{}).get('distance_km',0.))
        if action.get('operation')=='repair':load*=1.15
        if action.get('operation')=='close':load*=1.30
        if action['kind']=='uav_relay':load=max((self.nodes[e.to_node].active_power_kw for e in self.edges.values()
                                                if e.edge_id not in known['targets']),default=0.)*.35
        if action['kind']=='deploy' and action['route']['purpose']=='connect':load=max(load,50.)
        return load-duration/300.-distance

    def select(self,observation,joints):
        known=received_state(observation,max_age_s=self.resources['observation_max_age_s'])
        rows=[]
        remaining=sum(self.nodes[self.edges[k].to_node].active_power_kw for k,r in known['targets'].items()
                      if r['failed'] and not r['restored'])
        for joint in joints:
            immediate=sum(self._action_value(a,known) for a in joint)
            targets={a.get('target') for a in joint if a.get('target')}
            if self.method=='ewmac_rh_mpc':
                continuation=.25*max(0.,remaining-sum(self.nodes[self.edges[k].to_node].active_power_kw
                    for k in targets if k in self.edges))
                score=immediate+continuation
            else:
                # Stage 1 values task-device assignments; stage 2 is the common
                # actionability projection that produced this joint catalogue.
                diversity=len(targets)+len({a['route']['unit_id'] if a['kind']=='deploy' else a.get('unit_id')
                                            for a in joint if a['kind']!='wait'})
                score=immediate+2.*diversity
            rows.append(score)
        index=max(range(len(rows)),key=lambda i:(rows[i],-i))
        ordered=sorted(rows,reverse=True)[:3]
        self.last_decision_trace=dict(method=self.method,selected_index=index,candidate_count=len(joints),
            selected_score=rows[index],top_scores=ordered,
            score_gap=ordered[0]-ordered[1] if len(ordered)>1 else None,
            information_scope='received_reports_public_topology_no_simulator_lookahead',
            selected_joint_sha256=[action_id(a) for a in joints[index]])
        return deepcopy(joints[index])
