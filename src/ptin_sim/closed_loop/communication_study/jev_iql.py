"""Outcome-calibrated frozen-teacher preferences on a categorical IQL actor.

Teacher requests never instantiate inside this module. Calibration uses new
calibration fault cases; policy updates use training cases only. Risk is the
range-normalized excess critical ENS under a declared physical continuation,
not an assertion that provider confidence is an empirical error probability.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ..publication_iql import CandidateIQL, IQLConfig


def _probabilities(values, temperature=1.):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values)<2 or not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:
        raise ValueError('Finite nonnegative multi-choice probabilities required')
    logits = np.log(np.maximum(values / values.sum(), 1e-12))/temperature
    probs = np.exp(logits-logits.max())
    return probs/probs.sum()


@dataclass(frozen=True)
class OutcomeCalibration:
    model: str
    temperature: float
    risk_bins: tuple
    evidence_sha256: str
    metric: str = 'critical_ens_kwh'
    risk_definition: str = 'range_normalized_excess_critical_ens'

    def __post_init__(self):
        if self.metric!='critical_ens_kwh' or not math.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError('Invalid outcome calibration')
        if len(self.evidence_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.evidence_sha256):
            raise ValueError('Calibration needs a physical-evidence SHA256')
        if not self.model or len(self.risk_bins)!=4:
            raise ValueError('Pinned teacher and four risk bins required')
        for risk in self.risk_bins:
            if risk is not None and (not math.isfinite(risk) or not 0<=risk<=1):
                raise ValueError('Invalid calibrated risk')

    def weight(self, probabilities):
        p = _probabilities(probabilities, self.temperature)
        risk = self.risk_bins[min(3, int(np.searchsorted([.5,.75,.9], p.max(), side='right')))]
        return 0. if risk is None else max(0., 1.-risk)


def fit_outcome_calibration(rows, *, model, evidence_sha256):
    """Fit temperature and case-bootstrap upper risk, with unsupported bins off.

Rows contain full candidate outcome vectors on calibration cases only. A bin
needs at least three fault cases. Bootstrap intervals are conditional on the
declared continuation/rollout scenarios, not a universal safety guarantee.
"""
    if not rows:
        raise ValueError('Physical candidate outcomes required')
    prepared=[]
    for row in rows:
        if not row['case_id'].startswith('validation_jev_'):
            raise ValueError('New held-out calibration fault cases required')
        p = _probabilities(row['probabilities'])
        loss = np.asarray(row['critical_ens_kwh'], dtype=float)
        if loss.shape!=p.shape or not np.isfinite(loss).all() or (loss<0).any():
            raise ValueError('One nonnegative physical outcome per candidate required')
        best = np.isclose(loss,loss.min(),rtol=0.,atol=1e-6).astype(float)
        best /= best.sum()
        regret = (loss-loss.min())/max(float(np.ptp(loss)),1e-6)
        prepared.append((row['case_id'],p,best,regret))
    def case_mean(items):
        groups={}
        for case,value in items:groups.setdefault(case,[]).append(value)
        return float(np.mean([np.mean(values) for values in groups.values()]))
    def loss_at(temperature):
        return case_mean([(case,float(-(best*np.log(_probabilities(p,temperature))).sum()))
                          for case,p,best,_ in prepared])
    temperature = min((.5,1.,2.,4.),key=loss_at)
    buckets=[{} for _ in range(4)]
    for case,p,_,regret in prepared:
        calibrated=_probabilities(p,temperature)
        index=min(3,int(np.searchsorted([.5,.75,.9],calibrated.max(),side='right')))
        # Distillation follows a distribution, so measure its expected regret.
        buckets[index].setdefault(case,[]).append(float(calibrated@regret))
    rng=np.random.default_rng(20260928);risks=[]
    for bucket in buckets:
        values=np.array([np.mean(v) for _,v in sorted(bucket.items())])
        if len(values)<3:
            risks.append(None);continue
        sampled=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
        risks.append(float(np.clip(np.quantile(sampled,.95),0.,1.)))
    return OutcomeCalibration(model,temperature,tuple(risks),evidence_sha256)


class TeacherIQL(CandidateIQL):
    algorithm='outcome-calibrated frozen-teacher categorical IQL v1'

    def __init__(self,config,*,calibration,teacher_weight=.1,use_outcome_calibration=True):
        if not isinstance(calibration,OutcomeCalibration) or not math.isfinite(teacher_weight) or teacher_weight<0:
            raise ValueError('Physical calibration and finite nonnegative teacher coefficient required')
        super().__init__(config)
        self.calibration=calibration;self.teacher_weight=teacher_weight
        self.use_outcome_calibration=use_outcome_calibration
        self.last_teacher_loss=0.;self.last_teacher_admitted=0

    def actor_objective(self,batch,advantages):
        base=super().actor_objective(batch,advantages)
        self.last_teacher_loss=0.;self.last_teacher_admitted=0
        if self.teacher_weight==0:return base
        annotations=batch.get('teacher_annotations',[None]*len(batch['states']))
        if len(annotations)!=len(batch['states']):raise ValueError('Teacher batch mismatch')
        logits=self.candidate_logits(batch['states'],batch['candidates'])
        losses=[]
        for i,annotation in enumerate(annotations):
            if annotation is None or annotation.get('abstained',False):continue
            if annotation.get('split')!='training' or annotation['model']!=self.calibration.model:
                raise ValueError('Teacher model or training split differs')
            indices=torch.as_tensor(annotation['candidate_indices'],device=logits.device,dtype=torch.long)
            if (indices.ndim!=1 or len(indices)<2 or len(set(indices.tolist()))!=len(indices)
                    or (indices<0).any() or (indices>=logits.shape[1]).any()
                    or not batch['valid'][i,indices].all()):
                raise ValueError('Teacher choices must belong to the admitted catalogue')
            p=annotation['probabilities']
            target=_probabilities(p,self.calibration.temperature if self.use_outcome_calibration else 1.)
            if len(target)!=len(indices):raise ValueError('Teacher probability width differs')
            abstain=float(annotation.get('abstain_probability',0.))
            if not math.isfinite(abstain) or not 0<=abstain<=1:raise ValueError('Invalid abstention probability')
            weight=(self.calibration.weight(p) if self.use_outcome_calibration else 1.)*(1.-abstain)
            if weight<=0:continue
            loss=F.kl_div(F.log_softmax(logits[i,indices],dim=0),
                torch.as_tensor(target,device=logits.device,dtype=logits.dtype),reduction='sum')*weight
            losses.append(loss)
        if not losses:return base
        auxiliary=torch.stack(losses).sum()/len(logits)
        self.last_teacher_loss=float(auxiliary.detach());self.last_teacher_admitted=len(losses)
        return base+self.teacher_weight*auxiliary

    def save(self,path,*,evidence):
        if (evidence.get('contains_test_or_validation_data') is not False
                or evidence.get('reward_metric')!='critical_ens_kwh'
                or not evidence.get('dataset_sha256')):
            raise ValueError('Training-only critical-ENS corpus provenance required')
        torch.save(dict(algorithm=self.algorithm,config=asdict(self.config),state_dict=self.state_dict(),
            calibration=asdict(self.calibration),teacher_weight=self.teacher_weight,
            use_outcome_calibration=self.use_outcome_calibration,updates=self.updates,evidence=evidence,
            optimizer_states=dict(q=self.q_optimizer.state_dict(),v=self.v_optimizer.state_dict(),
                                  actor=self.actor_optimizer.state_dict())),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('algorithm')!=cls.algorithm:raise ValueError('Teacher IQL identity mismatch')
        model=cls(IQLConfig(**data['config']),calibration=OutcomeCalibration(**data['calibration']),
            teacher_weight=data['teacher_weight'],use_outcome_calibration=data['use_outcome_calibration'])
        model.load_state_dict(data['state_dict']);model.updates=data['updates']
        model.q_optimizer.load_state_dict(data['optimizer_states']['q'])
        model.v_optimizer.load_state_dict(data['optimizer_states']['v'])
        model.actor_optimizer.load_state_dict(data['optimizer_states']['actor'])
        return model,data['evidence']


class TeacherIQLPolicy:
    """Local deployment adapter; contains no teacher client or credential lookup."""
    def __init__(self,checkpoint,encoder,*,checkpoint_sha256,dataset_sha256):
        from .physical_encoding import PhysicalIQLPolicy
        checkpoint=Path(checkpoint)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=checkpoint_sha256:
            raise ValueError('Teacher IQL checkpoint digest differs')
        model,evidence=TeacherIQL.load(checkpoint)
        if evidence.get('dataset_sha256')!=dataset_sha256 or evidence.get('reward_metric')!='critical_ens_kwh':
            raise ValueError('Teacher IQL corpus/objective provenance differs')
        self.policy=PhysicalIQLPolicy(encoder,model,checkpoint_evidence=evidence)
        self.last_decision_trace={}

    def select(self,observation,joints):
        result=self.policy.select(observation,joints)
        self.last_decision_trace=dict(self.policy.last_decision_trace,
            teacher_model=self.policy.model.calibration.model,execution_uses_cloud_api=False)
        return result
