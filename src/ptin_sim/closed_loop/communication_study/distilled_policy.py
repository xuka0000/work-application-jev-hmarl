"""Offline teacher-assisted policy with one optimizer step per update.

Inference uses local weights and the existing mask only; importing or loading
this policy does not instantiate a network client or require API credentials.
"""
from dataclasses import asdict
import math
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .decision_teacher import TeacherCalibration, distillation_loss, fit_calibration, JEV_MODEL, LLM_MODEL, value_hash
from .physical_research_policies import CandidateSafeGPINN, SafeGPINNConfig


def load_teacher_training_inputs(replay,annotations_path,requests_path,calibration_path,*,workspace,method):
    """Bind annotations to replay rows and fit calibration from outcome labels.

    Calibration JSON must reference a held-out physical evaluation artifact by
    path and digest. That artifact carries probabilities and evaluator labels.
    No teacher confidence or synthetic calibration is accepted as a substitute.
    """
    if any(p is None for p in (annotations_path,requests_path,calibration_path)):
        raise ValueError('Teacher annotations, bound requests and physical calibration are required')
    paths=[Path(p) for p in (annotations_path,requests_path,calibration_path)]
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    annotations=json.loads(paths[0].read_text(encoding='utf-8'))
    requests=[json.loads(line) for line in paths[1].read_text(encoding='utf-8').splitlines()]
    calibration_record=json.loads(paths[2].read_text(encoding='utf-8'))
    expected=JEV_MODEL if method=='jev_distilled' else LLM_MODEL
    corpus_hash=hashlib.sha256(replay.path.read_bytes()).hexdigest()
    if annotations.get('model')!=expected or annotations.get('requests_sha256')!=hashes[1] or annotations.get('source_manifest_sha256')!=corpus_hash:
        raise ValueError('Teacher annotations do not match the frozen training corpus and requests')
    outcome_path=(Path(workspace)/calibration_record['physical_outcomes_path']).resolve()
    if not outcome_path.is_relative_to(Path(workspace).resolve()):
        raise ValueError('Physical calibration evidence escapes workspace')
    outcome_hash=hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    if outcome_hash!=calibration_record['physical_outcomes_sha256']:
        raise ValueError('Physical calibration evidence changed')
    outcomes=json.loads(outcome_path.read_text(encoding='utf-8'))
    if (outcomes.get('status')!='completed_physical_candidate_evaluation' or outcomes.get('model')!=expected
            or outcomes.get('label_source')!='physical_evaluator' or not outcomes.get('evaluator_sha256')):
        raise ValueError('Independent physical calibration outcomes required')
    calibration=fit_calibration(outcomes['probabilities'],outcomes['labels'],case_ids=outcomes['case_ids'],
                                evidence_sha256=outcome_hash,model=expected)
    request_index={r['transition_index']:r for r in requests}
    if len(request_index)!=len(requests):raise ValueError('Duplicate teacher transition requests')
    labels={}
    for item in annotations['rows']:
        index=item['transition_index'];request=request_index[index];annotation=item['annotation']
        if index in labels or item['split']!='training' or request['split']!='training':
            raise ValueError('Teacher optimizer annotations must be unique training rows')
        shard_index,step=replay.index[index];shard=replay.metadata['shards'][shard_index]
        if (item['case_id']!=request['case_id'] or request['case_id']!=shard['case_id']
                or request['source_events_sha256']!=shard['source_events_sha256'] or step!=request['source_decision_step']):
            raise ValueError('Teacher annotation replay identity mismatch')
        envelope=request['request'];row=replay.row(index)
        actual=hashlib.sha256(row['state'].tobytes()+row['candidates'].tobytes()).hexdigest()
        if request.get('encoded_transition_sha256')!=actual:
            raise ValueError('Teacher request lacks an exact replay feature binding')
        if (envelope['request_sha256']!=value_hash(envelope['body'])
                or annotation['request_sha256']!=envelope['request_sha256']
                or annotation['candidate_ids']!=envelope['candidate_ids']
                or annotation['candidate_indices']!=envelope['candidate_indices'] or annotation['model']!=expected):
            raise ValueError('Teacher response does not match its candidate request')
        if envelope['full_candidate_count']!=len(row['candidates']):
            raise ValueError('Teacher and replay catalogues differ')
        labels[index]=annotation
    if not labels:raise ValueError('No verified teacher annotations')
    return labels,calibration,dict(annotations_sha256=hashes[0],requests_sha256=hashes[1],
        calibration_registration_sha256=hashes[2],physical_outcomes_sha256=outcome_hash,
        model=expected,annotated_training_transitions=len(labels))


class DistilledCandidatePolicy(CandidateSafeGPINN):
    algorithm='calibrated frozen-teacher physical candidate policy v1'

    def __init__(self,config,*,calibration,distillation_weight=.1,variant='full'):
        super().__init__(config,variant=variant)
        if calibration is None or not math.isfinite(distillation_weight) or distillation_weight<0:
            raise ValueError('Explicit calibration and finite nonnegative teacher weight required')
        self.calibration=calibration
        self.distillation_weight=distillation_weight

    def update(self,batch,returns,annotations):
        original,components=self.objective(batch,returns)
        scores=self.scores(batch['states'],batch['candidates'])
        auxiliary=distillation_loss(scores,batch['valid'],annotations,self.calibration)
        loss=original+self.distillation_weight*auxiliary
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(),10.)
        self.optimizer.step();self.updates+=1
        return {k:float(v.detach()) for k,v in dict(components,loss=loss,
                original_loss=original,distillation_loss=auxiliary).items()}

    def save(self,path,*,evidence):
        if evidence.get('contains_test_or_validation_data') is not False:
            raise ValueError('Student optimizer data must be training-only')
        torch.save(dict(algorithm=self.algorithm,variant=self.variant,config=asdict(self.config),
                        calibration=asdict(self.calibration),distillation_weight=self.distillation_weight,
                        updates=self.updates,state_dict=self.state_dict(),evidence=evidence),path)

    @classmethod
    def load(cls,source):
        data=torch.load(source,map_location='cpu',weights_only=True)
        if data.get('algorithm')!=cls.algorithm:
            raise ValueError('Distilled policy identity mismatch')
        model=cls(SafeGPINNConfig(**data['config']),variant=data['variant'],
                  calibration=TeacherCalibration(**data['calibration']),
                  distillation_weight=data['distillation_weight'])
        model.load_state_dict(data['state_dict']);model.updates=data['updates']
        return model,data
