"""Load teacher data only after binding requests to replay and physical costs."""
import hashlib
import json
from pathlib import Path
from .physical_replay import file_sha256
from .decision_teacher import value_hash,parse_answer
from .jev_iql import fit_outcome_calibration


def load_evidence(replay,requests_path,annotations_path,calibration_path,*,workspace,model):
    root=Path(workspace).resolve()
    def verified_artifact(path,digest):
        path=(root/path).resolve()
        if not path.is_relative_to(root) or file_sha256(path)!=digest:
            raise ValueError('Physical calibration artifact escaped workspace or changed')
        return json.loads(path.read_text(encoding='utf-8'))
    if replay.metadata.get('reward_metric')!='critical_ens_kwh':
        raise ValueError('Teacher IQL requires the common critical-ENS objective')
    requests=[json.loads(line) for line in Path(requests_path).read_text(encoding='utf-8').splitlines()]
    annotations=json.loads(Path(annotations_path).read_text(encoding='utf-8'))
    if (annotations['model']!=model or annotations['requests_sha256']!=file_sha256(requests_path)
            or annotations['source_manifest_sha256']!=file_sha256(replay.path)):
        raise ValueError('Teacher annotations belong to a different model or corpus')
    request_map={r['transition_index']:r for r in requests}
    if len(request_map)!=len(requests):raise ValueError('Duplicate teacher request indices')
    labels={}
    for entry in annotations['rows']:
        index=entry['transition_index'];req=request_map[index];annotation=entry['annotation']
        teacher_record=verified_artifact(entry['teacher_record_path'],entry['teacher_record_sha256'])
        if (teacher_record.get('status')!='completed' or teacher_record.get('request')!=req['request']
                or teacher_record.get('model')!=model
                or parse_answer(teacher_record['response'],req['request'],expected_model=model)!=annotation):
            raise ValueError('Training annotation differs from its completed provider record')
        shard_id,step=replay.index[index];shard=replay.metadata['shards'][shard_id];row=replay.row(index)
        encoded=hashlib.sha256(row['state'].tobytes()+row['candidates'].tobytes()).hexdigest()
        if (index in labels or entry['split']!='training' or req['split']!='training'
                or entry['case_id']!=req['case_id'] or req['case_id']!=shard['case_id']
                or not req['case_id'].startswith('training_') or step!=req['source_decision_step']
                or encoded!=req['encoded_transition_sha256']
                or req['source_events_sha256']!=shard['source_events_sha256']):
            raise ValueError('Teacher sample is not bound to the exact training transition')
        envelope=req['request']
        if (envelope['request_sha256']!=value_hash(envelope['body'])
                or envelope['body']['state'].get('objective_metric')!='critical_ens_kwh'
                or annotation['model']!=model or annotation['request_sha256']!=envelope['request_sha256']
                or annotation['candidate_ids']!=envelope['candidate_ids']
                or annotation['candidate_indices']!=envelope['candidate_indices']
                or envelope['full_candidate_count']!=len(row['candidates'])):
            raise ValueError('Teacher answer/options differ from the registered request')
        labels[index]=dict(annotation,split='training')
    if not labels:raise ValueError('No verified teacher annotations')
    registry=json.loads(Path(calibration_path).read_text(encoding='utf-8'));physical_rows=[]
    if registry['model']!=model:raise ValueError('Calibration teacher version differs')
    seen=set()
    for entry in registry['entries']:
        outcome=verified_artifact(entry['outcomes_path'],entry['outcomes_sha256'])
        answer_record=verified_artifact(entry['teacher_record_path'],entry['teacher_record_sha256'])
        if (outcome.get('status')!='completed_physical_candidate_evaluation' or not outcome.get('formal_eligible')
                or outcome.get('horizon_s')!=14400. or outcome.get('label_source')!='physical_evaluator'
                or outcome['case_id'] not in {f'validation_jev_{i:02d}' for i in range(12)} or not outcome.get('evaluator_sha256')
                or answer_record.get('status')!='completed'):
            raise ValueError('Complete independent physical calibration outcomes required')
        request=answer_record['request']
        if outcome.get('request_sha256')!=request['request_sha256']:
            raise ValueError('Physical calibration did not reproduce this teacher request')
        if request['request_sha256'] in seen:raise ValueError('Repeated calibration state')
        seen.add(request['request_sha256'])
        answer=parse_answer(answer_record['response'],request,expected_model=model)
        if entry['observation_sha256']!=outcome['observation_sha256']:
            raise ValueError('Calibration observation identity differs')
        if [r['candidate_id'] for r in outcome['candidate_outcomes']]!=answer['candidate_ids']:
            raise ValueError('Physical outcomes use different teacher candidates or ordering')
        if sum(answer['probabilities'])<=0:continue
        physical_rows.append(dict(case_id=outcome['case_id'],probabilities=answer['probabilities'],
            critical_ens_kwh=[r['remaining_critical_ens_kwh'] for r in outcome['candidate_outcomes']]))
    calibration=fit_outcome_calibration(physical_rows,model=model,evidence_sha256=file_sha256(calibration_path))
    return labels,calibration,dict(annotations_sha256=file_sha256(annotations_path),
        requests_sha256=file_sha256(requests_path),calibration_sha256=file_sha256(calibration_path),
        model=model,annotated_training_transitions=len(labels),calibration_states=len(physical_rows))
