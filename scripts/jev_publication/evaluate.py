"""One registered local-policy restoration trial, including formal admission.

All method arms use the same physical controller, report boundary, resource
constraints and measured selection-time accounting. No teacher API is called.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform as host_platform
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_jev_publication_campaign import read,digest,DEADLINE,MATURE,OLD_LEARNED


def source_hashes():
    paths=[*sorted((ROOT/'src/ptin_sim/closed_loop').rglob('*.py')),
           ROOT/'src/ptin_sim/jev_hmarl_extension.py',Path(__file__)]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def load_job(plan_path,task_id):
    plan=read(plan_path);matrix=ROOT/plan['matrix']['path'].replace('\\','/')
    if digest(matrix)!=plan['matrix']['sha256']:raise ValueError('Registered matrix changed')
    matches=[r for line in matrix.read_text(encoding='utf-8').splitlines()
             if (r:=json.loads(line))['task_id']==task_id]
    if len(matches)!=1:raise ValueError('Unknown or ambiguous registered task')
    job=matches[0]
    if job['horizon_s']!=14400 or job['wall_limit_s']!=1200:
        raise ValueError('Registered physical horizon or wall budget differs')
    return plan,job


def verify_binding(record):
    path=(ROOT/record['path'].replace('\\','/')).resolve()
    if not path.is_relative_to(ROOT.resolve()) or digest(path)!=record['sha256']:
        raise ValueError('Evidence artifact changed or escapes the workspace')
    return path


def admission(plan_path,job,models_path,freeze_path,source):
    if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):
        raise TimeoutError('Publication deadline reached')
    registry=read(models_path)
    if registry.get('baseline_complete') is not True:
        raise ValueError('Completed baseline gate required')
    baseline=read(verify_binding(registry['baseline_completion_receipt']))
    if baseline.get('protocol_sha256')!=digest(ROOT/'outputs/jev_online_20260928/baseline_campaign_01/protocol.json'):
        raise ValueError('Baseline completion identity mismatch')
    key=job['method']+('' if job['training_seed'] is None else f'_s{job["training_seed"]}')
    model=registry['models'].get(key)
    if job['method'] not in MATURE:
        if model is None:raise ValueError('Registered trained model is unavailable')
        verify_binding(model['checkpoint'])
        if model.get('training_seed')!=job['training_seed'] or model.get('updates')!=20000:
            raise ValueError('Model seed/update budget differs from comparison')
    if job['stage']=='formal':
        if freeze_path is None:raise ValueError('Formal evaluation requires an evidence-backed freeze')
        freeze=read(freeze_path)
        if (freeze.get('status')!='frozen_for_formal_evaluation'
            or freeze.get('protocol_sha256')!=digest(plan_path)
            or freeze.get('models_sha256')!=digest(models_path)
            or freeze.get('source_code_sha256')!=source):
            raise ValueError('Formal protocol/model/source freeze mismatch')
        required={'baseline','teacher_evidence','calibration','development','independent_review',
                  'storage','matched_hardware','observer','deadline'}
        if set(freeze.get('gates',{}))!=required:
            raise ValueError('Incomplete formal admission evidence')
        for name,binding in freeze['gates'].items():
            evidence=read(verify_binding(binding))
            if evidence.get('status')!='pass':raise ValueError('Formal gate failed: '+name)
        for binding in freeze['inputs']:
            verify_binding(binding)
    return model


def execute(plan_path,task_id,models_path,freeze_path):
    plan,job=load_job(plan_path,task_id);source=source_hashes()
    binding=admission(plan_path,job,models_path,freeze_path,source)
    from ptin_sim.closed_loop.communication_study.physical_factory import build_physical_case
    from ptin_sim.closed_loop.communication_study.physical_rule_baselines import PhysicalRuleBaseline
    from ptin_sim.closed_loop.communication_study.packets import ChannelConfig
    from ptin_sim.closed_loop.communication_study.decision_trace import DecisionTrace,verify_trace
    from ptin_sim.closed_loop.communication_study.online_restoration import TimedPhysicalController
    import torch
    torch.set_num_threads(1)
    output=(ROOT/job['output']).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Output escapes workspace')
    platform=read(ROOT/'outputs/complete_communication_study_20260918/physical_learning_expiryfix_20260919/freeze.json')
    resources=dict(platform['resources'],reward_metric='critical_ens_kwh')
    identity=dict(job=job,protocol_sha256=digest(plan_path),models_sha256=digest(models_path),
        source_code_sha256=source,formal_experiment=job['stage']=='formal',teacher_api_calls=0,
        host=dict(system=host_platform.platform(),machine=host_platform.machine(),processor=host_platform.processor(),
                  python=sys.version,torch=str(torch.__version__),threads=1),
        pid=os.getpid())
    start=time.monotonic();state=dict(identity,status='initializing');env=None;controller=None
    def publish(**fields):
        state.update(fields,observed_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-start)
        tmp=output/'status.tmp';tmp.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(output/'status.json')
    with DecisionTrace(output,identity=identity,parameters=resources,compression='gzip') as trace:
        try:
            publish()
            channel=ChannelConfig(profile=job['profile'],seed=job['channel_seed'],horizon_s=14400.,
                loss_rate=.01 if job['profile']=='good' else .2,delay_ms=35.,jitter_ms=platform['jitter_ms'],burst_slots=10)
            env,session=build_physical_case(ROOT,job['case_id'],resources=resources,channel=channel,trace=trace,
                input_manifest=ROOT/'outputs/jev_study_20260928/input_manifest.json')
            fallback=PhysicalRuleBaseline(env.scenario,env.resource_configuration,method='critical_load_greedy')
            method=job['method'];policy=None;model=None;evidence=None
            if method=='causal_rolling_milp':
                from ptin_sim.closed_loop.communication_study.main_case import load_main_case
                from ptin_sim.closed_loop.communication_study.rolling_restoration_milp import CausalRollingMILP
                _,_,electrical,_=load_main_case(ROOT,job['case_id'],input_manifest=ROOT/'outputs/jev_study_20260928/input_manifest.json')
                policy=CausalRollingMILP(env.scenario,env.resource_configuration,electrical,periods=4,wall_limit_s=20.,horizon_s=14400.)
            elif method in MATURE:
                policy=PhysicalRuleBaseline(env.scenario,env.resource_configuration,method=method)
            else:
                from ptin_sim.closed_loop.communication_study.jev_encoding import TargetAwarePhysicalEncoder
                from ptin_sim.closed_loop.publication_iql import CandidateIQL
                from ptin_sim.closed_loop.communication_study.categorical_cql import CandidateCQL
                from ptin_sim.closed_loop.communication_study.physical_research_policies import CandidateHMARL
                from ptin_sim.jev_hmarl_extension import JevHMARL
                checkpoint=verify_binding(binding['checkpoint'])
                cls={'iql':CandidateIQL,'cql':CandidateCQL,'hmarl':CandidateHMARL}.get(method,JevHMARL)
                model,evidence=cls.load(checkpoint)
                if method=='hmarl':evidence=evidence['evidence']
                encoder=TargetAwarePhysicalEncoder(env.scenario,env.resource_configuration,horizon_s=14400.)
                if (evidence.get('training_seed')!=job['training_seed'] or model.updates!=20000
                    or evidence.get('encoder_sha256')!=encoder.sha256
                    or evidence.get('contains_test_or_validation_data') is not False
                    or evidence.get('reward_metric')!='critical_ens_kwh'
                    or evidence.get('dataset_sha256')!=digest(ROOT/'outputs/jev_study_20260928/target_identity_replay/manifest.json')):
                    raise ValueError('Trained model provenance differs from registered comparison')
                if method not in OLD_LEARNED:
                    expected_weight=.03 if method.endswith('lambda003') else .3 if method.endswith('lambda030') else .1
                    expected_calibration=method!='jev_raw_hmarl'
                    if model.teacher_weight!=expected_weight or model.use_outcome_calibration!=expected_calibration:
                        raise ValueError('Teacher intervention differs from assigned arm')
                    expected_model=('qwen/qwen3.8-max-0902' if method.startswith('llm') else
                                    'local/causal-rolling-milp-v1' if method.startswith('planner') else 'jev-1.13.0')
                    if model.calibration.model!=expected_model:raise ValueError('Teacher identity differs from assigned arm')
                model.eval().requires_grad_(False)
                trace.emit('learned_policy_loaded',env.time_s,dict(method=method,evidence=evidence))
            def select(observation,joints,*,timeout_s):
                if policy is not None:
                    joint=policy.select(observation,joints);audit=getattr(policy,'last_decision_trace',{})
                else:
                    with torch.inference_mode():
                        s,c=encoder.encode(observation,joints)
                        fn='candidate_logits' if method=='iql' else 'values' if method=='cql' else 'scores'
                        scores=getattr(model,fn)(s[None],c[None])[0]
                        if not torch.isfinite(scores).all():raise ValueError('Nonfinite learned selection')
                        index=int(scores.argmax());joint=joints[index]
                        audit=dict(method=method,selected_index=index,selected_score=float(scores[index]),
                                   candidate_count=len(joints),training_seed=job['training_seed'])
                return dict(joint=joint,selection_audit=audit,source='local_compute')
            controller=TimedPhysicalController(session,select,fallback.select,
                proposal_seed=platform['joint_proposal_seed'],proposal_draws=platform['joint_proposal_draws'])
            while env.time_s<env.horizon_s:
                if time.monotonic()-start>=1200 or datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):
                    raise TimeoutError('Registered trial or publication deadline reached')
                controller.step();publish(status='running',physical_time_s=env.time_s,decisions=len(controller.records))
            ground=max(map(abs,env.ledger.conservation_residuals_kwh().values()),default=0.)
            uav=max(map(abs,env.uav_execution.conservation_residuals_km().values()),default=0.)
            if max(ground,uav)>1e-7 or not env.service_metrics()['complete']:raise ValueError('Incomplete or infeasible physical trajectory')
            if any(digest(ROOT/p)!=sha for p,sha in source.items()):raise ValueError('Evaluation source changed')
            (output/'online_decisions.json').write_text(json.dumps(controller.records,allow_nan=False),encoding='utf-8')
            publish(status='verifying',metrics=env.service_metrics(),ground_energy_residual_kwh=ground,
                    uav_energy_residual_km=uav,fallbacks=sum(r['fallback_used'] for r in controller.records))
        except BaseException as exc:
            publish(status='failed',error_type=type(exc).__name__,physical_time_s=env.time_s if env else None)
            raise
    verification=verify_trace(output/'events.jsonl.gz')
    (output/'verification.json').write_text(json.dumps(verification),encoding='utf-8')
    publish(status='completed')
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--job-id',required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute(a.protocol,a.job_id,a.models,a.freeze)
    else:
        plan,job=load_job(a.protocol,a.job_id);print(json.dumps(dict(status='registered_not_executed',job=job)))
