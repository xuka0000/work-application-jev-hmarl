"""Summarize verified development controls without treating them as test results."""
from collections import Counter
from datetime import datetime,timezone
import json
import gzip
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/jev_online_20260928'


def command_attribution(path,decisions):
    """Separate model/rule selections from fallback and join actual command IDs."""
    messages={};counts={};reasons={}
    for label in ('accepted_selector','fallback','other'):
        counts[label]=Counter();reasons[label]=Counter()
    with gzip.open(path/'events.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            # DecisionTrace writes canonical sorted top-level keys. Skip large
            # physical-state payloads before JSON parsing, never by searching
            # for a nested event name inside the payload.
            if line.startswith('{"kind":') and not line.startswith(('{"kind":"message",','{"kind":"command_execution",')):
                continue
            e=json.loads(line);d=e['payload']
            is_dispatch=(d.get('kind') in ('deployment_command','relay_authorization','control_command','depot_service')
                         or (d.get('kind')=='authorization' and 'uav_mission' in d.get('payload',{})))
            if e['kind']=='message' and is_dispatch:
                mid=d['message_id']
                if d.get('event')=='send':
                    matches=[r for r in decisions if r['issued_at_s'] is not None
                             and math.isclose(r['issued_at_s'],d['sent_s'],abs_tol=1e-8,rel_tol=0)]
                    if len(matches)>1:raise ValueError('Ambiguous command ownership')
                    if not matches:continue
                    row=matches[0]
                    label='fallback' if row['fallback_used'] else ('accepted_selector' if row['response'].get('source') in {'local_compute','live_http','cache'} and row['response_accepted'] else 'other')
                    messages[mid]=label;counts[label]['sent']+=1
                elif mid in messages and d.get('event')=='drop':
                    label=messages[mid];counts[label]['dropped']+=1;reasons[label][d['outcome']]+=1
            elif e['kind']=='command_execution' and d.get('message_id') in messages:
                label=messages[d['message_id']]
                counts[label]['applied' if d.get('applied') else 'execution_rejected']+=1
    return {key:dict(**dict(value),drop_reasons=dict(reasons[key])) for key,value in counts.items()}


def main():
    batch=json.loads((BASE/'local_control_batch_lease2.json').read_text())
    report=dict(observed_utc=datetime.now(timezone.utc).isoformat(),batch_status=batch['status'],
        independent_cases=1,case_id='validation_01',profile='good',channel_seed=301,
        case_selection='First registered case with more than one initial candidate; all twelve initial intervals audited.',
        formal_test_evidence=False,statistical_superiority_claim=False,
        waiting_case_retained='validation_00: two full wait-only controls; not removed from registered final matrix',methods=[])
    for job in batch['jobs']:
        path=BASE/job['output'];status_path=path/'status.json'
        if not status_path.exists():
            report['methods'].append(dict(method=job['method'],status='not_started'));continue
        state=json.loads(status_path.read_text());row=dict(method=job['method'],status=state['status'],
            source_code_sha256=state['source_code_sha256'],evidence=str(status_path.relative_to(ROOT)))
        if state['status']!='completed':
            row['error']=state.get('error');report['methods'].append(row);continue
        if not (path/'verification.json').is_file():raise ValueError('Missing physical trace verification')
        decisions=json.loads((path/'online_decisions.json').read_text())
        calls=[r for r in decisions if r['response'].get('source')=='local_compute']
        elapsed=[r['policy_elapsed_s'] for r in calls]
        row.update(metrics=state['metrics'],wall_elapsed_s=state['elapsed_s'],ticks=len(decisions),
            selector_calls=len(calls),singleton_ticks=sum(r['response'].get('source')=='singleton' for r in decisions),
            nonwait_issued_ticks=sum(any(a['kind']!='wait' for a in r.get('issued_joint') or []) for r in decisions),
            accepted_nonwait_selector_choices=sum(r['response_accepted'] and any(a['kind']!='wait' for a in r.get('issued_joint') or []) for r in calls),
            fallback_nonwait_ticks=sum(r['fallback_used'] and any(a['kind']!='wait' for a in r.get('issued_joint') or []) for r in decisions),
            issued_atomic_actions=dict(Counter(a['kind'] for r in decisions for a in r.get('issued_joint') or [])),
            fallbacks=sum(r['fallback_used'] for r in decisions),
            rejections=dict(Counter(r['rejection_reason'] for r in decisions if r['rejection_reason'])),
            selector_latency_s=dict(mean=statistics.mean(elapsed),maximum=max(elapsed)) if elapsed else None,
            selection_audit_evidence=str((path/'online_decisions.json').relative_to(ROOT)),
            command_attribution=command_attribution(path,decisions))
        report['methods'].append(row)
    complete=[r for r in report['methods'] if r['status']=='completed']
    baseline=next((r for r in complete if r['method']=='critical_load_greedy'),None)
    if baseline:
        for row in complete:
            row['critical_ens_difference_from_greedy_kwh']=row['metrics']['critical_ens_kwh']-baseline['metrics']['critical_ens_kwh']
    report['limitations']=['One development case and one field-channel seed; CQL has one completed training seed.',
        'Measured computing waits can change physical trajectories; before/after latency is not an isolated causal intervention.',
        'A completed training job or feasible rollout alone does not establish a competitive public-baseline reproduction.',
        'No Jev or LLM selector was executed in this control batch.',
        'Applied command attribution means command execution admission or mission start, not completed work or restored electricity. Unsent actions are not silently counted as successful commands.']
    (BASE/'local_control_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([dict(method=r['method'],status=r['status'],metrics=r.get('metrics'),
        selector_calls=r.get('selector_calls'),nonwait_issued_ticks=r.get('nonwait_issued_ticks')) for r in report['methods']]))


if __name__=='__main__':main()
