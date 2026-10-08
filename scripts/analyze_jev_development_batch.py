"""Evidence summaries for complete and failed development trials, preserving both."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from analyze_jev_local_controls import command_attribution

ROOT=Path(__file__).resolve().parents[1]
ANALYZER_VERSION='paired-development-v2-named-anchors'


def named_anchors(request):
    """Recover the two anchored rules from the exact public action summaries.

    Both rule maximizers and wait are mandatory shortlist members. Their
    original indices preserve the full-catalogue deterministic tie break.
    This labels historical choices; it never changes a policy or outcome.
    """
    options=request['body']['questions']['action']['criteria']
    results={}
    for method in ('critical_load_greedy','gain_per_execution_time'):
        rows=[]
        for cid,index in zip(request['candidate_ids'],request['candidate_indices']):
            actions=options[cid];gains={}
            for action in actions:
                duration=action['estimated_resource_occupation_s']
                for key,value in action['weighted_local_load_proxies_kw'].items():
                    score=value/duration if method=='gain_per_execution_time' else value
                    gains[key]=max(gains[key],score) if key in gains else score
            rows.append((sum(gains.values()),sum(a['kind']=='uav_replace' for a in actions),-len(actions),-index,cid,index))
        best=max(rows,key=lambda row:row[:4])
        if best[5] not in request['anchor_indices']:
            raise ValueError('Reconstructed rule is not a registered anchor')
        results[method]=dict(choice=best[4],index=best[5])
    return results


def analyze(job):
    output=ROOT/job['output'];source=output/'status.json'
    if not source.exists():return dict(job,status='not_started')
    state=json.loads(source.read_text(encoding='utf-8'))
    identity=hashlib.sha256(source.read_bytes()).hexdigest()
    cached=output/'decision_outcome_analysis.json'
    if cached.exists():
        old=json.loads(cached.read_text(encoding='utf-8'))
        if old['status_sha256']==identity and old.get('analyzer_version')==ANALYZER_VERSION:return old
    result=dict(job,status=state['status'],status_sha256=identity,analyzer_version=ANALYZER_VERSION,
                physical_time_s=state['physical_time_s'],evidence=str(source.relative_to(ROOT)),
                model_settings=state['trial'].get('llm_settings'),
                observed_utc=datetime.now(timezone.utc).isoformat())
    if state['status']!='completed':
        result['error']=state.get('error')
        return result
    if not (output/'verification.json').is_file() or not state['metrics']['complete']:
        raise ValueError('Completed trial lacks full physical evidence')
    records=json.loads((output/'online_decisions.json').read_text(encoding='utf-8'))
    calls=[r for r in records if r['response'].get('source') in ('local_compute','live_http','cache')]
    live=[r for r in calls if r['response'].get('source')=='live_http']
    valid=[r for r in live if r['response'].get('choice') is not None]
    nonabstain=[r for r in valid if r['response']['choice']!='abstain']
    latencies=[r['policy_elapsed_s'] for r in calls]
    requests={}
    for path in (output/'model_requests').glob('*_request.json'):
        request=json.loads(path.read_text(encoding='utf-8'))
        if 'summary' in request.get('information_loss','') or request.get('contract_version')=='abstention_factorial_diagnostic_v1':
            requests[request['request_sha256']]=named_anchors(request)
    result.update(metrics=state['metrics'],ticks=len(records),selector_calls=len(calls),
        valid_api_answers=len(valid),abstentions=sum(r['response']['choice']=='abstain' for r in valid),
        nonabstain=len(nonabstain),accepted_nonwait=sum(r['response_accepted'] and any(a['kind']!='wait' for a in r['issued_joint'] or []) for r in calls),
        fallbacks=sum(r['fallback_used'] for r in records),
        singleton_ticks=sum(r['response'].get('source')=='singleton' for r in records),
        anchor_matches=sum(r['response'].get('selected_index') in (r['response'].get('rule_anchor_indices') or []) for r in nonabstain),
        rejections=dict(Counter(r['rejection_reason'] for r in records if r['rejection_reason'])),
        policy_latency_s=dict(mean=statistics.mean(latencies),maximum=max(latencies)) if latencies else None,
        service_latency_s=[r['response']['service_elapsed_s'] for r in live],
        command_attribution=command_attribution(output,records),
        choices=[dict(time_s=r['grid_start_s'],choice=r['response'].get('choice'),selected_index=r['response'].get('selected_index'),
                      anchor_indices=r['response'].get('rule_anchor_indices'),accepted=r['response_accepted'],fallback=r['fallback_used']) for r in calls],
        boundary='Development data only. Applied command means work/mission admission, not completed restoration. Confidence is not empirical accuracy.')
    for choice,record in zip(result['choices'],calls):
        anchors=requests.get(record['response'].get('request_sha256'))
        if anchors:
            choice['named_anchors']=anchors
            choice['matched_rules']=[name for name,anchor in anchors.items() if anchor['choice']==choice['choice']]
    cached.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[]
    for path in args.protocol:
        protocol=json.loads(path.read_text(encoding='utf-8'))
        rows.extend(analyze(job) for job in protocol['jobs'])
    # A continued queue may mention an unexecuted output more than once; IDs stay explicit.
    report=dict(observed_utc=datetime.now(timezone.utc).isoformat(),trials=rows,
                independent_unit='fault case, not tick or API call',formal_test_evidence=False,
                scientific_superiority=False,failed_attempts_retained=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([{k:r.get(k) for k in ('case_id','method','status','metrics','valid_api_answers','abstentions','accepted_nonwait')} for r in rows]))


if __name__=='__main__':main()
