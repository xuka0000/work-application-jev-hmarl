"""Materialize the authorized publication matrix; never launch a job.

This registration replaces only the unexecuted online-teacher test design.
Existing baseline jobs, counts, results, and scientific source files are immutable.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jev_hmarl_publication_20261002'
BASE = ROOT / 'outputs/jev_online_20260928/baseline_campaign_01'
SEEDS = [13, 29, 47, 61, 83]
PROFILES = ['good', 'iid', 'burst']
CHANNELS = [301, 307, 311]
MATURE = ['causal_rolling_milp', 'critical_load_greedy', 'gain_per_execution_time']
OLD_LEARNED = ['iql', 'cql', 'hmarl']
NEW_LEARNED = ['jev_calibrated_hmarl', 'jev_raw_hmarl',
               'llm_calibrated_hmarl', 'planner_calibrated_hmarl']
SENSITIVITY = ['jev_calibrated_lambda003', 'jev_calibrated_lambda030']
from jev_publication.execution_policy import deadline
DEADLINE = deadline()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def build_matrix():
    rows = []
    for stage, cases, channels, methods in (
        ('development', [f'validation_{i:02d}' for i in range(12)], [301], NEW_LEARNED + SENSITIVITY),
        ('formal', [f'jev_test_{i:02d}' for i in range(24)], CHANNELS, MATURE + OLD_LEARNED + NEW_LEARNED)):
        for case in cases:
            for profile in PROFILES:
                for channel in channels:
                    for method in methods:
                        for seed in ([None] if method in MATURE else SEEDS):
                            key = f'{stage}_{case}_{profile}_c{channel}_{method}' + ('' if seed is None else f'_s{seed}')
                            rows.append(dict(task_id=key, kind='physical_evaluation', stage=stage,
                                case_id=case, profile=profile, channel_seed=channel, method=method,
                                training_seed=seed, horizon_s=14400, wall_limit_s=1200,
                                dependencies=(['baseline_complete', 'teacher_evidence', 'development_review', 'formal_freeze']
                                    if stage == 'formal' else ['baseline_complete', 'teacher_evidence']),
                                output=f'outputs/jev_hmarl_publication_20261002/trials/{key}',
                                execution_state='not_submitted'))
    assert len(rows) == 9288 and sum(x['stage'] == 'formal' for x in rows) == 8208
    assert len({x['task_id'] for x in rows}) == len(rows)
    return rows


def main():
    reg = OUT / 'registration'
    reg.mkdir(parents=True, exist_ok=True)
    protocol_path = reg / 'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Registration exists; use an explicit reviewed amendment, not overwrite')
    old = read(BASE / 'protocol.json')
    receipts = [read(p) for p in (BASE / 'completed').glob('eval_*.json')]
    rows = build_matrix()
    matrix_path = reg / 'evaluation_matrix.jsonl'
    with matrix_path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    old_jobs = {j['task_id']: j for j in old['jobs'] if j['kind'] == 'trial'}
    valid = {}
    for receipt in receipts:
        job = old_jobs.get(receipt['task_id'])
        if job is None or receipt['job_sha256'] != hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest():
            raise ValueError('Baseline receipt job identity mismatch')
        if receipt['facts']['metrics'].get('complete') is not True:
            raise ValueError('Incomplete result cannot count as completed')
        valid[receipt['task_id']] = receipt
    remaining = [j for j in old_jobs.values() if j['task_id'] not in valid]
    write(reg / 'baseline_remaining.json', dict(protocol_sha256=digest(BASE / 'protocol.json'),
        completed=len(valid), total=648, remaining=remaining,
        receipt_check='Job identity and complete flag; full raw-file rehash required before reuse admission',
        retry_requires_diagnosis=['eval_validation_07_good_cql_s29']))
    means = [r['facts']['elapsed_s'] for r in receipts]
    trace_sizes = []
    for r in receipts:
        for path in r['files']:
            if path.endswith('events.jsonl.gz'):
                trace_sizes.append((ROOT / path).stat().st_size)
    mean_s = statistics.mean(means)
    mean_bytes = statistics.mean(trace_sizes)
    known_evals = len(rows) + len(remaining)
    # The calibration cost is explicitly unknown until registered real branches run.
    schedule = dict(deadline_utc=DEADLINE, timezone='Asia/Hong_Kong',
        baseline_remaining=len(remaining), new_evaluations=len(rows),
        empirical_trial_seconds=mean_s, empirical_trace_bytes=mean_bytes,
        timing_sample=len(means), trace_sample=len(trace_sizes),
        numerical_eval_cpu_hours=known_evals*mean_s/3600,
        evaluation_hours_at_16_workers_with_25percent_margin=known_evals*mean_s/16/3600*1.25,
        projected_raw_trace_gib=known_evals*mean_bytes/2**30,
        training_jobs=30, training_max_cpu_or_gpu_hours=30*1200/3600,
        calibration_physical_branches_maximum=3840,
        calibration_duration_estimate=None, analysis_reserve_hours=8,
        independent_persistent_destination=None,
        deadline_feasible=None,
        reason='Conditional estimate only. Calibrations, matched server throughput, storage and evidence review are unadmitted.',
        checkpoints=[
            dict(by_hkt='2026-10-03 08:00', deliverable='Baseline remainder and teacher/calibration queues admitted; deployed job/observer evidence'),
            dict(by_hkt='2026-10-04 12:00', deliverable='Baseline, labels, calibration and 30 training checkpoints with completed development matrix'),
            dict(by_hkt='2026-10-04 16:00', deliverable='Independent development review and source/input/model freeze before any test use'),
            dict(by_hkt='2026-10-06 15:00', deliverable='All formal trials complete, coverage and physical invariants verified'),
            dict(by_hkt='2026-10-06 23:59', deliverable='Paired statistics, figures, tables and bounded manuscript conclusions')])
    write(reg / 'capacity_and_schedule.json', schedule)
    protocol = dict(schema_version=1, campaign_id='jev_hmarl_publication_20261002',
        registered_utc=datetime.now(timezone.utc).isoformat(), status='registered_pending_admission',
        authorized_scope='User requested complete publication-supporting experiments by before October 7; Jev is inside heterogeneous MARL training',
        supersedes_unexecuted='Online primary method and unexecuted 4320-row test design only; no existing numerical results overwritten',
        amendment_reason='Four teacher arms distinguish teacher usefulness, calibration, language-model substitution and cheap planner explanations; the user changed the method to Jev-assisted MARL',
        previous_protocol='outputs/jev_online_20260928/registration/protocol.json',
        previous_protocol_sha256=digest(ROOT/'outputs/jev_online_20260928/registration/protocol.json'),
        deadline_utc=DEADLINE,
        claims=[
            dict(id='C1', question='Does evidence-calibrated Jev supervision improve restoration or seed stability over identical reward-only HMARL?',
                contrasts=['jev_calibrated_hmarl vs hmarl'], primary='critical_ens_kwh',
                falsifier='No meaningful paired gain, worsened feasibility or unacceptably unstable seeds; report as unsupported or bounded'),
            dict(id='C2', question='Does physical-outcome calibration improve on raw Jev preferences?',
                contrasts=['jev_calibrated_hmarl vs jev_raw_hmarl'],
                evidence=['critical ENS', 'admitted teacher gradients', 'case-held-out expected regret', 'unsupported-bin coverage'],
                falsifier='Raw supervision equals or improves the calibrated arm at lower cost; do not claim calibration necessary'),
            dict(id='C3', question='Does Jev provide useful teaching signal beyond a matched LLM and causal planner?',
                contrasts=['jev_calibrated_hmarl vs llm_calibrated_hmarl', 'jev_calibrated_hmarl vs planner_calibrated_hmarl'],
                evidence=['critical ENS', 'teacher excess ENS', 'coverage and abstention', 'actual labeling time and cost'],
                falsifier='LLM or planner explains the improvement; narrow Jev-specific novelty'),
            dict(id='C4', question='Under which registered communication conditions does the mechanism help or fail?',
                evidence=['good/iid/burst stratified paired effects', 'critical-load recovery trajectories', 'stale reports and rejected commands', 'physical energy/route constraints'],
                falsifier='Benefit disappears or reverses under weak communication; retain these cells and bound conclusions')],
        innovation_boundary='A teacher name replacement and a standard KL term are not themselves new contributions; evidence must establish the scene-specific calibrated learning mechanism and its limits',
        algorithm=dict(backbone='Existing centralized received-observation heterogeneous device-factor actor and joint mixer',
            transfer='Teacher preferences change local model training; no teacher call during deployment',
            no_claims=['Jev weight fine-tuning', 'decentralized execution', 'guaranteed superiority', 'online Jev speed versus distilled LLM policy']),
        data=dict(training_cases=60, development_cases=12, calibration_cases=12, formal_cases=24,
            training_transitions=8640, training_cap=20000, training_seeds=SEEDS,
            profiles=PROFILES, development_channel_seeds=[301], formal_channel_seeds=CHANNELS,
            split_manifest='outputs/jev_study_20260928/input_manifest.json',
            test_access='No test rollout, scoring, parameter selection or result-dependent pruning before evidence-backed freeze'),
        teachers=dict(training_requests=305, providers=['jev', 'llm'], local_teacher='causal_rolling_milp',
            same_received_information=True, same_candidate_subset=True,
            candidate_limit=32, model_ids=dict(jev='jev-1.13.0', llm='qwen/qwen3.8-max-0902'),
            per_provider_new_training_call_cap=305, per_provider_calibration_call_cap=120,
            total_provider_cap_including_history=12240, automatic_paid_retry=False,
            model_failure='Retain request attempt; stop that provider and diagnose. Independent admissible jobs may continue.',
            abstention='Retain all abstentions. Zero gradient for missing/abstained/unsupported labels. No fabricated replacement labels.',
            planner_provenance='Native planner record with received-observation and source hashes; never forge an HTTP/provider receipt'),
        calibration=dict(case_bundles=12, cases=[f'validation_jev_{i:02d}' for i in range(12)],
            profile='good', channel_seed=301, selection='First ten non-singleton states per case under frozen greedy continuation; if fewer, report actual coverage without substitute cases',
            state_cap_per_case=10, state_cap_total=120, branch_cap_per_state=32, physical_branch_cap=3840,
            branch_wall_limit_s=1200, horizon_s=14400,
            candidate_outcomes='All candidates on the identical reproduced received prefix, then full greedy physical continuation',
            execution='Independent branch jobs with per-branch timeout; aggregate a state only after every registered candidate is verified',
            minimum_fault_cases_per_risk_bin=3, unsupported_bin_weight=0,
            calibration_performance='Leave-one-fault-case-out diagnostic; final map fit only to these twelve calibration cases; report subgroup transport limits'),
        methods=dict(mature=MATURE, existing_learned=OLD_LEARNED, primary_new=NEW_LEARNED,
            sensitivity=SENSITIVITY, primary_lambda=.1, sensitivity_lambdas=[.03,.3],
            train_updates=20000, train_wall_limit_s=1200, checkpoint='Last fixed-budget checkpoint, no best-test selection',
            sensitivity_policy='Development-only robustness evidence; primary lambda remains .1 regardless of test outcomes'),
        counts=dict(inherited_baseline_training=15, inherited_baseline_evaluations=648,
            new_training=30, new_development_evaluations=1080, formal_evaluations=8208,
            calibration_case_bundles=12, teacher_training_requests_per_provider=305,
            formal_previous_unexecuted=4320, formal_difference_reason='Seven learned arms with five seeds and three mature methods replace two online arms, three learned and three mature methods'),
        statistics=dict(independent_unit='fault case', primary='critical_ens_kwh', secondary='system_ens_kwh',
            repeats='Average channel and training seed repeats within a case; preserve seed variability separately',
            paired=True, bootstrap_repetitions=10000, confidence=.95,
            primary_contrasts=4, multiplicity='Holm across four predeclared primary contrasts',
            uncertainty='Case-cluster bootstrap CI and paired case randomization test; seed distributions separate',
            subgroup='Three predefined profiles; missing cells and failures remain visible, never dropped as unfavorable',
            selection='Illustrative process case fixed by case ID before test; any outcome-selected extrema explicitly marked'),
        fairness=dict(hardware='All compared methods for each case/profile/channel block run on the same pinned hardware class, affinity and thread settings',
            measured_latency='Selection time advances physical restoration identically for every method; simulator/trace serialization time excluded',
            reused_development='Preserve old 429 values and hardware strata; do not attribute cross-host timing differences to algorithm',
            formal_training='Reuse existing 15 valid baseline checkpoints; no needless retraining'),
        execution=dict(workers_maximum=16, per_trial_wall_s=1200, outer_timeout_s=1230,
            heartbeat_s=15, observation_s=300, heartbeat_stale_s=90,
            unknown_failure='No automatic retry; preserve and diagnose, one reviewed bounded retry without budget reset',
            physical_timeout='Preserve partial outcome and mark incomplete, never count as complete',
            scientific_negative='Complete registered comparisons; no endless score chasing',
            provider_calls_during_formal=0,
            unattended_supervision=False, notification_verified=False,
            release='Verified effective results saved independently, shutdown, release, official slot-return check'),
        delivery=['Full fixed-denominator coverage and failure table', 'Raw physical traces, configs, weights and exact dependency lock',
            'Paired main performance table with case CIs and four primary contrasts',
            'Teacher coverage/calibration/physical-regret table', 'Training and label time/cost table',
            'Communication-stratified recovery trajectories and seed variability',
            'Mechanism process figure connecting preference, learned action, physical state and restored service',
            'Calibration and lambda ablations including negative results',
            'Claims-to-evidence ledger and reproducible statistical scripts'],
        matrix=dict(path=str(matrix_path.relative_to(ROOT)), sha256=digest(matrix_path)),
        admission_gates=dict(baseline_complete=False, teacher_evidence=False, complete_runner=False,
            independent_design_review=False, matched_hardware=False, persistent_storage=False,
            deployed_observer=False, deadline_feasibility=False, formal_freeze=False),
        rule_versions=['R09 v1.3', 'R10 v1.0', 'R14 v1.0', 'STORAGE-EFFICIENCY-01 v1.0', 'CLOUD-AUTODL-RELEASE-01 v2.0'])
    write(protocol_path, protocol)
    print(json.dumps(dict(protocol=str(protocol_path.relative_to(ROOT)), sha256=digest(protocol_path),
        new_evaluations=len(rows), baseline_remaining=len(remaining),
        estimated_trace_gib=schedule['projected_raw_trace_gib'], admitted=False)))


if __name__ == '__main__':
    main()
