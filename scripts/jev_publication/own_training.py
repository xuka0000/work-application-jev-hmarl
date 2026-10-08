"""Train decision-teacher MARL from verified evidence, without API calls.

Default mode is a light admission check. --execute additionally requires the
complete public evidence and storage admission. Runtime checkpoints resume only
launches the frozen baseline, expands budgets, or creates cloud resources.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import signal
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'outputs/jev_online_20260928/baseline_campaign_01'
DEADLINE = datetime.fromisoformat('2026-10-06T18:08:15.976677+00:00')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


from own_admission import admission
from own_window import require_batch, stage_deadline
from execution_policy import wall_limit, remaining_wall
from training_runtime import save_runtime, restore_runtime


@contextmanager
def training_lock(output):
    """Reject concurrent new/resumed attempts without guessing stale ownership."""
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    path=output.parent/('.'+output.name+'.training.lock')
    token=uuid.uuid4().hex
    import psutil
    record=dict(pid=os.getpid(),birth=psutil.Process().create_time(),token=token,
                output=str(output),observed_at=time.time())
    try:
        handle=path.open('x',encoding='utf-8')
    except FileExistsError as exc:
        raise RuntimeError('Training ownership exists; duplicate start requires diagnosis') from exc
    try:
        with handle:json.dump(record,handle)
        yield record
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('token')==token:
            path.unlink()


def replay_cache_allocation(batch,output):
    """Bind cache to the queue's verified task RAM, never host free memory."""
    empty=dict(cache_bytes=0,reason='No verified per-process RAM allocation')
    names=('JEV_QUEUE_MANIFEST','JEV_QUEUE_MANIFEST_SHA256','JEV_TASK_ID','JEV_SERVER_ADMISSION_SHA256')
    if not all(os.environ.get(name) for name in names):return empty
    manifest=(ROOT/os.environ[names[0]]).resolve()
    if not manifest.is_relative_to(ROOT.resolve()) or digest(manifest)!=os.environ[names[1]]:
        raise ValueError('Replay resource queue identity differs')
    queue=json.loads(manifest.read_text(encoding='utf-8'))
    if queue.get('phase')!='own_training' or queue.get('run_class')!='scientific':
        raise ValueError('Replay resources belong to a different stage')
    jobs=[j for j in queue['jobs'] if j['task_id']==os.environ[names[2]]]
    if len(jobs)!=1 or (ROOT/jobs[0]['output']).resolve()!=Path(output).resolve():
        raise ValueError('Replay resources belong to a different training task')
    server=(ROOT/queue['admission']).resolve()
    if not server.is_relative_to(ROOT.resolve()) or digest(server)!=os.environ[names[3]]:
        raise ValueError('Replay server admission changed')
    admission=json.loads(server.read_text(encoding='utf-8'))
    allocation=admission.get('training_ram_allocation')
    if allocation is None:return empty
    if admission.get('status')!='pass' or admission.get('hostname')!=platform.node():
        raise ValueError('Replay RAM was not admitted on this server')
    keys=('total_ram_bytes','process_ram_bytes','maximum_training_processes')
    if allocation.get('status')!='verified' or any(type(allocation.get(k)) is not int or allocation[k]<=0 for k in keys):
        raise ValueError('Verified integer training RAM allocation required')
    total,process,count=(allocation[k] for k in keys)
    if (count!=queue['workers'] or count!=queue['maximum_gpu_training_jobs']
            or process*count>total):
        raise ValueError('Training RAM exceeds the original shared process allocation')
    config=(ROOT/allocation['resource_config_path']).resolve()
    if (not config.is_relative_to(ROOT.resolve()) or digest(config)!=allocation['resource_config_sha256']
            or batch.get('resource_config_sha256')!=allocation['resource_config_sha256']):
        raise ValueError('Training RAM is not bound to the verified batch resource configuration')
    resource=json.loads(config.read_text(encoding='utf-8'))
    ram=resource.get('resources',{}).get('ram_gib')
    if (type(ram) not in (int,float) or not math.isfinite(ram) or ram<=0
            or total>int(ram*2**30) or resource.get('run_class')!='scientific'
            or resource.get('task_id')!=batch.get('task_id') or resource.get('ready') is not True):
        raise ValueError('Replay RAM exceeds the ready original scientific resource configuration')
    limit=allocation.get('replay_cache_bytes',0)
    if type(limit) is not int or not 0<=limit<=process//5:
        raise ValueError('Explicit replay cache exceeds the verified process RAM ceiling')
    return dict(cache_bytes=limit,cache_ceiling_bytes=process//5,
                process_ram_bytes=process,total_ram_bytes=total,
                maximum_training_processes=count,queue_sha256=os.environ[names[1]],
                server_admission_sha256=os.environ[names[3]],
                resource_config_sha256=allocation['resource_config_sha256'],task_id=jobs[0]['task_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Exact pinned teacher model ID')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(13, 29, 47, 61, 83), required=True)
    parser.add_argument('--teacher-weight', type=float, choices=(0., .03, .1, .3), default=.1)
    parser.add_argument('--uncalibrated-ablation', action='store_true')
    parser.add_argument('--wall-limit-s', type=float, default=wall_limit(True))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-evidence', type=Path)
    a = parser.parse_args()
    if not math.isfinite(a.wall_limit_s) or not 0 < a.wall_limit_s <= wall_limit(True):
        parser.error('Authorized cumulative training wall limit exceeded')
    report = admission(a)
    print(json.dumps(report), flush=True)
    if report['blockers']:
        return 2
    if not a.execute:
        return 0
    current_batch=require_batch('own_training')
    with training_lock(a.output):
        return execute_training(a,current_batch)


def execute_training(a,current_batch):
    # Replay is loaded only after scientific/resource admission.
    import torch
    from ptin_sim.jev_hmarl_extension import JevHMARL, HMARLConfig
    from ptin_sim.closed_loop.communication_study.physical_replay import PhysicalReplay
    from ptin_sim.closed_loop.communication_study.jev_training_evidence import load_evidence
    started = time.monotonic()
    stop_requested=False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable')
    cache_allocation=replay_cache_allocation(current_batch,a.output)
    replay = PhysicalReplay(a.manifest, cache_bytes=cache_allocation['cache_bytes'])
    labels, calibration, provenance = load_evidence(
        replay, a.requests, a.annotations, a.calibration, workspace=ROOT, model=a.model)
    if a.teacher_weight > 0 and not any(
            not v.get('abstained') and v.get('abstain_probability', 0.) < 1.
            and (a.uncalibrated_ablation or calibration.weight(v['probabilities']) > 0)
            for v in labels.values()):
        raise ValueError('No physically supported non-abstained teacher supervision')
    meta = replay.metadata
    lengths = [s['transitions'] for s in meta['shards']]
    if len(set(lengths)) != 1:
        raise ValueError('Same complete-episode sampling as baseline required')
    units = len({f.split(':')[0] for f in meta['action_features'] if ':' in f})
    identity_dim = (meta['action_dim'] - 1) // units - 15
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = random.Random(a.seed)
    model = JevHMARL(HMARLConfig(meta['state_dim'], meta['action_dim'], units=units,
                               identity_dim=identity_dim), calibration=calibration,
                     teacher_weight=a.teacher_weight,
                     use_outcome_calibration=not a.uncalibrated_ablation)
    model.to(a.device)
    # No new files are added to the frozen closed_loop tree.
    source_paths = [Path(__file__), ROOT / 'src/ptin_sim/jev_hmarl_extension.py',
        Path(__file__).with_name('training_runtime.py'),Path(__file__).with_name('train.py')]
    source_paths += list((ROOT / 'src/ptin_sim/closed_loop').rglob('*.py'))
    evidence = dict(provenance, dataset_sha256=digest(a.manifest),
        encoder_sha256=meta['encoder_sha256'], contains_test_or_validation_data=False,
        reward_metric='critical_ens_kwh', reward_divisor_kwh=meta['reward_divisor_kwh'],
        training_seed=a.seed, formal_eligible=False, development_only=True,
        observation_schema='physical_target_identity_v2',
        compute=dict(device=a.device, torch_version=str(torch.__version__), tf32=False,
                     hardware=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'CPU'),
        selection_policy='final_fixed_budget_no_test_selection',
        replay_batch_layout='lossless_row_major_packed_candidates_v1',
        source_code_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths})
    task_identity=dict(seed=a.seed,teacher_weight=a.teacher_weight,uncalibrated=a.uncalibrated_ablation,
        model=a.model,dataset=digest(a.manifest),requests=digest(a.requests),annotations=digest(a.annotations),
        calibration=digest(a.calibration),encoder=meta['encoder_sha256'],
        scientific_source={str(p.relative_to(ROOT)):digest(p) for p in source_paths if p.name!='own_training.py'})
    prior_elapsed=0.;teacher_updates=0
    if a.resume_checkpoint:
        if not a.resume_evidence:raise ValueError('Diagnosed recovery evidence required')
        recovery=json.loads(a.resume_evidence.read_text(encoding='utf-8-sig'))
        if recovery.get('status')!='pass' or recovery.get('checkpoint_sha256')!=digest(a.resume_checkpoint):
            raise ValueError('Recovery evidence does not bind this checkpoint')
        model,old_evidence,resumed=restore_runtime(a.resume_checkpoint,rng,task_identity=task_identity,device=a.device)
        if old_evidence!=evidence:raise ValueError('Training provenance changed during recovery')
        prior_elapsed=max(resumed['cumulative_elapsed_s'],float(recovery['prior_cumulative_elapsed_s']))
        teacher_updates=resumed['teacher_updates']
        if not a.output.is_dir():raise ValueError('Resume output does not exist')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    attempt=1+len(list(a.output.glob('optimizer_steps_attempt_*.jsonl')))
    if not a.resume_checkpoint and attempt!=1:raise ValueError('Existing training attempt cannot be overwritten')
    status = dict(status='running', pid=os.getpid(), updates=0, maximum_updates=20000,
                  provider_calls=0, scientific_performance_evidence=False, evidence=evidence,
                  replay_cache_allocation=cache_allocation)

    def publish(**fields):
        status.update(fields, elapsed_s=prior_elapsed+time.monotonic() - started,
                      observed_utc=datetime.now(timezone.utc).isoformat())
        temp = a.output / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temp.replace(a.output / 'status.json')

    def check_time():
        if (stop_requested or time.monotonic()-started>=min(a.wall_limit_s-prior_elapsed,remaining_wall(True))
                or time.time()>=stage_deadline('own_training')
                or time.time()>=float(os.environ.get('JEV_BATCH_DEADLINE',stage_deadline('own_training')))):
            raise TimeoutError('Existing training/study bound reached')

    try:
        publish()
        returns = []
        for shard_id,shard in enumerate(meta['shards']):
            check_time()
            n = shard['transitions']
            rewards = replay.episode_rewards(shard_id)
            values = [0.] * n
            total = 0.
            for i in range(n - 1, -1, -1):
                total += rewards[i]
                values[i] = total
            returns.extend(values)
        returns = torch.tensor(returns, dtype=torch.float32)
        with (a.output / f'optimizer_steps_attempt_{attempt:03d}.jsonl').open('x', encoding='utf-8') as log:
            for step in range(model.updates+1, 20001):
                check_time()
                episodes = [rng.randrange(len(lengths)) for _ in range(4)]
                indices = [e * lengths[e] + rng.randrange(lengths[e])
                           for e in episodes for _ in range(16)]
                batch = replay.batch(indices, packed=True)
                batch = {k: v.to(a.device) for k, v in batch.items()}
                batch['teacher_annotations'] = [labels.get(i) for i in indices]
                losses = model.update(batch, returns[indices], available_only=True)
                teacher_updates += int(losses['teacher_admitted'] > 0)
                log.write(json.dumps(dict(step=step, indices=indices, losses=losses), allow_nan=False) + '\n')
                if step == 1 or step % 100 == 0:
                    log.flush()
                    save_runtime(model,rng,a.output/'runtime_checkpoint.pt',evidence=evidence,
                        task_identity=task_identity,elapsed_s=prior_elapsed+time.monotonic()-started,
                        teacher_updates=teacher_updates,device=a.device)
                    publish(updates=step, teacher_updates=teacher_updates, last_losses=losses,
                        checkpoint_step=step,runtime_checkpoint_sha256=digest(a.output/'runtime_checkpoint.pt'),attempt=attempt)
        if a.teacher_weight > 0 and teacher_updates == 0:
            raise ValueError('Training had no admitted teacher gradients')
        checkpoint = a.output / 'checkpoint_020000.pt'
        model.save(checkpoint, evidence=evidence)
        restored, _ = JevHMARL.load(checkpoint)
        restored.to(a.device)
        probe = {k: v.to(a.device) for k, v in replay.batch([0, 1]).items()}
        with torch.no_grad():
            if not torch.equal(model.scores(probe['states'], probe['candidates']),
                               restored.scores(probe['states'], probe['candidates'])):
                raise ValueError('Checkpoint reload changed scores')
        if any(digest(ROOT / p) != sha for p, sha in evidence['source_code_sha256'].items()):
            raise ValueError('Training source changed during execution')
        completion_integrity=replay.verify_integrity()
        publish(status='completed_training_not_performance_evaluated', updates=model.updates,
                checkpoint_sha256=digest(checkpoint), reload_exact=True,
                replay_completion_integrity=completion_integrity,replay_io_metrics=replay.metrics)
    except BaseException as exc:
        save_runtime(model,rng,a.output/'runtime_checkpoint.pt',evidence=evidence,task_identity=task_identity,
            elapsed_s=prior_elapsed+time.monotonic()-started,teacher_updates=teacher_updates,device=a.device)
        publish(status='failed', updates=model.updates,
                error=dict(type=type(exc).__name__, message=str(exc)),
                sampling_rng_state=rng.getstate(), automatic_retry=False)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
