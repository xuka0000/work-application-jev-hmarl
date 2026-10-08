"""Canonical, local authorization and runtime-only evidence compatibility."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'outputs/jev_hmarl_publication_20261002/registration/public_baseline12_protocol.json'
LEGACY_DEADLINE = '2026-10-06T15:59:00+00:00'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@lru_cache(maxsize=2)
def _plan(path,mtime_ns,size):
    return read(path)

def plan():
    if not PROTOCOL.is_file():return {}
    stat=PROTOCOL.stat()
    return _plan(str(PROTOCOL),stat.st_mtime_ns,stat.st_size)

def cached_read(path):
    path=Path(path);stat=path.stat()
    return _plan(str(path),stat.st_mtime_ns,stat.st_size)

@lru_cache(maxsize=4)
def _hash(path,mtime_ns,size):return digest(path)

def protocol_digest(path=PROTOCOL):
    path=Path(path);stat=path.stat()
    return _hash(str(path),stat.st_mtime_ns,stat.st_size)

def policy():
    return plan().get('execution_policy', {})

def deadline():
    return plan().get('deadline_utc', LEGACY_DEADLINE)

def wall_limit(training=False):
    return policy().get('training_cumulative_wall_s' if training else 'trajectory_cumulative_wall_s', 1200)

def remaining_wall(training=False):
    return min(wall_limit(training), float(os.environ.get('JEV_TASK_REMAINING_WALL_S', wall_limit(training))))

def scientific_job(job):
    return {k:v for k,v in job.items() if k not in ('timeout_s','wall_limit_s','cpu_slots','_prior_elapsed_s','_attempt','_recovery_fingerprint')}

def compatible_job(old, new):
    return scientific_job(old) == scientific_job(new)

def compatible_result(status, job, protocol_path=PROTOCOL):
    """Accept only the pinned ancestor and an identical scientific task."""
    protocol_path=Path(protocol_path)
    if status.get('protocol_sha256') == protocol_digest(protocol_path):
        return status.get('job') == job
    current_plan=cached_read(protocol_path)
    receipt=current_plan.get('runtime_compatibility')
    if not receipt:return False
    path=ROOT/receipt['path']
    if digest(path)!=receipt['sha256']:raise ValueError('Compatibility receipt changed')
    audit=cached_read(path)
    ancestors=audit.get('ancestors')
    if ancestors is None:ancestors=[dict(path=audit['old_protocol_path'],sha256=audit['old_protocol_sha256'])]
    for binding in ancestors:
        ancestor=ROOT/binding['path']
        if protocol_digest(ancestor)!=binding['sha256']:raise ValueError('Ancestor protocol changed')
        if status.get('protocol_sha256')!=binding['sha256']:continue
        original=cached_read(ancestor)
        expected=next((j for key in ('development_jobs','formal_jobs') for j in original[key]
                       if j['task_id']==job['task_id']),None)
        return expected is not None and status.get('job')==expected and compatible_job(expected,job)
    return False

def verify_manifest_handoff(old_hash, manifest_path, evidence):
    """No automatic authorization from a newer file alone."""
    receipt=evidence.get('execution_amendment')
    if not receipt:raise ValueError('Resume scientific manifest differs')
    p=ROOT/receipt['path']
    if digest(p)!=receipt['sha256']:raise ValueError('Execution amendment receipt changed')
    a=read(p);old_path=ROOT/a['old_manifest_path'];current=read(manifest_path)
    if digest(old_path)!=old_hash or a['old_manifest_sha256']!=old_hash:
        raise ValueError('Original manifest identity differs')
    old=read(old_path)
    ignored={'deadline_utc','source_sha256','input_sha256','jobs','execution_amendment'}
    if {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:
        raise ValueError('Queue resource or scientific scope changed')
    if len(old['jobs'])!=len(current['jobs']) or any(not compatible_job(x,y) for x,y in zip(old['jobs'],current['jobs'])):
        raise ValueError('Scientific task matrix changed')
    for rel,h in old['source_sha256'].items():
        if current['source_sha256'].get(rel)!=h and rel not in a['runtime_source_changes']:
            raise ValueError('Unapproved scientific source change: '+rel)
        if rel in a['runtime_source_changes'] and current['source_sha256'].get(rel)!=a['runtime_source_changes'][rel]['new_sha256']:
            raise ValueError('Amended execution source changed: '+rel)
    protocol=PROTOCOL.relative_to(ROOT).as_posix()
    if {k:v for k,v in old['input_sha256'].items() if k!=protocol}!={k:v for k,v in current['input_sha256'].items() if k!=protocol}:
        raise ValueError('Scientific input changed')

def cost_state(lease, ledger, now=None):
    """Conservative accrued cost plus live reservations, never reset old spend."""
    now=now or datetime.now(timezone.utc).timestamp()
    rate=lease.get('price_cny_per_hour')
    if not isinstance(rate,(int,float)) or rate<=0:raise ValueError('Verified hourly price required')
    compute=max(0,now-lease['started_unix_s'])*rate/3600
    spent=ledger.get('settled_cost_cny',0)+compute
    reserved=sum(x['amount_cny'] for x in ledger.get('live_reservations',[]))
    cap=policy().get('additional_cost_cap_cny')
    if not isinstance(cap,(int,float)) or cap<=0:raise ValueError('Additional cost authority required')
    return dict(accrued_upper_bound_cny=spent,live_reserved_cny=reserved,cap_cny=cap,
                remaining_cny=cap-spent-reserved,stop_required=spent+reserved>=cap)
