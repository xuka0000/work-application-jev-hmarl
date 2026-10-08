"""Run a pre-registered serial development batch with active-session observation."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    if any(job['method'].startswith('random_') for job in protocol['jobs']):
        actual=hashlib.sha256((ROOT/'scripts/jev_random_controls.py').read_bytes()).hexdigest()
        if actual!=protocol.get('helper_sha256'):
            raise ValueError('Random-control helper differs from its registered version')
    status_path = args.protocol.with_name('status.json')
    if status_path.exists():
        raise ValueError('Existing batch attempts are never silently resumed or retried')
    state = dict(status='registered', completed=[], protocol=str(args.protocol), automatic_retry=False,
                 supervision='Active Codex session only; no independent notification service')

    def save():
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8'); temp.replace(status_path)

    save()
    try:
        for job in protocol['jobs']:
            if datetime.now(timezone.utc) >= datetime.fromisoformat(protocol['deadline_utc']):
                raise TimeoutError('Registered study deadline reached')
            output = ROOT / job['output']; log = output.with_suffix('.process.log')
            if output.exists() or log.exists():
                raise ValueError('Immutable trial output already exists')
            command = [sys.executable, 'scripts/run_jev_online_trial.py', '--stage', 'development',
                       '--case-id', job['case_id'], '--method', job['method'], '--profile', job['profile'],
                       '--channel-seed', '301', '--horizon-s', '14400', '--wall-limit-s', '1200',
                       '--output', str(output)]
            if job['method'].startswith('online_'):
                command += ['--request-variant', protocol['request_variant'], '--stop-on-model-failure']
            if job.get('checkpoint'):
                command += ['--checkpoint', str(ROOT / job['checkpoint'])]
            if job.get('llm_config'):
                command += ['--llm-config', str(ROOT / job['llm_config'])]
            if job.get('random_seed') is not None:
                command += ['--random-seed', str(job['random_seed'])]
            state.update(status='running', current=dict(job)); save()
            with log.open('x', encoding='utf-8') as handle:
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > 1230:
                            raise TimeoutError('Registered trial plus cleanup allowance exceeded')
                        source = output/'status.json'
                        if source.exists():
                            trial = json.loads(source.read_text(encoding='utf-8'))
                            state['current'].update(trial_status=trial['status'], physical_time_s=trial['physical_time_s'],
                                                    elapsed_s=trial['elapsed_s'])
                            if trial['status'].startswith('failed'):
                                raise RuntimeError('Trial failed; batch stops for diagnosis')
                        save(); print(json.dumps(state['current']), flush=True); time.sleep(10)
                    if process.returncode:
                        raise RuntimeError('Trial exited unsuccessfully: '+str(process.returncode))
                finally:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=30)
            trial = json.loads((output/'status.json').read_text(encoding='utf-8'))
            if trial['status'] != 'completed' or not (output/'verification.json').is_file():
                raise ValueError('Physical evidence incomplete')
            state['completed'].append(dict(job, metrics=trial['metrics'])); save()
        state.update(status='completed_development_batch', current=None); save()
    except BaseException as exc:
        state.update(status='failed_stopped', error_type=type(exc).__name__, error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
