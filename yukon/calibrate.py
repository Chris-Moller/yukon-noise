"""Runs the baseline on gpu-runner several times and reports its spread, for benchmark.json.

    GPU_RUNNER_URL=… GPU_RUNNER_TOKEN=… python yukon/calibrate.py --runs 5

Each run is benchmark.json's job without a submission, so the participant image runs its own
baseline. The median training time is the baseline score; the validation loss's spread sets the
gate's margin. Runs go to gpu-runner as they would from Yukon, so they wait for GPUs the same way.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

BENCHMARK = json.loads((Path(__file__).resolve().parents[1] / 'benchmark.json').read_text())
VERIFIER_RESULT = {'path': '/verifier/result.json', 'maxBytes': 65536}  # as Yukon's gpu provider sends it


def api(method: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        os.environ['GPU_RUNNER_URL'].rstrip('/') + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={'authorization': f"Bearer {os.environ['GPU_RUNNER_TOKEN']}", 'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def baseline_job(runner: dict, name: str) -> dict:
    command = runner['command']
    if '--submission' in command:  # no archive: the image runs its own baseline
        at = command.index('--submission')
        command = command[:at] + command[at + 2:]
    return {
        'externalId': name, 'teamId': 'calibration', 'image': runner['image'], 'command': command,
        'machine': runner['machine'], 'allowedMachines': [runner['machine']], 'output': runner['output'],
        'verifier': {**runner['verifier'], 'output': VERIFIER_RESULT},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--name', default=time.strftime('noise-calibration-%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    runner = BENCHMARK['runner']
    jobs = [api('POST', '/v1/jobs', baseline_job(runner, f'{args.name}-{n}'))['id'] for n in range(1, args.runs + 1)]
    print(f'submitted {len(jobs)} runs: {", ".join(jobs)}', flush=True)
    done: dict[str, dict] = {}
    while len(done) < len(jobs):
        time.sleep(30)
        for job in jobs:
            if job not in done and (view := api('GET', f'/v1/jobs/{job}'))['state'] in ('succeeded', 'failed', 'cancelled'):
                done[job] = view
                print(f"{job}: {view['state']} {json.dumps(view['result'] or view['failure'])}", flush=True)
    scored = [view['result'] for view in done.values() if view['state'] == 'succeeded']
    if len(scored) < 2:
        raise SystemExit(f'{len(scored)} of {len(jobs)} runs scored; need at least two for a spread')
    seconds = [result['score'] for result in scored]
    losses = [result['metrics']['val_loss'] for result in scored]
    print(json.dumps({
        'runs': len(scored),
        'training_seconds': {'median': statistics.median(seconds), 'stdev': statistics.stdev(seconds), 'min': min(seconds), 'max': max(seconds)},
        'val_loss': {'mean': statistics.mean(losses), 'stdev': statistics.stdev(losses), 'min': min(losses), 'max': max(losses)},
        'suggested': {'baselineScore': statistics.median(seconds), 'gate.margin': round(2 * statistics.stdev(losses), 4)},
    }, indent=2))


if __name__ == '__main__':
    main()
