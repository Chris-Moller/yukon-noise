"""Scores one Noise run on Yukon: the supervisor's times, and Q Labs' own evaluation of the weights.

Reads the participant's output, which gpu-runner mounts read-only at --submission: run.json, which
the supervisor wrote after the run was killed, and final.pt, the weights it took the moment the
clock stopped. Writes gpu-runner's verifier result to --result:

- scored: the training time is the score; the validation loss (utils/eval.py, unchanged) is a
  metric, which Yukon checks against the target;
- rejected: the run did not submit, or its weights do not load into the fixed model, or their loss
  is not finite. These are the submission's own failures.

Anything else (a CUDA error during evaluation, say) is not the submission's fault: the verifier
crashes, and gpu-runner retries it as an infrastructure failure.
"""
import argparse
import json
import pickle
from pathlib import Path

import torch

from utils.eval import evaluate, load_model


def verdict(submission: Path, data_dir: Path) -> dict:
    record = json.loads((submission / 'run.json').read_text())
    if record['outcome'] != 'submitted':
        return {'status': 'rejected', 'message': f"the run did not submit weights ({record['outcome']})"}
    try:
        state = torch.load(submission / 'final.pt', map_location='cpu', weights_only=True)
        load_model(state, torch.device('cpu'))  # a missing, extra or reshaped tensor raises here
    except (pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as error:
        return {'status': 'rejected', 'message': f'the weights do not load into the fixed model: {error}'}
    try:
        val_loss = evaluate(state, data_dir)
    except FloatingPointError:
        return {'status': 'rejected', 'message': 'the weights give a non-finite validation loss'}
    return {
        'status': 'scored',
        'score': record['trainingSeconds'],
        'metrics': {
            'val_loss': val_loss,
            'training_seconds': record['trainingSeconds'],
            'warmup_seconds': record['warmupSeconds'],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--submission', type=Path, default=Path('/submission'))
    parser.add_argument('--data-dir', type=Path, default=Path('/opt/noise/data'))
    parser.add_argument('--result', type=Path, default=Path('/verifier/result.json'))
    args = parser.parse_args()
    torch.cuda.set_device(0)
    result = verdict(args.submission, args.data_dir)
    args.result.write_text(json.dumps(result) + '\n')
    print(f'verifier: {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
