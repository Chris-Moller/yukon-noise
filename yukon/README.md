# Noise on Yukon

This branch runs [Noise](../README.md)'s speedrun track (train the fixed model to validation loss
3.5 without backpropagation, as fast as possible) on Yukon. Submissions change one file,
[`train/speedrun/run.py`](../train/speedrun/run.py); Yukon runs it on one 8xH100 node through
gpu-runner, measures it with a trusted harness and scores it with Q Labs' own `utils/eval.py`.

## What is timed

On Yukon the clock is not your script's timer. A supervisor runs as root and starts your script as
an unprivileged user; the training data stays unreadable until you ask for it:

```python
from yukon.harness import noise_harness

path = noise_harness.training_data(local=args.data_dir / 'train.pt')  # the clock starts
...                                                                    # timed: all your training
torch.save(state, args.output / 'final.pt')
noise_harness.submit(args.output / 'final.pt')                         # the clock stops
```

- **Timed:** everything from the first `training_data()` call, on any rank, to `submit()`.
- **Untimed warmup:** everything before `training_data()`: imports, compilation, CUDA graph capture
  on random tokens. It cannot depend on the data, and it is limited to 15 minutes. Q Labs' rule
  still applies: training starts from a fresh state.
- **`submit()` does not return.** The supervisor stops every process of the run the moment it is
  called, takes the weights, and kills the run. Call it from one rank, after the weights are saved.
- **No validation set.** Only the verifier has `val.pt`; skip your own evaluation when it is absent,
  as the baseline does.

Run locally (no supervisor), `training_data()` returns its `local` path and `submit()` does nothing,
so the same script runs in both places.

## How a run is scored

A submission passes when its validation loss is at most 3.5, and its score is the training time.
Runs are slightly noisy (about ±0.0015 in loss), so a run whose loss lands within 0.003 of the
target is repeated, up to three runs; the mean loss then decides, and the score is the median time.
The settings are in [`benchmark.json`](../benchmark.json).

## The images

[`Dockerfile`](Dockerfile) builds both images from this repository, and
[`.github/workflows/yukon-images.yml`](../.github/workflows/yukon-images.yml) pushes them:

| Image | Holds | Runs |
| --- | --- | --- |
| participant | the pinned stack, the baseline, the harness, `train.pt` (root-only) | [`harness/supervise.py`](harness/supervise.py) with your script |
| verifier | the pinned stack, `utils/eval.py`, `val.pt` | [`verifier/verify.py`](verifier/verify.py) on the run's output |

`docker build -f yukon/Dockerfile --target harness-test .` runs the supervisor's tests, as root,
without a GPU.
