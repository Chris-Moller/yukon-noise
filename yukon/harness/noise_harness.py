"""The two calls that bracket a timed Noise run on Yukon.

On Yukon a supervisor owns the training data and the clock (see supervise.py):

    from yukon.harness import noise_harness
    path = noise_harness.training_data(local=args.data_dir / 'train.pt')  # starts the clock
    ...train...
    noise_harness.submit(weights_path)  # stops the clock; the supervisor takes these weights

Everything between the two calls is timed. Anything before the first is untimed warmup, which
cannot depend on the training data: it is not readable until the call returns. Every rank may call
training_data(); the first call starts the clock. One rank calls submit() after saving the final
state dict with torch.save, and never returns from it: the supervisor stops every process of the run
the moment it receives the call.

Without a supervisor (a local run) training_data() returns its `local` path and submit() does
nothing, so the same script runs unchanged in both places.
"""
import json
import os
import socket
from pathlib import Path

SOCKET_ENV = 'NOISE_HARNESS_SOCKET'


def training_data(local: Path) -> Path:
    """The training set's path: under Yukon the supervisor's, and the first call starts the clock."""
    if SOCKET_ENV not in os.environ:
        return local
    return Path(_request({'op': 'training_data'})['path'])


def submit(weights: Path) -> None:
    """Hand the final weights over; under Yukon this stops the clock and does not return."""
    if SOCKET_ENV not in os.environ:
        return
    _request({'op': 'submit', 'path': str(Path(weights).resolve())})
    raise RuntimeError('the supervisor did not stop the run after submit()')


def _request(message: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.environ[SOCKET_ENV])
        connection.sendall(json.dumps(message).encode() + b'\n')
        reply = connection.makefile('rb').readline()
    if not reply:
        raise RuntimeError(f'the supervisor closed the connection on {message["op"]}')
    return json.loads(reply)
