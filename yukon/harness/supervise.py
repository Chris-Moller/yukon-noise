"""Runs one Noise submission on Yukon as an unprivileged user, and times it.

The supervisor runs as root; the submission runs as `participant` under torchrun. The clock starts
when the run first asks for the training data and stops when it submits its final weights (the two
calls in noise_harness.py). The run cannot cheat the clock:

- the training data sits in a root-only directory until the clock starts, so nothing that depends
  on it can happen before;
- the clock is this process's monotonic clock, which the run cannot touch;
- every process of the run is stopped the moment it submits, before the weights are copied, and
  killed before anything is written to --out, so nothing it does after the clock stops counts.

Both phases are bounded: a run that has not asked for the data by --warmup-limit, or not submitted
by --training-limit after it did, is killed. Writes to --out, and leaves nothing else there:
final.pt, the submitted weights, and run.json, the outcome and its times. Exits 0 only when the run
submitted.
"""
import argparse
import json
import os
import pwd
import shutil
import socket
import socketserver
import stat
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from noise_harness import SOCKET_ENV

PARTICIPANT = 'participant'
REPO = Path('/opt/noise/repo')        # the benchmark's own files: its baseline script, and this harness
DATA = Path('/opt/noise/data')        # root-only (0700) until the clock starts
RUN = Path('/run/noise')              # the socket, and the supervisor's copy of the weights
WORKDIR = Path('/home/participant/work')
MAX_WEIGHTS_BYTES = 1 << 30           # the fixed model's state dict is ~150 MB
REAP_SECONDS = 30.0

Outcome = Literal['submitted', 'exited', 'warmup_limit', 'training_limit', 'refused']


class Refused(Exception):
    """The submission did something the harness does not accept; the message is for its author."""


@dataclass(frozen=True)
class Record:
    """run.json: what happened, and the times the verifier scores."""
    outcome: Outcome
    reason: str | None = None
    warmupSeconds: float | None = None
    trainingSeconds: float | None = None
    exitCode: int | None = None


@dataclass
class Run:
    """The clock and what the run handed over; the socket's threads write it under `lock`."""
    user: pwd.struct_passwd
    started: float = 0.0
    data_released: float | None = None
    submitted: float | None = None
    weights: Path | None = None
    refusal: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    ended: threading.Event = field(default_factory=threading.Event)

    def release_data(self) -> None:
        with self.lock:
            if self.data_released is None:
                self.data_released = time.monotonic()
                DATA.chmod(0o755)

    def take_submission(self, path: Path) -> None:
        with self.lock:
            if self.ended.is_set():
                return
            self.submitted = time.monotonic()
            signal_run(self.user, 'STOP')
            try:
                if self.data_released is None:
                    raise Refused('submit() was called before training_data()')
                self.weights = take_weights(path, self.user)
            except Refused as refusal:
                self.refusal = str(refusal)
            except OSError as error:
                self.refusal = f'the harness could not copy the submitted weights: {error}'
            finally:
                self.ended.set()


class Requests(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        run: Run = self.server.run  # type: ignore[attr-defined]
        try:
            request = json.loads(self.rfile.readline())
        except ValueError:
            return
        match request:
            case {'op': 'training_data'}:
                run.release_data()
                self.wfile.write(json.dumps({'path': str(DATA / 'train.pt')}).encode() + b'\n')
            case {'op': 'submit', 'path': str(path)}:
                run.take_submission(Path(path))  # the run is stopped: no reply


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, run: Run):
        super().__init__(str(path), Requests)
        self.run = run
        path.chmod(0o666)  # the run connects as `participant`


def as_participant(user: pwd.struct_passwd, *command: str, no_new_privs: bool = False) -> list[str]:
    return ['setpriv', f'--reuid={user.pw_uid}', f'--regid={user.pw_gid}', '--clear-groups',
            *(['--no-new-privs'] if no_new_privs else []), '--', *command]


def signal_run(user: pwd.struct_passwd, signal_name: str) -> None:
    """Signals every process of the run: kill -1 as its user reaches all of them and nothing else."""
    subprocess.run(as_participant(user, 'kill', f'-{signal_name}', '-1'), check=False, stderr=subprocess.DEVNULL)


def live_processes(user: pwd.struct_passwd) -> bool:
    """Whether any process of the run is still alive; zombies are dead and hold nothing."""
    states = subprocess.run(['ps', '-u', str(user.pw_uid), '-o', 'stat='], capture_output=True, text=True).stdout
    return any(not state.startswith('Z') for state in states.split())


def take_weights(path: Path, user: pwd.struct_passwd) -> Path:
    """Copies the submitted weights where the run cannot reach them: only a regular file it wrote."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise Refused(f'cannot read the submitted weights {path}: {error.strerror}') from error
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != user.pw_uid:
            raise Refused(f'{path} is not a file the run wrote')
        if info.st_size > MAX_WEIGHTS_BYTES:
            raise Refused(f'{path} is {info.st_size} bytes; the limit is {MAX_WEIGHTS_BYTES}')
        private = RUN / 'final.pt'
        with private.open('wb') as target:
            shutil.copyfileobj(source, target)
    return private


def resolve_hostname_locally() -> None:
    """The sandbox has no network, so no DNS; torchrun's rendezvous advertises the hostname."""
    hostname = socket.gethostname()
    hosts = Path('/etc/hosts')
    if hostname not in hosts.read_text().split():
        with hosts.open('a') as file:
            file.write(f'127.0.0.1 {hostname}\n')


def prepare_workdir(entry: Path, archive: Path | None, user: pwd.struct_passwd) -> None:
    """The benchmark's files, with the submission's script in place of the baseline's."""
    shutil.copytree(REPO, WORKDIR)
    if archive is not None:
        with tarfile.open(archive) as tar:
            member = next((m for m in tar.getmembers() if Path(m.name) == entry), None)
            if member is None or not member.isfile():
                raise Refused(f'the submission has no {entry}')
            tar.extract(member, WORKDIR, filter='data')
    for path in [WORKDIR, *WORKDIR.rglob('*')]:
        os.chown(path, user.pw_uid, user.pw_gid, follow_symlinks=False)


def end_run(process: subprocess.Popen, user: pwd.struct_passwd) -> int | None:
    """Kills and reaps every process of the run; torchrun's exit code if it exited by itself."""
    exited = process.poll()
    signal_run(user, 'KILL')
    process.wait()
    deadline = time.monotonic() + REAP_SECONDS
    while live_processes(user):
        if time.monotonic() > deadline:
            # A process that outlives SIGKILL could still write to --out: nothing there can be trusted.
            raise Refused(f'processes of the run survived SIGKILL for {REAP_SECONDS:.0f} seconds')
        time.sleep(0.1)
    return exited


def supervise(args: argparse.Namespace) -> tuple[Record, Path | None]:
    user = pwd.getpwnam(PARTICIPANT)
    try:
        prepare_workdir(args.entry, args.submission, user)
    except Refused as refusal:
        return Record('refused', reason=str(refusal)), None
    resolve_hostname_locally()
    RUN.mkdir(mode=0o711, exist_ok=True)
    run = Run(user=user)
    server = Server(RUN / 'harness.sock', run)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Loopback is the only interface; NCCL leaves it out of its bootstrap unless told otherwise.
    env = {**os.environ, 'HOME': user.pw_dir, 'USER': user.pw_name, 'NCCL_SOCKET_IFNAME': 'lo',
           SOCKET_ENV: str(RUN / 'harness.sock')}
    command = ['torchrun', '--standalone', f'--nproc_per_node={args.nproc}', str(args.entry)]
    run.started = time.monotonic()
    process = subprocess.Popen(as_participant(user, *command, no_new_privs=True), cwd=WORKDIR, env=env)
    outcome: Outcome = 'exited'
    while not run.ended.wait(timeout=1):
        if process.poll() is not None:
            break
        if run.data_released is None and time.monotonic() - run.started > args.warmup_limit:
            outcome = 'warmup_limit'
            break
        if run.data_released is not None and time.monotonic() - run.data_released > args.training_limit:
            outcome = 'training_limit'
            break
    try:
        exit_code = end_run(process, user)
    except Refused as refusal:
        return Record('refused', reason=str(refusal)), None
    finally:
        server.shutdown()
    warmup = None if run.data_released is None else round(run.data_released - run.started, 3)
    if not run.ended.is_set():
        return Record(outcome, warmupSeconds=warmup, exitCode=exit_code), None
    if run.refusal is not None:
        return Record('refused', reason=run.refusal, warmupSeconds=warmup), None
    training = round(run.submitted - run.data_released, 3)
    return Record('submitted', warmupSeconds=warmup, trainingSeconds=training), run.weights


def write_outputs(out: Path, record: Record, weights: Path | None) -> None:
    for child in out.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    if weights is not None:
        shutil.copyfile(weights, out / 'final.pt')
    (out / 'run.json').write_text(json.dumps(asdict(record), indent=2) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', type=Path, required=True, help='The script to run, relative to the repository.')
    parser.add_argument('--submission', type=Path, help="Yukon's submission archive; absent: the baseline script.")
    parser.add_argument('--out', type=Path, default=Path('/out'))
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--warmup-limit', type=float, default=900.0, help='Seconds before the clock must start.')
    parser.add_argument('--training-limit', type=float, default=3600.0, help='Seconds the clock may run.')
    args = parser.parse_args()
    record, weights = supervise(args)
    write_outputs(args.out, record, weights)
    print(f'harness: {json.dumps(asdict(record))}', flush=True)
    return 0 if record.outcome == 'submitted' else 1


if __name__ == '__main__':
    sys.exit(main())
