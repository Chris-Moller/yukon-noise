"""The supervisor's guarantees, checked by running it for real: as root, on the image's layout,
with each test's script as the submission and a one-process stand-in for torchrun (no GPU).

    docker build -f yukon/Dockerfile --target harness-test .
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path('/opt/noise/repo')
DATA = Path('/opt/noise/data')
WORKDIR = Path('/home/participant/work')
RUN = Path('/run/noise')
OUT = Path('/tmp/out')
ENTRY = Path('train/speedrun/run.py')
SUPERVISE = REPO / 'yukon/harness/supervise.py'
TRAINING_SET = b'the training set'
PRELUDE = """\
import os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yukon.harness import noise_harness
"""
# After the run submits, keeps rewriting its weights and forging an output, if it still can.
TAMPER = """\
import time
from pathlib import Path
time.sleep(0.3)
while True:
    Path('final.pt').write_bytes(b'tampered')
    Path('/tmp/out/forged.json').write_text('{}')
"""


@unittest.skipUnless(os.geteuid() == 0, 'the supervisor runs as root')
class Supervisor(unittest.TestCase):
    def setUp(self):
        for path in (WORKDIR, RUN, OUT):
            shutil.rmtree(path, ignore_errors=True)
        OUT.mkdir()
        DATA.mkdir(mode=0o700, exist_ok=True)
        DATA.chmod(0o700)
        (DATA / 'train.pt').write_bytes(TRAINING_SET)
        (DATA / 'train.pt').chmod(0o444)
        (REPO / ENTRY).parent.mkdir(parents=True, exist_ok=True)

    def supervise(self, script: str, *options: str) -> tuple[subprocess.CompletedProcess, dict]:
        (REPO / ENTRY).write_text(PRELUDE + textwrap.dedent(script))
        result = subprocess.run([sys.executable, str(SUPERVISE), '--entry', str(ENTRY), '--out', str(OUT),
                                 '--nproc', '1', *options], capture_output=True, text=True, timeout=120)
        return result, json.loads((OUT / 'run.json').read_text())

    def test_times_from_data_release_to_submission_and_keeps_the_submitted_weights(self):
        result, record = self.supervise("""
            time.sleep(2)  # warmup: untimed
            data = noise_harness.training_data(local=Path('unused'))
            time.sleep(1)  # training: timed
            weights = Path('final.pt')
            weights.write_bytes(b'weights of ' + data.read_bytes())
            noise_harness.submit(weights)
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record['outcome'], 'submitted')
        self.assertGreaterEqual(record['trainingSeconds'], 1.0)
        self.assertLess(record['trainingSeconds'], 2.0)
        self.assertGreaterEqual(record['warmupSeconds'], 2.0)
        self.assertEqual((OUT / 'final.pt').read_bytes(), b'weights of ' + TRAINING_SET)

    def test_the_training_data_is_unreadable_before_the_clock_starts(self):
        result, record = self.supervise("""
            try:
                Path('/opt/noise/data/train.pt').read_bytes()
                print('read the training data')
            except PermissionError:
                raise SystemExit(3)
        """)
        self.assertNotIn('read the training data', result.stdout)
        self.assertEqual((record['outcome'], record['exitCode']), ('exited', 3))
        self.assertEqual(result.returncode, 1)

    def test_a_run_that_never_submits_fails(self):
        result, record = self.supervise("noise_harness.training_data(local=Path('unused'))\n")
        self.assertEqual((record['outcome'], record['exitCode']), ('exited', 0))
        self.assertEqual(result.returncode, 1)
        self.assertFalse((OUT / 'final.pt').exists())

    def test_the_warmup_is_bounded(self):
        result, record = self.supervise("time.sleep(60)\n", '--warmup-limit', '1')
        self.assertEqual(record['outcome'], 'warmup_limit')
        self.assertEqual(result.returncode, 1)

    def test_the_training_is_bounded(self):
        result, record = self.supervise("""
            noise_harness.training_data(local=Path('unused'))
            time.sleep(60)
        """, '--training-limit', '1')
        self.assertEqual(record['outcome'], 'training_limit')
        self.assertIsNone(record['trainingSeconds'])
        self.assertEqual(result.returncode, 1)

    def test_a_submission_before_the_data_is_refused(self):
        _, record = self.supervise("""
            Path('final.pt').write_bytes(b'weights')
            noise_harness.submit(Path('final.pt'))
        """)
        self.assertEqual(record['outcome'], 'refused')
        self.assertIn('before training_data()', record['reason'])

    def test_only_a_regular_file_the_run_wrote_is_taken(self):
        _, record = self.supervise("""
            noise_harness.training_data(local=Path('unused'))
            Path('final.pt').symlink_to('/etc/shadow')
            noise_harness.submit(Path('final.pt'))
        """)
        self.assertEqual(record['outcome'], 'refused')
        self.assertFalse((OUT / 'final.pt').exists())

    def test_nothing_the_run_does_after_submitting_reaches_the_output(self):
        OUT.chmod(0o777)  # writable by the run, as a misconfigured volume might be
        result, record = self.supervise(f"""
            noise_harness.training_data(local=Path('unused'))
            weights = Path('final.pt')
            weights.write_bytes(b'submitted')
            subprocess.Popen(['python', '-c', {TAMPER!r}])
            noise_harness.submit(weights)
        """)
        time.sleep(1)  # a surviving process would have written by now
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record['outcome'], 'submitted')
        self.assertEqual((OUT / 'final.pt').read_bytes(), b'submitted')
        self.assertEqual(sorted(p.name for p in OUT.iterdir()), ['final.pt', 'run.json'])

    def test_the_output_holds_only_what_the_supervisor_wrote(self):
        (OUT / 'forged.json').write_text('{}')
        _, record = self.supervise("""
            noise_harness.training_data(local=Path('unused'))
            Path('final.pt').write_bytes(b'weights')
            noise_harness.submit(Path('final.pt'))
        """)
        self.assertEqual(record['outcome'], 'submitted')
        self.assertEqual(sorted(p.name for p in OUT.iterdir()), ['final.pt', 'run.json'])


if __name__ == '__main__':
    unittest.main()
