"""Turns a local run's result.json into Yukon's score file, for `yukon run` on your own 8xH100 node.

    uv run yukon/local_score.py logs/speedrun/result.json

A local score is the script's own timer and validation loss. Yukon's official score comes from its
own harness instead: the time from the first read of the training data to the submitted weights
(yukon/README.md).
"""
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
score = Path('.yukon/score.json')
score.parent.mkdir(exist_ok=True)
score.write_text(json.dumps({'score': result['train_seconds'], 'metrics': {'val_loss': result['val_loss']}}) + '\n')
