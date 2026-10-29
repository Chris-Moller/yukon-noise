# syntax=docker/dockerfile:1.7
# The two images Yukon runs Noise with, built from the repository root:
#
#   docker build -f yukon/Dockerfile --target harness-test .  # the supervisor's tests; no GPU
#   docker build -f yukon/Dockerfile --target participant .   # runs a submission (gpu-runner's job)
#   docker build -f yukon/Dockerfile --target verifier .      # scores its output (gpu-runner's verifier)
#
# Both share one Python stack, pinned by uv.lock, and one copy of the data, made by Q Labs'
# prepare_data.py, which checks it against pinned hashes. Only the participant image holds the
# training set, root-only until the supervisor starts the clock; only the verifier holds the
# validation set.

FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS stack
# Triton and Inductor compile kernels at run time; procps and util-linux are the supervisor's ps and setpriv.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ libc6-dev procps util-linux \
 && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/noise/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/noise/venv/bin:$PATH
WORKDIR /opt/noise/repo
COPY pyproject.toml uv.lock ./
# uv and its download cache are mounted for this step only: the image holds the environment alone.
RUN --mount=from=uv,source=/uv,target=/bin/uv --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY utils/ utils/
COPY tokenizer/ tokenizer/

FROM stack AS data
# prepare_data.py can abort in its own teardown after writing correct files (a streaming thread at
# exit), so its exit code does not decide: load_rows checks both files against the pinned hashes.
RUN python utils/prepare_data.py; \
    python -c "from utils.prepare_data import TRAIN_ROWS, VAL_ROWS, load_rows; load_rows('data/train.pt', TRAIN_ROWS); load_rows('data/val.pt', VAL_ROWS)"

FROM stack AS participant
RUN useradd --create-home --uid 1000 participant
COPY train/speedrun/run.py train/speedrun/run.py
COPY yukon/harness/ yukon/harness/
COPY --from=data --chmod=0444 /opt/noise/repo/data/train.pt /opt/noise/data/train.pt
# The data directory opens when the clock starts. No setuid binary may undo the privilege drop.
RUN chmod 0700 /opt/noise/data \
 && find / -xdev -perm /6000 -type f -exec chmod a-s {} +
CMD ["python", "/opt/noise/repo/yukon/harness/supervise.py", "--entry", "train/speedrun/run.py"]

FROM stack AS verifier
COPY yukon/verifier/ /opt/noise/verifier/
COPY --from=data /opt/noise/repo/data/val.pt /opt/noise/data/val.pt
ENV PYTHONPATH=/opt/noise/repo
CMD ["python", "/opt/noise/verifier/verify.py"]

# The supervisor run for real, as root, with a one-process stand-in for torchrun: building this target
# runs the tests.
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337 AS harness-test
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 participant
COPY yukon/harness/ /opt/noise/repo/yukon/harness/
COPY --chmod=0755 yukon/tests/torchrun /usr/local/bin/torchrun
COPY yukon/tests/ /opt/noise/tests/
RUN python -m unittest discover --start-directory /opt/noise/tests --verbose
