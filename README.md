# failure-aware-checkpoint

A long training job saves its state (a checkpoint) so it can resume after a crash. Saving often costs training time; saving rarely means redoing more work after each failure. The right interval depends on how often failures happen.

Today the interval is usually a number typed into the job's configuration, or, with CheckFreq (FAST 2021), it is derived from an overhead budget such as "spend at most 5% of the time on checkpoints". Neither uses the failure rate. This library chooses the interval from the measured failure rate and the real cost of a two-phase save, and measures whether that loses less GPU-time than the rules people use today.

Course project for Machine Learning & Artificial Intelligence, Soongsil University, Fall 2026.

## The cost model

For a checkpoint interval T, the fraction of training time lost is

    L(T) = C/T + λ (τ + T/2 + R)

- C: the cost of one save to the training loop (the stall while the state is copied to CPU memory, plus the step slowdown while the background write runs);
- τ: the background write time; a snapshot only counts once its write has finished;
- R: the restore time after a failure;
- λ = 1/M: the failure rate, with M the mean time between failures.

The first term is paid on every save. The second is paid on every failure: the newest finished checkpoint is on average τ + T/2 old when a failure hits, and that work is redone. The minimum is Young's T* = √(2CM), never shorter than the time one background write needs.

## What is compared

| Policy | Interval comes from |
|---|---|
| Fixed interval | a number in the configuration |
| PyTorch Distributed Checkpoint `async_save`, fixed interval | a number in the configuration; the baseline used in practice in 2026 |
| CheckFreq, 1% and 5% budgets | stall ÷ budget, re-implemented from the paper |
| Failure-aware (this project) | minimum of L(T) at the measured failure rate |

Every policy sees the same seeded failure times. Lost time is computed from the execution record (redone steps, stall, slowdown, restore), not from the policy's own estimate, and is cross-checked against the wall clock.

## Status

| Part | Status |
|---|---|
| Cost model and interval rules (`src/facp/cost_model.py`) | Done; tests reproduce the proposal's estimate table |
| Two-phase checkpoint library and the DCP baseline (`src/facp/checkpoint.py`) | Done; tested with kills on a tiny model and on GPT-2 small on a GPU, every redone step reproduces its logged loss |
| Failure injector and supervisor (`src/facp/supervisor.py`) | Done |
| Lost-time accounting (`src/facp/accounting.py`) | Done; the four parts explain the measured loss to within a few seconds |
| Baseline runs (GPT-2 large, one T4) | Done 15–16 November, three seeds at one failure per hour; table in [`results/baseline_table.md`](results/baseline_table.md), notes in [`results/README.md`](results/README.md) |
| Failure-aware rule against the baselines, at one failure per 15 and per 60 minutes | Done 26–27 November, five seeds, all four policies at both rates in one boot per seed; the cost model checked against each run. Tables in [`results/w13/w13_table.md`](results/w13/w13_table.md) |

## Running it

    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[train]"
    python -m unittest discover -s tests -v      # about 2 minutes on a CPU

One run: GPT-2 small, a checkpoint policy, kills at 60 and 150 seconds, then the accounting.

    python -m facp.supervisor --duration 240 --record run/supervisor.jsonl --kill-at 60 150 -- \
        python examples/train_gpt2.py --model gpt2 --policy failure-aware:3600 \
            --ckpt-dir run/ckpt --record run/train.jsonl --batch 2 --seq 256 --grad-ckpt
    python -m facp.accounting run/

Policies: `none`, `fixed:N` (every N steps), `dcp:N` (PyTorch Distributed Checkpoint `async_save` every N steps), `checkfreq:BUDGET` (for example `checkfreq:0.01`), `failure-aware:MTBF_SECONDS`. With `--mtbf` and `--seed` instead of `--kill-at`, the supervisor draws the kill times, and the same seed gives the same times for every policy.

## Reproducing the baseline

The baseline table comes from one command per seed on a GPU machine. It runs a thrown-away warm-up, a reference without checkpoints, then CheckFreq at 1% and 5% and DCP every 46 steps, 2 hours each, all killed at the same seeded moments.

    SEED=1 CELLS="ref checkfreq:0.01 checkfreq:0.05 dcp:46" SHUTDOWN=0 \
        bash ~/facp/experiments/run-matrix.sh

It expects the repository at `~/facp` and the checkpoint storage at `/mnt/store`. On Google Cloud, `experiments/cloud.sh` creates a matching pair of machines (an n1-standard-8 with one T4, and a small NFS storage machine) and copies the repository over; `experiments/setup-store.sh` and `experiments/setup-gpu.sh` prepare them:

    PROJECT=my-project experiments/cloud.sh create 1
    PROJECT=my-project experiments/cloud.sh ssh 1 store "bash -s" < experiments/setup-store.sh
    PROJECT=my-project experiments/cloud.sh push 1
    PROJECT=my-project experiments/cloud.sh ssh 1 gpu bash ~/facp/experiments/setup-gpu.sh $(PROJECT=my-project experiments/cloud.sh store-ip 1)

A matrix takes about 6.5 hours and about 5.5 USD on demand. Afterwards, `cloud.sh pull 1 results/pair1` copies the results back, and `python experiments/baseline_table.py results` rebuilds the table.

## Reproducing the Week 13 validation

Each seed runs on its own GPU machine, in one boot: a warm-up, a reference, then the four policies at one failure per 15 minutes, then the four at one per 60 minutes, 2 hours each (about 16 hours 20 minutes in all).

    experiments/launch-w13.sh 1      # creates or starts pair 1 and starts seed 1

`experiments/watch-w13.sh` watches the pairs from another machine: it restarts a spot machine the cloud stopped, resumes its seed (a failure-rate group that did not finish in one boot is rerun in full), copies the results and stops each pair when its seed is done. With the results in `results/w13/seedN/`, `python experiments/w13_table.py results` rebuilds the tables, including the cost model against measurement.

## Licence

MIT. All code here was written for this project.
