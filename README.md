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
| Two-phase checkpoint library | In progress |
| Failure injector and supervisor | In progress |
| Lost-time accounting | In progress |
| Baseline runs (GPT-2 large, one T4) | Planned for 20 November |

## Running the tests

    python -m venv .venv && . .venv/bin/activate
    pip install -e .
    python -m unittest discover -s tests -v

## Licence

MIT. All code here was written for this project.
