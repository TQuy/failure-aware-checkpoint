# Baseline results, 15–16 November 2026

GPT-2 large on one NVIDIA T4 (n1-standard-8, 30 GB), checkpoints written over
NFS to a separate storage machine, one failure per hour on average, 2-hour runs.
`pairN/results/seedN/` is the matrix of seed N; `pairN/results/rerun-seedN/` is
the DCP rerun of 16 November, which replaces `seedN/dcp-46`.

`seedN/dcp-46` is kept only as a record of a fault: those runs never deleted old
DCP checkpoints, filled the storage disk and trained on without checkpoints
(fixed in ff0fed0). Do not use their numbers.

`baseline_table.md` is built by `python experiments/baseline_table.py results`.
Each cell folder holds the supervisor's kill record, the training record
(losses, step times, saves), the training log and the accounting.

## Week 13 validation, 26–27 November 2026

`w13/seedN/` holds seed N of the Week 13 matrix: the same model, machines and
storage, both failure rates (`m900/`, `m3600/`), four policies each
(CheckFreq 1% and 5% with the first-stall fix, DCP every 64 steps, the
failure-aware rule), 2-hour runs, kill times shared within a seed and a rate.
Seeds 1 and 5 were stopped once by the cloud provider; the folders ending in
`-aborted-*` and the second `ref-`/`warmup-` folders come from that, and the
unfinished group was rerun on the new boot. Only folders whose `cell.txt` says
`finished:` are used.

`w13/w13_table.md` is built by `python experiments/w13_table.py results`.
