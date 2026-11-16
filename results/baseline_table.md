## Lost GPU-seconds per training hour (one failure per hour, 2-hour runs)

| Policy | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|
| CheckFreq 1% | 301 | 60 | 399 | 253 |
| CheckFreq 5% | 140 | 54 | 344 | 179 |
| DCP every 46 steps | 635 | 476 | 1059 | 723 |
| Kills in the run | 1 | 0 | 3 | |

## Paired differences (first minus second, seconds per hour, mean and 95% interval over seeds)

- CheckFreq 1% minus CheckFreq 5%: +74 [-124, +272] (n = 3)
- DCP every 46 steps minus CheckFreq 1%: +470 [+49, +891] (n = 3)
- DCP every 46 steps minus CheckFreq 5%: +544 [+165, +923] (n = 3)

## Parts of the lost time (seconds, summed over seeds)

| Policy | Stall | Slowdown | Redone | Restart | Residual | Saves |
|---|---|---|---|---|---|---|
| CheckFreq 1% | 213 | 72 | 799 | 408 | +26.4 | 216 |
| CheckFreq 5% | 274 | 77 | 317 | 399 | +6.0 | 277 |
| DCP every 46 steps | 2473 | 11 | 1562 | 286 | +4.6 | 84 |

Largest residual in any one run: 12.5 s.
Redone steps that reproduced their logged loss: 548 of 548.

## Host memory (GB of 30)

- seed1: peak used 27.6, lowest available 1.3
- rerun-seed1: peak used 17.7, lowest available 11.2
- seed2: peak used 27.6, lowest available 1.3
- rerun-seed2: peak used 17.8, lowest available 11.1
- seed3: peak used 27.6, lowest available 1.3
- rerun-seed3: peak used 17.9, lowest available 11.1
