Seeds: 1, 2, 3, 4, 5. Runs found: 40 of 40.

## Lost GPU-seconds per training hour, one failure per 15 minutes (2-hour runs)

| Policy | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean |
|---|---|---|---|---|---|---|
| CheckFreq 1% | 894 | 302 | 896 | 910 | 480 | 696 |
| CheckFreq 5% | 893 | 281 | 926 | 855 | 438 | 679 |
| DCP every 64 steps | 1827 | 610 | 1780 | 1623 | 1219 | 1412 |
| Failure-aware | 866 | 283 | 856 | 906 | 511 | 684 |
| Kills in the run | 10 | 4 | 10 | 10 | 4 | |
| Interval chosen, steps (failure-aware) | 16–30 | 17–18 | 10–18 | 10–27 | 15–32 | |
| Interval chosen, steps (CheckFreq 1%) | 17–19 | 18–19 | 10–19 | 10–17 | 16–19 | |

## Lost GPU-seconds per training hour, one failure per 60 minutes (2-hour runs)

| Policy | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean |
|---|---|---|---|---|---|---|
| CheckFreq 1% | 137 | 51 | 356 | 670 | 161 | 275 |
| CheckFreq 5% | 159 | 59 | 379 | 696 | 172 | 293 |
| DCP every 64 steps | 442 | 188 | 935 | 1547 | 488 | 720 |
| Failure-aware | 165 | 50 | 346 | 663 | 166 | 278 |
| Kills in the run | 1 | 0 | 3 | 6 | 1 | |
| Interval chosen, steps (failure-aware) | 18–23 | 18–22 | 17–30 | 16–47 | 15–56 | |
| Interval chosen, steps (CheckFreq 1%) | 18–20 | 18–19 | 18–19 | 16–18 | 16–20 | |

## Paired differences over shared kill times (first minus second, seconds per hour, mean [95% interval])

One failure per 15 minutes:

- Failure-aware minus CheckFreq 1%: -12 [-46, +23] (n = 5; -2% of CheckFreq 1%)
- Failure-aware minus CheckFreq 5%: +6 [-66, +78] (n = 5; +1% of CheckFreq 5%)
- CheckFreq 1% minus CheckFreq 5%: +17 [-24, +59] (n = 5; +3% of CheckFreq 5%)
- Failure-aware minus DCP every 64 steps: -727 [-1040, -415] (n = 5; -52% of DCP every 64 steps)

One failure per 60 minutes:

- Failure-aware minus CheckFreq 1%: +3 [-16, +22] (n = 5; +1% of CheckFreq 1%)
- Failure-aware minus CheckFreq 5%: -15 [-36, +6] (n = 5; -5% of CheckFreq 5%)
- CheckFreq 1% minus CheckFreq 5%: -18 [-27, -8] (n = 5; -6% of CheckFreq 5%)
- Failure-aware minus DCP every 64 steps: -442 [-809, -74] (n = 5; -61% of DCP every 64 steps)

## Cost-model inputs, pooled over all runs of each checkpointer

| Input | Two-phase (CheckFreq, failure-aware) | DCP async_save |
|---|---|---|
| Stall per save, median (s) | 0.74 | 2.93 |
| Background write, mean (s) | 59 | 293 |
| Step slowdown while writing | 0.8% | -0.2% |
| Cost of one save C (s) | 1.20 | 2.48 |
| Restart per kill R (s) | 90 | 40 |

Waiting for the previous write, per 2-hour run: two-phase 11–36 s, DCP 140–297 s. DCP's measured interval was 278–319 s.

## Model against measurement (seconds, summed over seeds)

The model is given each run's measured interval and its kills.

The measured saving includes waiting for the previous write, which the model does not have.

| Rate | Policy | Kills | Saving: model / measured | Redone: model / measured | Redone, restart-aware model | Restart: model / measured | Total: model / measured | Total miss |
|---|---|---|---|---|---|---|---|---|
| 15 min | CheckFreq 1% | 38 | 565 / 574 | 3680 / 3084 (+19%) | 2871 (-7%) | 3427 / 3306 | 7672 / 6949 | +10% |
| 15 min | CheckFreq 5% | 38 | 586 / 633 | 3624 / 2900 (+25%) | 2836 (-2%) | 3427 / 3258 | 7637 / 6776 | +13% |
| 15 min | DCP every 64 steps | 38 | 303 / 886 | 16747 / 11814 (+42%) | 11361 (-4%) | 1521 / 1407 | 18571 / 14084 | +32% |
| 15 min | Failure-aware | 38 | 576 / 712 | 3649 / 2872 (+27%) | 2853 (-1%) | 3427 / 3246 | 7652 / 6831 | +12% |
| 60 min | CheckFreq 1% | 11 | 563 / 659 | 1066 / 1028 (+4%) | 1066 (+4%) | 992 / 1140 | 2622 / 2744 | -4% |
| 60 min | CheckFreq 5% | 11 | 583 / 602 | 1050 / 1170 (-10%) | 1050 (-10%) | 992 / 1157 | 2625 / 2926 | -10% |
| 60 min | DCP every 64 steps | 11 | 305 / 1387 | 4864 / 5259 (-8%) | 4557 (-13%) | 440 / 553 | 5609 / 7186 | -22% |
| 60 min | Failure-aware | 11 | 549 / 569 | 1088 / 1065 (+2%) | 1088 (+2%) | 992 / 1151 | 2629 / 2778 | -5% |

## Expected lost time before the kills are drawn (seconds per hour)

The model's 3600 L(T) at the set failure rate, against the measured mean over seeds (95% interval). The interval is wide because a 2-hour run sees only a few kills.

| Rate | Policy | Model | Measured |
|---|---|---|---|
| 15 min | CheckFreq 1% | 804 | 696 [+341, +1051] |
| 15 min | CheckFreq 5% | 801 | 679 [+310, +1048] |
| 15 min | DCP every 64 steps | 1951 | 1412 [+781, +2043] |
| 15 min | Failure-aware | 802 | 684 [+343, +1026] |
| 60 min | CheckFreq 1% | 243 | 275 [-32, +582] |
| 60 min | CheckFreq 5% | 244 | 293 [-22, +607] |
| 60 min | DCP every 64 steps | 510 | 720 [+56, +1384] |
| 60 min | Failure-aware | 243 | 278 [-20, +576] |

Kills drawn: [10, 4, 10, 10, 4] at 15 minutes (8 expected per run), [1, 0, 3, 6, 1] at 60 minutes (2 expected).

## CheckFreq 1% first-stall fix: before (Week 12) and after (Week 13), one failure per hour

| Seed | Same kill times | First stall (s) | Longest interval, before / after (steps) | Redone, before / after (s) | Lost per hour, before / after |
|---|---|---|---|---|---|
| 1 | yes (1 kills) | 3.1 | 108 / 20 | 404 / 68 | 301 / 137 |
| 2 | yes (0 kills) | 4.9 | 79 / 19 | 0 / 0 | 60 / 51 |
| 3 | yes (3 kills) | 1.8 | 84 / 19 | 395 / 289 | 399 / 356 |

## Checks

- Accounting residual: under 13.7 s in all runs but one; the largest is -69.6 s (w13/seed5/m3600/checkfreq-0-01).
- Kills before the first step: 0.
- Redone steps that reproduced their logged loss: 6269 of 6269.
- Clean step time by seed: seed 1 4.09–4.37 s, seed 2 4.10–4.12 s, seed 3 4.13–4.16 s, seed 4 4.38–4.56 s, seed 5 4.43–4.77 s.
