# 08 sensitivity

Does the 07 result hold when the k of QBCB and the recall target change?

## Hypothesis

Locked on 2026-11-26, before the run.

On CAD and MELD, QBCB over CAL with link features costs less than QBCB over CAL alone, CAL with kNN features and CAL with rewired link features at each of these settings:

- k 22, recall target 0.9
- k 80, recall target 0.9
- k 40, recall target 0.8
- k 80, recall target 0.95

## Pass rule

At each setting, over the predicates of `cad` and `meld`, take the mean of `cal+links` minus each of `cal`, `cal+knn` and `cal+rewired`.
A setting passes if the whole 95% cluster bootstrap interval of that mean is below 0 for all three arms on both datasets.
It passes if all four settings pass, and the verdict of each setting is reported.
Every CAL cell on the full data at every setting has to reach the recall target in at least 88% of runs, or the result is invalid and the bug is found before anything else.
k 40 with target 0.9 is the 07 setting, rerun as a check that the full data reproduces 07, and is not part of the rule.
`bargain` is reported next to them and is not part of the rule.
`cora` and `stack` are the control, run on the full data only, and are not part of the rule.
If it fails it is reported as failing and no other variant is tried.

Seen before the lock:

- [07](../07_cal_links/README.md) passed at k 40 and target 0.9.
- `playground/tar_screen` ran the 06 orders, not CAL, at k 22, 40 and 80 on CAD and MELD.
- Target 0.95 at δ 0.1 needs k of at least 45, since below that 0.95^k > 0.1 even when every hidden positive is screened, so it runs at k 80.
- At k 80 the uniform sample is a larger share of the cost on the rare predicates, and a predicate with fewer than k positives in the pool or a draw costs every arm 100%, both of which push the gaps toward 0.

## Method

As in [07](../07_cal_links/README.md), with the same arms, data, cluster bootstrap and draws.
A CAL order does not depend on k or the target, so each draw fits the four orders once and runs QBCB at every setting on them.
r is the smallest value where P(Bin(k, target) ≥ r) ≤ δ.

BARGAIN plus checking runs at each target at the budget with the lowest cost over the grid of `playground/bargain_screen/wide.py`, picked in hindsight on the full data, which favours BARGAIN.
At target 0.9 this is the same pick as in 07.

## Run

`python run.py` writes the BARGAIN budgets to `results/budgets.jsonl`, every finished draw to `results/draws.jsonl` and the tables, the gaps with their intervals and the verdict to `results/costs.txt`.
A stopped run resumes from the two jsonl files.

## Result

Fail. At k 40 with target 0.8 the interval of `cal+links` against `cal+knn` and `cal+rewired` on CAD reaches above 0, the other three settings pass.

Cal+links minus each arm, mean over predicates, in points of the rows, full data then the 95% cluster bootstrap interval, from `results/costs.txt`:

| k   | target | dataset | cal                  | cal+knn              | cal+rewired          | bargain                |
| --- | ------ | ------- | -------------------- | -------------------- | -------------------- | ---------------------- |
| 22  | 0.9    | cad     | -2.43 [-3.17, -1.03] | -2.51 [-3.00, -0.38] | -2.22 [-2.93, -0.85] | -11.29 [-14.32, -7.75] |
| 22  | 0.9    | meld    | -2.78 [-3.66, -1.20] | -2.55 [-3.20, -0.85] | -3.01 [-3.00, -0.76] | -2.82 [-4.91, -0.60]   |
| 22  | 0.9    | cora    | +0.37                | +0.18                | -0.75                | -15.80                 |
| 22  | 0.9    | stack   | -0.04                | -0.15                | -0.11                | -8.36                  |
| 80  | 0.9    | cad     | -2.29 [-2.93, -1.18] | -1.99 [-2.32, -0.52] | -2.21 [-2.67, -0.77] | -13.86 [-15.98, -9.98] |
| 80  | 0.9    | meld    | -3.58 [-4.49, -1.51] | -3.34 [-3.91, -1.00] | -3.93 [-3.78, -0.97] | -10.01 [-11.46, -7.20] |
| 80  | 0.9    | cora    | +0.21                | +0.16                | +0.22                | +31.75                 |
| 80  | 0.9    | stack   | +0.02                | +0.09                | -0.11                | -1.31                  |
| 40  | 0.8    | cad     | -1.35 [-2.26, -0.53] | -0.75 [-1.72, +0.29] | -1.40 [-1.93, +0.04] | -15.53 [-17.84, -11.89] |
| 40  | 0.8    | meld    | -2.38 [-3.84, -1.50] | -2.47 [-3.29, -1.16] | -3.01 [-3.24, -0.85] | -9.31 [-11.04, -6.47]  |
| 40  | 0.8    | cora    | +0.18                | +0.17                | +0.10                | +12.25                 |
| 40  | 0.8    | stack   | +0.04                | +0.17                | -0.06                | -3.27                  |
| 80  | 0.95   | cad     | -2.48 [-3.54, -1.26] | -2.40 [-3.15, -0.92] | -2.43 [-3.32, -1.04] | -10.56 [-13.20, -8.44] |
| 80  | 0.95   | meld    | -2.33 [-3.29, -0.73] | -2.07 [-2.90, -0.56] | -2.43 [-2.88, -0.43] | -4.39 [-6.30, -1.98]   |
| 80  | 0.95   | cora    | +0.19                | +0.01                | +0.02                | -3.53                  |
| 80  | 0.95   | stack   | +0.04                | -0.04                | -0.13                | -9.09                  |

Closed.
