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
