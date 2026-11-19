# 06 TAR links

Under the same recall guarantee, does ordering the oracle by links return the verified positives in fewer oracle calls than the other orders and than BARGAIN?

## Hypothesis

Locked on 2026-11-19, before the run.

On CAD and MELD, QBCB over the link order costs less than QBCB over the proxy, kNN and rewired orders, and less than BARGAIN plus checking at its best budget.

## Pass rule

Over the predicates of `cad` and `meld`, the mean of links minus each of `bargain best`, `proxy`, `knn` and `rewired` is below 0.
Every QBCB cell has to reach the recall target in at least 88% of runs, or the result is invalid and the bug is found before anything else.
`cora` and `stack` are the control, where links did not help in 03, and are not part of the rule.
If it fails it is reported as failing and no other variant is tried.

`playground/tar_screen` ran these orders with k of 22, 40 and 80 on CAD and MELD before this lock, so this is a confirmation with k fixed at the middle value, not a blind test.

## Method

The query is the verified one, return the positives with every returned row confirmed by the oracle, with P(recall ≥ 0.9) ≥ 0.9.
Cost is the distinct oracle calls as a share of the rows.
Replay on the labels and 8B proxy scores of [03](../03_link_guided_search), so no new calls, on the pool with junk and repeated texts dropped.

QBCB, Lewis, Yang and Frieder, CIKM 2021, arXiv 2108.12746.
Sample rows uniformly until k positives are found and keep them hidden from the order.
Screen in the order until r of them are screened, with r the smallest value where P(Bin(k, 0.9) ≥ r) ≤ 0.1.
The order never sees which positives were sampled, so the guarantee holds for an adaptive order.

The four orders take the 8B p(yes) and put any neighbour of a confirmed positive above every other row, over:

- proxy, no graph
- knn, the top k cosine neighbours, k matched to the mean link degree
- rewired, the links with every row's degree kept and the ends shuffled
- links, the dataset's own edges, CAD reply edges and MELD previous line

BARGAIN plus checking runs BARGAIN's recall query, Zeighami et al. 2025, on the same p(yes), then asks the oracle about every returned row it has not asked.
Checking only removes negatives, so its guarantee holds.
It is reported at the paper's budget of 400 and at the lowest cost over a grid of budgets, picked in hindsight, which favours BARGAIN.

## Run

`python run.py` writes `results/costs.txt` with the table and the verdict.

## Result

Pass. On CAD and MELD, QBCB over the links costs less than every other arm, by 0.98 to 6.03 points of the rows.

Links minus each arm, mean over predicates, in points of the rows, from `results/costs.txt`:

| dataset | bargain best | proxy | knn   | rewired |
| ------- | ------------ | ----- | ----- | ------- |
| cad     | -6.03        | -4.55 | -0.98 | -4.84   |
| meld    | -1.95        | -1.98 | -1.86 | -2.96   |
| cora    | -37.44       | +0.51 | +1.93 | -4.21   |
| stack   | -18.64       | +2.38 | +2.37 | -4.62   |

Closed.
