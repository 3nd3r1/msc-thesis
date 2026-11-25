# 07 CAL links

Under the same recall guarantee, does a continuous active learning order with counts of confirmed neighbours along the links return the verified positives in fewer oracle calls than the same order with no neighbours, with a cosine kNN or with the links rewired?

## Hypothesis

Locked on 2026-11-24, before the run.

On CAD and MELD, QBCB over CAL with link features costs less than QBCB over CAL alone, CAL with kNN features and CAL with rewired link features.

## Pass rule

Over the predicates of `cad` and `meld`, take the mean of `cal+links` minus each of `cal`, `cal+knn` and `cal+rewired`.
It passes if the whole 95% cluster bootstrap interval of that mean is below 0 for all three arms on both datasets.
Every CAL cell on the full data has to reach the recall target in at least 88% of runs, or the result is invalid and the bug is found before anything else.
`bargain` is reported next to them and is not part of the rule.
`cora` and `stack` are the control, run on the full data only, and are not part of the rule.
If it fails it is reported as failing and no other variant is tried.

Seen before the lock:

- `playground/cal_screen` ran `cal` and `cal+links` on the full data of all four datasets, so that gap is a confirmation.
- `playground/bootstrap_screen` found the 06 link order ties a cosine kNN on CAD, so `cal+knn` on CAD is the cell most likely to fail.
- `cal+knn` and `cal+rewired` have not been run.

## Method

The query, the cost, QBCB with k 40 and BARGAIN plus checking are as in [06](../06_tar_links/README.md).
Replay on the labels and 8B proxy scores of [03](../03_link_guided_search), so no new calls, on the pool with junk and repeated texts dropped.

CAL follows the 8B p(yes) until both classes are seen, then refits a logistic regression on the proxy logit and the embedding on the verdicts every 1% of the rows and orders the rest by it.
These settings come from `playground/cal_screen` and were not tuned.
The four arms add the counts of confirmed positive and negative neighbours as two features, over:

- cal, no graph
- cal+knn, the top k cosine neighbours, k matched to the mean link degree
- cal+rewired, the links with every row's degree kept and the ends shuffled
- cal+links, the dataset's own edges, CAD reply edges and MELD previous line

BARGAIN plus checking runs at the budget of its wide best per predicate from `playground/bargain_screen/results/wide.txt`, picked in hindsight on the full data, which favours BARGAIN.

The interval is a cluster bootstrap over the connected components of the links, threads on CAD and dialogues on MELD, drawn with replacement.
Each draw rebuilds the kNN among the drawn rows without pairs of copies of one row, redoes the rewiring, refits every CAL order and reruns QBCB and BARGAIN.
The interval runs from the 2.5th to the 97.5th percentile of the draws.

## Run

`python run.py` writes `results/costs.txt` with the table, the gaps with their intervals and the verdict.
Every finished draw goes to `results/draws.jsonl`, so a stopped run resumes.

## Result

Pass. On CAD and MELD, QBCB over CAL with link features costs less than CAL alone, CAL with kNN features and CAL with rewired link features, by 2.10 to 3.68 points of the rows.

Cal+links minus each arm, mean over predicates, in points of the rows, full data then the 95% cluster bootstrap interval, from `results/costs.txt`:

| dataset | cal                  | cal+knn              | cal+rewired          | bargain                |
| ------- | -------------------- | -------------------- | -------------------- | ---------------------- |
| cad     | -2.29 [-3.21, -1.09] | -2.22 [-2.90, -0.38] | -2.10 [-3.03, -0.73] | -13.25 [-16.60, -9.75] |
| meld    | -3.40 [-4.08, -1.32] | -3.15 [-3.72, -0.98] | -3.68 [-3.55, -0.81] | -5.92 [-7.88, -3.38]   |
| cora    | +0.62                | +0.15                | +0.19                | -2.02                  |
| stack   | +0.04                | +0.03                | -0.13                | -8.15                  |

Closed.
