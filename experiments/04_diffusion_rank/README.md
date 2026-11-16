# 04 diffusion rank

Does diffusing the oracle's corrections to the proxy over links save oracle calls, once the proxy is a rank instead of the 8B's raw p?

## Hypothesis

Locked on 2026-11-16, before the run.

With the proxy's rank in place of its p, `proxy+diffuse` over links needs fewer oracle calls than the proxy alone on the edge sets where links help, and fewer than the same diffusion over a cosine kNN.

## Pass rule

At 90% recall, over the predicates of `cad`, `meld` and `meld_dialogue`:

- the mean of `proxy+diffuse` minus the proxy over links is below 0
- and it is below the same mean over the kNN

If it fails it is reported as failing and no other variant is tried.
`cora`, `stack` and `stack_author` are the control, where links did not help in 03.

## Method

Replay on the labels and proxy scores of [03](../03_link_guided_search), so no new calls.
The proxy score of a row is its rank among the rows over the number of rows, ties sharing their mean rank.
`proxy+diffuse` scores a row by that rank plus the mean residual, verdict minus rank, of its neighbours the oracle has answered, one hop of the bias diffusion in Correct and Smooth.
The proxy order is the same under the rank as under p.

Each dataset runs over three graphs:

- links, the dataset's own edges
- knn, the top k cosine neighbours, k picked so it has about as many edges as the links
- both, the union

Only the pool with junk and repeated texts dropped.

## Run

`python run.py <dataset>` writes `results/<dataset>.txt`.
`python run.py verdict` applies the pass rule to those files and writes `results/verdict.txt`.

## Result
