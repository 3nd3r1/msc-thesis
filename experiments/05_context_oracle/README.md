# 05 context oracle

Do CAD and MELD labels need the neighbour's text, so that an LLM oracle reading the row alone misses the positives the links find?

In 03 links helped on CAD and MELD, labelled by people who saw the thread or the scene, and not on Cora and Stack, labelled by the 70B from the row.
So dataset and label source are mixed, see the 2026-11-17 log.

## Hypothesis

Locked on 2026-11-17, before the run.

Given the text along the link, the parent post on CAD and the line before on MELD, the 70B finds clearly more of the human positives than from the row alone, without saying yes much more on the negatives.

## Pass rule

Over the predicates of CAD and MELD:

- the mean of recall with the neighbour minus recall from the row alone is 10 points or more
- and the mean of the false positive rate with the neighbour minus from the row alone is 5 points or less

Recall and false positive rate are against the human labels.
If it fails it is reported as failing.

## Method

Per predicate a random sample, seed 0, of human positives and of human negatives, from the pool with junk and repeated texts dropped, among rows that have a neighbour along the link, a parent on CAD and a line before in the same dialogue on MELD.

Each sampled row is asked twice with the same prompt apart from the neighbour:

- alone, the row alone
- context, the neighbour's text above the row

The prompt is LOTUS sem_filter with the moderation framing from 03 on CAD, and plain LOTUS with "Always answer True or False." on MELD, since the 03 line alone prompt says there is no scene.

Also reported is the share of human positives whose neighbour is a positive for the same predicate, separately for those the row alone misses and those it finds.

Every human positive the two answers disagree on, or both miss, goes to `results/cases.txt` with the neighbour's text, to read case by case.

## Run

`python run.py label` asks the 70B and appends to `results/calls.jsonl`, resuming if stopped.
`python run.py report` writes `results/recall.txt` and `results/cases.txt`.
