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

## Result

Fail. With the neighbour's text the 70B finds 2.1 points more of the human positives on average, where 10 passes, and its false positive rate drops 0.9 points.

From `results/recall.txt`:

| dataset | predicate     | rec row | rec ctx | fpr row | fpr ctx | nbr+ missed | nbr+ found |
| ------- | ------------- | ------- | ------- | ------- | ------- | ----------- | ---------- |
| cad     | affiliation   | 67%     | 63%     | 0%      | 0%      | 30%         | 25%        |
| cad     | counterspeech | 50%     | 53%     | 13%     | 13%     | 0%          | 0%         |
| cad     | identity      | 80%     | 77%     | 10%     | 20%     | 83%         | 21%        |
| cad     | person        | 80%     | 90%     | 13%     | 7%      | 67%         | 21%        |
| cad     | slur          | 80%     | 73%     | 13%     | 13%     | 0%          | 12%        |
| meld    | anger         | 47%     | 53%     | 7%      | 3%      | 25%         | 36%        |
| meld    | disgust       | 47%     | 60%     | 7%      | 7%      | 12%         | 14%        |
| meld    | fear          | 67%     | 63%     | 13%     | 17%     | 20%         | 25%        |
| meld    | joy           | 57%     | 63%     | 13%     | 7%      | 8%          | 53%        |
| meld    | sadness       | 33%     | 37%     | 7%      | 0%      | 20%         | 20%        |
| meld    | surprise      | 93%     | 90%     | 27%     | 27%     | 0%          | 4%         |

DeepInfra served Llama 3.3 70B Turbo for these calls, while the Cora and Stack labels in 03 were served by Llama 3.1 70B Turbo.

Closed.
