# 01 entity correlation

Go/no-go for [entity-aware cascades](../../docs/Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).

## Question

Do LLM predicate verdicts agree inside a group more than chance explains?
If agreement inside entities sits near chance, or embedding clusters match it, the direction is dead.

## Data

- Amazon Reviews 2023 (McAuley Lab), category All_Beauty
- Downloaded from the hub on first run and cached in `data/hf`

## Models

Llama 3.1 70B on DeepInfra as the oracle, the same family as the LOTUS cascade, so the later
comparison is against their setup rather than a reimplementation of it.
Needs `DEEPINFRA_API_KEY`, see `.env.example` at the repo root.

DeepInfra logprobs are unreliable.
Support is per model and undocumented, no model returns top_logprobs, and logprobs go missing with
no error on the same request every time.
Those rows keep the verdict parsed from the reply text and store no p.
Verdicts are all the steps below need.
The LOTUS cascade baseline does need top_logprobs, so it needs another provider or a rented GPU
running vLLM.

## Steps

Each step is `python run.py <step>` and writes `results/<step>.txt`.
Steps run in order, each one needs the output of the last.

### 1. Sizes

Measure the sizes of the entity groups for both candidate keys, the product and the reviewer.

#### Result

BLUF: the product works as an entity key and the reviewer does not.

Share of rows in groups of at least n, from `results/sizes.txt`:

| at least | product | reviewer |
| -------- | ------- | -------- |
| 2        | 93.2%   | 16.8%    |
| 5        | 79.1%   | 2.1%     |
| 10       | 65.9%   | 1.0%     |
| 50       | 34.5%   | 0.2%     |

### 2. Sample

Take 100 products each from the 5-9, 10-49 and 50+ buckets, up to 20 reviews per product.
Rows go to `labels/all_beauty/sample.jsonl`.

#### Result

BLUF: the sample cannot measure the reviewer key, 29 of its 4,202 reviewers have more than one review.

Deferred.
The author level predicates stay in as a contrast class, they should group weakly by product if the
metric works.
Numbers in `results/sample.txt`.

### 3. Label

Label the sample with the LOTUS sem_filter prompt.

Store the verdict and the chosen token's probability as two fields, and derive p(yes) from them.
Without top_logprobs the leftover probability all goes to the opposite verdict, which breaks below
p = 0.5, so the step reports how often p lands there.

#### Predicates

| predicate                                        | level   |
| ------------------------------------------------ | ------- |
| reports skin irritation or an allergic reaction  | product |
| says the product doesn't work as advertised      | product |
| suspects the product is fake                     | product |
| mentions having sensitive skin                   | author  |
| bought it as a gift                              | author  |
| mentions another person (partner, child, friend) | row     |
| the review is positive                           | control |

#### Result

A hand read on 2026-09-29 found 5 to 10 false positives per 100 reviews on the rare predicates, all
on reviews with almost no text like 'Flimsy' or 'Apricot lotion'.
Nothing saved that count, so take it as an impression.
Yes rates in `results/labels.txt`.

### 4. Compare

For each predicate, measure how homogeneous each grouping is:

- product ID
- user ID (where the data allows it)
- embedding clusters, k-means with the same number and sizes of clusters as the product groups
- shuffled groups with the same sizes, as the chance baseline
- product again, with the chance term shuffled within rating strata

The metric is pairwise agreement corrected for chance, which handles the "a group of two agrees half
the time anyway" problem.
Also label 3 random rows per product, propagate the majority to the rest, and report the accuracy.

#### Result

BLUF: embedding clusters match or beat product identity, and the rating explains most of what product carries.

Agreement above chance, from `results/compare.txt`:

| predicate        | product | cluster | rating | product within rating |
| ---------------- | ------- | ------- | ------ | --------------------- |
| irritation       | 0.04    | -0.06   | 0.51   | 0.03                  |
| doesn't work     | 0.16    | 0.29    | 0.78   | 0.07                  |
| fake             | 0.07    | 0.16    | 0.40   | 0.06                  |
| sensitive skin   | 0.10    | -0.20   | -0.05  | 0.10                  |
| gift             | 0.13    | 0.09    | -0.12  | 0.13                  |
| another person   | 0.10    | 0.12    | -0.12  | 0.10                  |
| positive         | 0.16    | 0.39    | 0.89   | 0.01                  |

Propagating the majority of 3 sampled reviews per product loses to a constant guess on every
predicate, by 0.3 to 2.5 points.

## Conclusion

No-go for entity-aware cascades as written.
Experiment 02 measures oracle calls at an accuracy target instead.
