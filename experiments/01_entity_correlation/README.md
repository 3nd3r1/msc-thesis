# 01 entity correlation

Go/no-go for [entity-aware cascades](../../docs/Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).

## Question

Do LLM predicate verdicts agree inside a group more than chance explains? Reviews of one
product should share a verdict more often than two random reviews do, because a good
product mostly gets positive reviews.

## What it measures

1. Within-entity agreement against a chance baseline, per predicate.
2. The same measure for embedding clusters, to check the entity key adds signal that
   embeddings do not already give for free.
3. The entity size distribution, since tiny entities cannot be sampled and voted on.
4. Agreement conditioned on star rating, to check the verdict is not just a restatement of
   a column already in the data.

## Data

`data/All_Beauty.jsonl.gz`, Amazon Reviews 2023 (McAuley Lab). Entity key is the product
ASIN, rows are reviews.

## Outcome

If agreement inside entities is close to chance, or embedding clusters match it, the
cascade idea loses its core and the direction needs rethinking.

## Files

- `run.py` labels a sample with the oracle and writes verdicts plus logprobs
- `results/` outputs
