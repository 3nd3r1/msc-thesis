# 01 entity correlation

Go/no-go for [entity-aware cascades](../../docs/Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).

## Question

Do LLM predicate verdicts agree inside a group more than chance explains?
Reviews of one product should share a verdict more often than two random reviews do, because a good
product mostly gets positive reviews.

## Data

- Amazon Reviews 2023 (McAuley Lab), category All_Beauty
- Downloaded from the hub on first run and cached in `data/hf`

## Steps

### Sizes

First step is to measure the sizes of the entity groups.
A cascade only saves calls on groups large enough to sample a few rows and settle the rest, so
what matters is not how many groups are large but what share of rows lives in them.
We check both candidate keys, the product and the reviewer.

The product works as an entity key and the reviewer does not.
79% of reviews are in products with at least 5 reviews, against 2% for reviewers.
Sizes are long tailed either way, 42% of products have a single review, but those products hold
only 7% of the rows.

#### Verdict

We get a upperbound for savings on this dataset.
Suppose you sample k = 3 reviews per product for products with at least 5 reviews, propagate to the rest and run the oracle on everything in smaller products:

- 146,820 rows in products under 5 get full oracle calls
- 27,533 products * 3 = 82,599 sampled calls
- about 229k calls against 701k, about 67% saving.

This assumes propagation is always right, which is what the next steps test.

## Outcome

If agreement inside entities is close to chance, or embedding clusters match it, the
cascade idea loses its core and the direction needs rethinking.
