# 01 entity correlation

Go/no-go for [entity-aware cascades](../../docs/Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).

## Question

Do LLM predicate verdicts agree inside a group more than chance explains?
Reviews of one product should share a verdict more often than two random reviews do, because a good
product mostly gets positive reviews.

## Data

- Amazon Reviews 2023 (McAuley Lab), category All_Beauty
- Downloaded from the hub on first run and cached in `data/hf`

## Outcome

If agreement inside entities is close to chance, or embedding clusters match it, the
cascade idea loses its core and the direction needs rethinking.

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

### Sample

Labeling all reviews is unnecessary.
Lets take 100 products each from the 5–10, 10–50 and 50+ buckets.
Take up to 20 reviews per product, which gives roughly 5k reviews.

### Label

Label the sample with these 7 predicates:

- "reports skin irritation or an allergic reaction", "says the product doesn't work as advertised", "suspects the product is fake" - Product-level
- "mentions having sensitive skin", "bought it as a gift" - Author-level
- "mentions another person (partner, child, friend)" - Row-level
- "the review is positive" - Control

Use open-weight model as the oracle.
Save the verdict and yes/no logprob.

### Compare

For each predicate, measure how homogeneous each grouping is:

- product ID
- user ID (where the data allows it)
- embedding clusters, using k-means on sentence embeddings with the same number and sizes of clusters as the product groups
- shuffled groups with the same sizes, as the chance baseline

For the metric, use pairwise agreement corrected for chance.
Compare the probability that two rows in the same group agree with the probability that two random rows agree.
This works like kappa or an intra-class correlation, and it handles the "a group of two agrees half the time anyway" problem.

For the rating check, compute the same thing only over pairs with the same star rating.
If agreement within products stays high among, say, 3-star reviews only, the signal isn't just the rating.

Also add one practical number: label k = 3 random rows per product, propagate the majority to the rest, and report the accuracy.
That turns "correlation" into "how many calls would this save at what accuracy".
