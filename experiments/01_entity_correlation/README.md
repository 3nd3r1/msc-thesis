# 01 entity correlation

Go/no-go for [entity-aware cascades](../../docs/Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).

## Question

Do LLM predicate verdicts agree inside a group more than chance explains?
Reviews of one product should share a verdict more often than two random reviews do, because a good
product mostly gets positive reviews.

A negative answer sinks the direction. If agreement inside entities is close to chance, or
embedding clusters match it, the cascade idea loses its core.

## Data

- Amazon Reviews 2023 (McAuley Lab), category All_Beauty
- Downloaded from the hub on first run and cached in `data/hf`

## Models

Llama 3.1 70B as the oracle, Llama 3.1 8B as the proxy, both on DeepInfra.
Same family as the LOTUS cascade, so the later comparison is against their setup rather than a
reimplementation of it.
Which models return logprobs on DeepInfra is per model and undocumented, `smoketest.py` checks it.
Needs `DEEPINFRA_API_KEY`, see `.env.example` at the repo root.

## Steps

Each step is `python run.py <step>` and writes `results/<step>.txt`.
Steps run in order, each one needs the output of the last.

### 1. Sizes

Measure the sizes of the entity groups.
A cascade only saves calls on groups large enough to sample a few rows and settle the rest, so
what matters is not how many groups are large but what share of rows lives in them.
We check both candidate keys, the product and the reviewer.

#### Result

The product works as an entity key and the reviewer does not.
79% of reviews are in products with at least 5 reviews, against 2% for reviewers.
Sizes are long tailed either way, 42% of products have a single review, but those products hold
only 7% of the rows.

That gives an upper bound for savings on this dataset.
Suppose you sample k = 3 reviews per product for products with at least 5 reviews, propagate to
the rest and run the oracle on everything in smaller products:

- 146,820 rows in products under 5 get full oracle calls
- 27,533 products * 3 = 82,599 sampled calls
- about 229k calls against 701k, about 67% saving

This assumes propagation is always right, which is what the next steps test.

### 2. Sample

Labeling all reviews is unnecessary.
Take 100 products each from the 5-9, 10-49 and 50+ buckets, up to 20 reviews per product.
Rows go to `results/sample.jsonl`.

#### Result

4,235 reviews over 300 products, seed 0.
Only 29 reviewers have more than one review in the sample, so the user ID grouping cannot be
measured from it.
Deferred. The author-level predicates stay in as a contrast class, they should group weakly by
product if the metric works.

### 3. Label

Label the sample with these 7 predicates:

| Predicate                                        | Level   |
| ------------------------------------------------ | ------- |
| reports skin irritation or an allergic reaction  | product |
| says the product doesn't work as advertised      | product |
| suspects the product is fake                     | product |
| mentions having sensitive skin                   | author  |
| bought it as a gift                              | author  |
| mentions another person (partner, child, friend) | row     |
| the review is positive                           | control |

Use the LOTUS sem_filter prompt.
It still gives 5 to 10 false positives per 100 reviews on the rare predicates, all on reviews
with almost no text like 'Flimsy' or 'Apricot lotion'. Known limitation, noise attenuates
agreement so it errs toward a null result.

Store the verdict and the chosen token's probability as two fields, and derive p(yes) from them:
p if the verdict is yes, 1 - p if no.
No model on DeepInfra returns top_logprobs, so the alternatives are not available and this is the
only route to p(yes). We only learn how likely the model thought its own answer was, never the alternatives, so we
treat all the leftover probability as the opposite answer.
That is close enough when the model is sure.
It breaks when p drops below 0.5, because then the model answers no while 1 - p comes out above
0.5 and reads as yes.
The step reports how often p lands below 0.5.

### 4. Compare

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

## Conclusion

Open. Steps 3 and 4 are not run yet.
Sizes says the savings are reachable on this dataset, but nothing has measured agreement.
