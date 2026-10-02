---
date: 2026-10-02
---
# Structure outside the row

One question sits behind several of the loose ideas.
What cheap information outside a row's own text tells a cascade where the positives are?

Cascades score each row on its text alone.
Product and rating were the first answer to this, and [01](../../experiments/01_entity_correlation) and [02](../../experiments/02_cascade_replay) measure them.
Below are three mechanisms, all the same question with a different shape of information.

Text plus relational columns plus a citation graph is multi-model data, which is what UDBMS works on, so the framing fits the group.

## Columns as partitions

Partition on a schema column and spend the budget per partition.
This is what 02 runs and where the numbers already are.
03 would pick the partition per predicate.

## Entity metadata as a prior

Ask the oracle once per product on its metadata instead of once per review.
"Could a review of this product plausibly report skin irritation", over the title, category and description.
That is 300 products times 7 predicates, about 2,100 calls, against the 29,645 the sample needed.
The answer becomes a prior over every review of that product, and the cascade spends its calls on the high prior products.

This only works as a prior.
Shampoo can cause scalp irritation, so the low prior stratum still gets a small random sample to estimate how many positives it hides.
That is how SUPG style recall guarantees work with proxy scores, worth reading before building it.

Cheapest of the three to test, since it needs no new review labels and Amazon Reviews 2023 ships item metadata for All_Beauty:

- one call per product per predicate on the metadata
- AUC of the product prior against the 01 labels, and the share of positives in the top 20% of rows
- in the 02 replay, calls at a 90% recall target for uniform sampling, csv and prior stratified sampling

## Links between rows

Papers about semantic operators cite other papers about semantic operators.
Once the oracle says yes on a row its neighbours move up the queue, so label propagation decides the call order rather than the partition.
This is the graph edge extension listed in [Entity-aware cascades for semantic filters](../Directions/Entity-aware%20cascades%20for%20semantic%20filters.md), moved up because product grouping carried so little in 01.

Check homophily before building anything, with the 01 method.
Label a sample and their neighbours and compare P(yes | a neighbour is yes) against the base rate.
ogbn-arxiv has about 169k CS papers with a citation graph and abstracts.
Citation graphs are known to be homophilous so expect a positive result, which is fine as long as the dataset is picked for having links and not for the outcome.

Papers also carry more structured columns than All Beauty does, venue, year and affiliation, so the same corpus would serve the column mechanism too.

[Tiktok Comments](Tiktok%20Comments.md) is this mechanism with the thread or the video as the link.

Reading: active search on graphs (Garnett et al.), label propagation (Zhu and Ghahramani).

## Scope

One mechanism carries the thesis, the other two are related work or future work.

## Next

1. Finish 02 with the fixed replay.
2. Run the prescreening test, since it is cheapest and answers a clear question.
3. Email Lu with the 01 result, the 02 result and this framing, and ask which one he would back.

The graph mechanism is the likeliest to interest him given the group, and also the most new work.
