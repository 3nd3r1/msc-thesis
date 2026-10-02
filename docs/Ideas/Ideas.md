---
date: 2026-10-01
---
# Ideas

Loose ideas from 2026-10-01, nothing tried yet.

## Neighbour aware cascade

The current direction spends the oracle budget inside one entity key.
Edges between rows would do the same job without needing a key.
Once the oracle confirms a row, its neighbours start with a higher prior before the proxy scores them.

Papers are the case that made me think of it.
A paper that talks about semantic operators tends to cite other papers that do, so one confirmed hit raises the odds on everything it cites.
Same shape as an entity key, except the group is a neighbourhood and a row can sit in several.

This is the graph edge extension listed in [Entity-aware cascades for semantic filters](../Directions/Entity-aware%20cascades%20for%20semantic%20filters.md).
Moving it up because product grouping only reached 0.06 to 0.16 above chance in 01, so a different edge type is worth trying.

## Papers as the dataset

Finding papers that use semantic operators is a query I run by hand every week.
The corpus is public and the predicate is one I can check myself, which is more than I can say for the beauty reviews.

Predicate: the paper proposes or uses a semantic operator.
Grouping keys available: venue, year, affiliation, topic, citation neighbours, embeddings.
That is more groupings than All Beauty has, so it would give the 02 selector something to actually choose between.

Labels would have to be made by hand, a few hundred at most.

## Tiktok replies

[Tiktok Comments](Tiktok%20Comments.md) is about reading the first few comments instead of all of them.
Replies are the neighbour version of the same thing.
A thread under one comment agrees with itself more than two random comments do, so propagation should work one level down too.

## Semantic filter instead of vector search in RAG

RAG retrieves by vector similarity, which tells you what a chunk looks like and not whether it answers the question.
A semantic filter over the candidates answers the second one.
It only pays off if a cascade gets the per-row cost low enough, which is the number 02 is measuring anyway.

LLM rerankers already do something close, so the open part is what a filter with an accuracy target adds over a reranker with a score cutoff.
