---
date: 2026-10-21
---

# Learning from every verdict

Every experiment so far looks for a cheap signal outside the row that beats reading the row's text.
The text keeps winning, because the oracle decides from the text and embeddings and the proxy read the same text.
See the Oct 15 log.

## What a recommender does that a cascade does not

TikTok is not good at understanding videos.
It does three other things.

- It updates its model after every swipe, not once.
- It uses other users, people who liked these five also liked this one.
- It shows things it is unsure about on purpose, because those answers teach it the most.

Map it over with the predicate as the user, a row as a video and an oracle verdict as a swipe.
The current orders do none of the three.
The proxy scores each row once and never changes, and nothing carries over between queries.

## Trick 1, retrain on the verdicts

After each batch of oracle calls fit a small classifier on the verdicts so far, logistic regression on the embeddings with the proxy score as a feature, and rerank the rest.
Embedding order means close to the positives found so far, which is mostly topic.
A trained classifier learns which direction in embedding space matters for this predicate.
The non topical predicates are where embeddings do worst, proof, biology and ownstudy, so that is where it should show.

Positive and negative neighbour counts go in as two more features.
The classifier then learns per predicate whether links matter, which is the fallback from question 3 without a hand made check.

Cheap to test, one new order in 03 replayed on the existing labels.

This is continuous active learning from e-discovery and systematic review screening, with an LLM as the reviewer.
SemViews already does it for `sem_filter`, see below.

## Trick 2, remember other predicates

Each past predicate is a user.
Its verdicts on the same rows are features no embedding has.
The Stack and Cora label matrices already allow a leave one predicate out test.

## Motivation

Find every X in a large pile, with a recall target and an expensive judge, is how systematic reviews in medicine work.
A missed trial skews the meta analysis, so recall is the number that matters, and LLM screening is arriving there now.
Legal discovery has the same shape.

Pitch: semantic operators made LLM screening declarative but took their ordering from video analytics.
The fields that have done recall targeted screening for a decade learn from every verdict and know when to stop.
The thesis brings that loop into `sem_filter`, and the graph becomes one optional feature.

## SemViews covers both tricks

Read on 2026-10-21, Hassanzadeh and Subramanian, IBM, arXiv 2610.12768, posted Oct 8 2026.
Its CertifiedFilter fits a logistic score on an embedding projection, the helper score and past predicates' labels, then labels the rows closest to 0.5, refits and replans in rounds.
That is trick 1 and trick 2 in one operator, built as a LOTUS optimizer.
It then certifies precision and recall with fresh stratified samples, so a useless feature costs calls but never accuracy.
It leaves open links between rows, joins and ranking, real tables of millions of rows, and it cites nothing from technology assisted review.
Its certification floor means a rare predicate on a table of a few thousand rows costs most of exact evaluation whatever the score, which the calls to recall metric in 03 does not see.

## Risks

- The proxy is already a high bar, step 4 of [03](../../experiments/03_link_guided_search). A tie on topical predicates and a win on the non topical ones would still be a result.

Related: [Structure outside the row](Structure%20outside%20the%20row.md), [Oracle quality](Oracle%20quality.md), [Link-guided oracle ordering for semantic filters](../Directions/Link-guided%20oracle%20ordering%20for%20semantic%20filters.md).
