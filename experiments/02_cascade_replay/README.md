# 02 cascade replay

Replay cascade algorithms offline against the labels from [01](../01_entity_correlation).

## Idea

Every row in the 01 sample has an oracle label for all 7 predicates, which is enough to run a cascade without calling anything.
When the algorithm asks for an oracle call, look up the stored label and add one to the counter.
When it guesses, check the guess against the stored label.
A run takes seconds, so it can be repeated over many seeds, and it reports the two numbers the cascade papers report, oracle calls used and accuracy of the final labels.

It also replaces the metric from 01.
Corrected pairwise agreement is biased downward whenever a predicate is enriched in the groups that hold most of the pairs, which happened on both the embedding clusters and the rating grouping.
Oracle calls at a target accuracy has no such problem.

## What it runs

The [@csv] procedure.
Split rows into groups, sample some per group, call the oracle on the sample, and if the sample agrees at least at the target rate, give the whole group the majority label.
Otherwise split the group and repeat, or call the oracle on every row in it.

Only the grouping changes between runs:

- embedding clusters, which is csv itself and the baseline
- product
- rating
- product and rating together
- a selector that tries a few of these on early oracle labels and picks the purest per predicate

## Output

Oracle calls needed to reach a 90% target, per predicate and per grouping.
Report F1 next to accuracy so that answering no everywhere cannot win on the rare predicates.

## Hypotheses

- H1: for some predicates a structured grouping needs fewer oracle calls than embedding clusters
- H2: the best grouping for a predicate can be picked from a small number of early labels
- H3: picking costs little, so the method beats csv where structure helps and loses only a little where it does not

From 01 I would expect rating to win H1 on the positive control and on "doesn't work as advertised", with embeddings winning or tying elsewhere.
If H1 holds for enough predicates then the selector is worth building and H2 and H3 go in the same code.
If embeddings win everywhere that is an afternoon spent instead of three more datasets labelled.

## Conclusion

Open. Nothing written yet.
