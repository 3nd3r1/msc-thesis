# 02 cascade replay

Does starting [@csv] from a structured column instead of k-means save oracle calls, and on which predicates?

## Method

Replays cascades offline on the labels from [01](../01_entity_correlation), so it makes no new oracle calls.
This replaces the agreement metric from 01 with the number the cascade papers report, oracle calls at a target.

Every variant runs the same [@csv] procedure with a different first split:

- k-means on embeddings, which is csv itself and the baseline
- rating
- product
- product and rating together

Start with the first split, sample max(⌈0.005N⌉, 100) rows per group and propagate the majority if it agrees at the target rate.
A group that fails is split in two with k-means, and a group too small to split goes to the oracle.
Settings are fixed for every variant, with no tuning on the labels.

## Output

Oracle calls per predicate and variant at accuracy targets of 90, 95 and 99%, and at a recall target of 90%.
The recall target keeps the rare predicates meaningful, since answering no everywhere meets the accuracy targets.

Then the best variant per predicate in hindsight and its saving over csv.
That is the upper bound for any method that picks the first split per predicate.

## Reading it

If the best variant saves clearly over csv on several predicates, a selector is worth building in 03.
If csv wins or ties nearly everywhere, structured first splits do not help on this data.

From 01 I expect rating to win on the positive control and on "doesn't work as advertised", and ties elsewhere.

The sample has 4,235 rows from products with 5+ reviews, so this is a feasibility check, not a final number.

## Conclusion

Open.
