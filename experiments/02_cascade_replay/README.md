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

Start with the first split, sample a share of each group and propagate the majority if it agrees at
the target rate.
A group that fails is split in two with k-means, and a group at or below its sample size goes to the
oracle.
Settings are fixed for every variant, with no tuning on the labels.

## Output

Oracle calls per predicate and variant at accuracy targets of 90, 95 and 99%, and at a recall target of 90%.
The recall target keeps the rare predicates meaningful, since answering no everywhere meets the accuracy targets.

Then the best variant per predicate in hindsight and its saving over csv.
That is the upper bound for any method that picks the first split per predicate.

## Reading it

A selector is worth building in 03 only if the best variant saves clearly over csv on several predicates.

## Result

BLUF: rating beats a k-means first split on the two predicates the star rating partly defines, and the sample rule decides everything else.

Under csv's own sample rule with its floor of 100 rows, at a 90% target:

| predicate    | csv calls | rating calls |
| ------------ | --------- | ------------ |
| doesn't work | 72%       | 32%          |
| positive     | 94%       | 25%          |

That holds at all three targets, at an F1 within 0.02 of csv's.
On the other five predicates it is a wash or slightly worse, and csv is cheaper on the two rarest.
Two predicates out of seven is not the "several" that would make a selector obviously worth building.

The entity key could not be tested under that rule.
No product in the sample is large enough to sample under the floor, so product and product+rating go
straight to full oracle at 100% of calls.
All Beauty has 744 products with 100 or more reviews, holding 163k rows, so even a sample built only
from those leaves the per group sample at about half the group.
That caps the saving on a product first split near 50% by the rule alone.

Dropping the floor breaks the procedure the other way, which is what `results/replay.txt` holds now.
Any small group then samples a single row, a single row agrees with itself at any target, so
propagation is unconditional and the target does nothing.

No rule tried clears the recall target.
Finding 90% of the yes rows costs full oracle on irritation, sensitive skin and gift under every one of them.

## Conclusion

Half answered.
With a fixed sample per group the cost is set by the number of groups, so granularity and grouping
quality cannot be told apart.
Give every variant the same budget and compare accuracy and F1 along a swept budget instead.
