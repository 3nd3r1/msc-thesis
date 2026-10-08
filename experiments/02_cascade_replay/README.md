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

Start with the first split, sample ⌈0.005 * group size⌉ rows per group and propagate the majority if it agrees at the target rate.
A group that fails is split in two with k-means, and a group at or below its sample size goes to the oracle.
Settings are fixed for every variant, with no tuning on the labels.

## Output

Oracle calls per predicate and variant at accuracy targets of 90, 95 and 99%, and at a recall target of 90%.
The recall target keeps the rare predicates meaningful, since answering no everywhere meets the accuracy targets.

Then the best variant per predicate in hindsight and its saving over csv.
That is the upper bound for any method that picks the first split per predicate.

## Reading it

A selector is worth building in 03 only if the best variant saves clearly over csv on several predicates.

From 01 I expect rating to win on the positive control and on "doesn't work as advertised", and ties elsewhere.

The sample has 4,235 rows from products with 5+ reviews, so this is a feasibility check, not a final number.

## Conclusion

Half answered.
Rating beats a k-means first split on the predicates coupled to the star rating, and the sample rule decides whether anything else can be measured at all.

Under csv's own sample rule, with the floor of 100 rows that the paper uses, rating needs 32% of the calls against csv's 72% on "doesn't work as advertised" at a 90% target, and 25% against 94% on the positive control.
That holds at all three targets, at an F1 within 0.02 of csv's.
On the other five predicates it is a wash or slightly worse, and csv is cheaper on the two rarest.
So the expectation from 01 held, and two predicates out of seven is not the "several" that would make a selector obviously worth building.

The entity key could not be tested under that rule.
csv samples 100 rows per group whenever 0.005N is below 100, and no product in the sample has more than 20 reviews, so product and product+rating go straight to full oracle at 100% of calls.
Re-sampling does not rescue it.
All Beauty has 744 products with 100 or more reviews, holding 163k rows, so even a sample built only from those leaves the per-group sample at about half the group, which caps the saving on a product first split near 50% by the rule alone.

Dropping the floor breaks the procedure the other way, which is what the current results show.
Any group of 200 rows or fewer then samples a single row, a single row agrees with itself at any target, so propagation is unconditional and the target does nothing.
Product spends 7.1% of calls at every target and reaches 68.0% accuracy on the positive control, below the 71.4% that answering no everywhere gets.

The best evidence on the entity key is still the first run, which swept k and the agreement threshold per predicate.
Those numbers are optimistic, since the configuration is picked on the same rows it is scored on, so read them as an upper bound.
Even as an upper bound product loses to embedding clusters at a 95% target on five of seven predicates, and ties on irritation and sensitive skin where both variants just answer no.

No rule tried clears the recall target.
Finding 90% of the yes rows costs full oracle on irritation, sensitive skin and gift under every one of them.

03 needs matched call budgets before a selector is worth building.
With a fixed sample per group the cost is set by the number of groups, 4 for csv against 300 for product, so granularity and grouping quality cannot be told apart.
Give every variant the same budget and compare accuracy and F1 along a swept budget instead.

Numbers above come from three runs, all in git:

- `7f7d406` swept k and the threshold, four groupings including embeddings, tuned per predicate
- `1daf8a7` csv's rule with the 100-row floor
- `ded4072` csv's rule with no floor, which is what `results/replay.txt` holds now

The 744 product count is not in any results file, it comes straight off `data/All_Beauty.jsonl.gz`.
