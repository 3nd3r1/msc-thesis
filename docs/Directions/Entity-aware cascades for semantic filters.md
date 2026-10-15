# Entity-aware cascades for semantic filters

Partition rows by an entity key, like the product a review is about, and spend the oracle budget per entity instead of per row, settling an entity's remaining rows from the verdicts already confirmed inside it.

Set aside after [01](../../experiments/01_entity_correlation) came out a no-go on 2026-10-01.

## Angle

A cascade scores every row on its own text and treats rows as independent.
Rows belong to entities, like the reviews of one product, and verdicts should correlate inside an entity, so once a few reviews of a product are confirmed the rest are close to decided.

Mechanisms to try were per-entity sampling and voting, neighbour verdicts as a feature of the proxy, and per-entity calibration thresholds.

## Gap

CSV [@csv] and adaptive Two-Phase [@kimFastLLMBasedSemantic2026] already exploit similarity between rows, but they infer the groups from embeddings because they target corpora with no schema.
Dataframes and SQL already carry entity keys, grouping by them is exact and free, and I have not found a method that uses them.

## Research questions

1. How strongly do verdicts correlate inside entities, and how does that compare with embedding clusters?
2. How many oracle calls does entity-aware cascading save at a fixed accuracy target?
3. Do accuracy guarantees survive when evidence is shared inside an entity?
4. Can weak grouping be detected early enough that the method never loses to the baselines?

## Evaluation

Amazon Reviews 2023 All Beauty [@amazonReviews2023], labelled once in [01](../../experiments/01_entity_correlation) so later comparisons run offline.

- Metric: oracle calls at a fixed accuracy target
- Baselines: full oracle, LOTUS cascade [@lotus], CSV, Two-Phase
- Implementation: extends the LOTUS cascade

## Why it was set aside

01 answered question 1 and the answer was no.
Embedding clusters match or beat product identity, and the star rating explains most of what the product key carries.
02 then replayed CSV from a structured column instead of k-means, and only the two predicates the rating partly defines won.

The question of what information outside a row's own text helps moved to links between rows, in [Link-guided oracle ordering for semantic filters](Link-guided%20oracle%20ordering%20for%20semantic%20filters.md).

## References

- csv - CSV, clusters rows by embedding and propagates a sampled label
- kimFastLLMBasedSemantic2026 - adaptive Two-Phase, clustering then a trained proxy
- lotus - LOTUS, semantic operators and the cascade this extends
- amazonReviews2023 - Amazon Reviews 2023, McAuley Lab
