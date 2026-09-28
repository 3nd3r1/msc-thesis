# Entity-aware cascades for semantic filters

## Problem

A semantic filter runs an LLM predicate on every row, so a million rows means a million
calls. A cascade cuts that down. A cheap proxy scores every row, only uncertain rows go to
the oracle, and the thresholds are calibrated to an accuracy target.

Cascades score each row on its own text and treat rows as independent. Real rows belong to
entities, like reviews of one product. Verdicts correlate inside an entity, so once a few
reviews of a product are confirmed the rest are close to decided. Cascades pay full price
for that evidence.

## Gap

CSV [@csv] and adaptive Two-Phase [@kimFastLLMBasedSemantic2026]
already exploit similarity between rows, but they infer the groups from embeddings because
they target corpora with no schema. Dataframes and SQL already carry entity keys. Grouping
by them is exact and free, and no method uses them.

## Idea

Partition rows by an entity key and spend the oracle budget per entity instead of per row,
then settle an entity's remaining rows from the verdicts already confirmed inside it.

Mechanisms to try:

- per-entity sampling and voting
- neighbour verdicts as a feature of the proxy
- per-entity calibration thresholds

Embedding clustering stays as the fallback when there is no useful key. Scope is single
entity keys. Foreign keys and graph edges are an extension, not the scope.

## Research questions

1. How strongly do verdicts correlate inside entities, and how does that compare with
   embedding clusters?
2. How many oracle calls does entity-aware cascading save at a fixed accuracy target?
3. Do accuracy guarantees survive when evidence is shared inside an entity?
4. Can weak grouping be detected early enough that the method never loses to the baselines?

## Evaluation

Two or three datasets with entity keys, several predicates each. Each dataset is labelled
once with the oracle, recording logprobs, so later comparisons run offline.

- Metric: oracle calls at a fixed accuracy target
- Baselines: full oracle, LOTUS cascade, CSV, Two-Phase
- Implementation: extends the LOTUS cascade

## Risks

- Verdicts may correlate only weakly inside entities.
- Embedding clusters may already capture the same structure.
- Entity sizes are long tailed, so most entities are too small to sample and vote on.

Experiment 01 measures all three before the method is committed to.

## References

- csv - CSV, clusters rows by embedding and propagates a sampled label
- kimFastLLMBasedSemantic2026 - adaptive Two-Phase, clustering first, then a trained proxy
- lotus - LOTUS, semantic operators and the cascade this extends
- zimmererSemCEBCardinalityEstimation2026 - SemCEB, cardinality estimation benchmark for semantic operators
- xuBridgingGapCardinality2026 - cardinality estimation for semantic queries on unstructured data
- urbanSelectivityEstimationSemantic2026 - selectivity estimation for semantic filters on images
- mangPLOPCostBasedPlacement2026 - PLOP, cost-based placement of semantic operators
- zhaoLarchLearnedQuery2026 - Larch, learned ordering of semantic predicates
- russoAbacusCostBasedOptimizer2026 - Abacus, cost-based optimizer for semantic operator systems
- qiSemaHighperformanceSystem2026 - Sema, system for LLM-based semantic query processing
- kumarasingheIPDBOptimizingSemantic2026 - iPDB, optimizing semantic SQL queries
- liuOptimizingLLMQueries2025 - prefix caching and dedup for LLM queries over relational data
