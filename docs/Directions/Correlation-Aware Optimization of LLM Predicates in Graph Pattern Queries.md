# Correlation-Aware Optimization of LLM Predicates in Graph Pattern Queries

Use correlation between LLM verdicts on connected nodes to plan graph pattern queries that carry semantic predicates.

Set aside on 2026-09-24, the first direction I wrote.

## Angle

A pattern matching a million posts costs a million oracle calls.
Engines parallelise those calls but never reduce how many are made.
Verdicts on connected nodes correlate, the way misinformation clusters in friend groups, and graph statistics engines already keep, such as clustering coefficient and assortativity, would predict that correlation before any LLM call.

Sampling along the pattern is affordable here in a way it would not be in a normal database.
Semantic predicates have no free statistics, so the optimizer has to sample with the LLM anyway, and sampled verdicts can be cached and reused during execution.

Contributions were to measure verdict correlation along edge types and how often assuming independence picks the wrong plan, extend PLOP-style placement [@mangPLOPCostBasedPlacement2026] to fixed-shape patterns with degree-weighted sampling [@liWanderJoinOnline2016], and let neighbour verdicts escalate a node to the large model.

## Gap

Semantic operator optimizers estimate each predicate separately and then combine the estimates as if the predicates were independent [@zhaoLarchLearnedQuery2026].
SemCEB [@zimmererSemCEBCardinalityEstimation2026] benchmarks cardinality estimation but evaluates filters and joins in isolation, and cost modelling of LLM operators in query plans is named as open [@yanDBMSLLMIntegrationStrategies2025].
I found no measurement of correlation between semantic predicates across an edge or a join.

## Research questions

1. How strongly do LLM verdicts correlate along edges, and how often does that change the best plan?
2. How much do the cost model and the escalation save at a fixed accuracy target?
3. Which graph properties predict when each technique helps?

## Why it was set aside

Cardinality estimation over graph patterns needed more machinery before it could say anything measurable.
The correlation part carried over to [Entity-aware cascades for semantic filters](Entity-aware%20cascades%20for%20semantic%20filters.md), which is smaller and has baselines to compare against directly.
Reasons in [2026-09-24](../Log/2026-09-24.md).

## References

- yanDBMSLLMIntegrationStrategies2025 - DBMS-LLM survey, names cost modelling of LLM operators as open
- zimmererSemCEBCardinalityEstimation2026 - SemCEB, cardinality estimation benchmark for semantic operators
- zhaoLarchLearnedQuery2026 - Larch, learned predicate ordering, states the independence assumption
- mangPLOPCostBasedPlacement2026 - PLOP, cost-based placement of semantic operators
- liWanderJoinOnline2016 - Wander Join, degree-weighted random walks over joins
- leisHowGoodAre2015 - Join Order Benchmark, estimators fail on predicates correlated across joins
- bestaGraphSeekNextGenerationGraph2026 - GraphSeek, LLM plans the query but never judges the data
- thapaliyaWhereLLMAnnotators2026 - homophilous neighbours share the same LLM mistake
- zhangLeveragingLargeLanguage2025 - label-free node classification from LLM annotations
