# Correlation-Aware Optimization of LLM Predicates in Graph Pattern Queries

## Problem

Graph queries with semantic predicates run the LLM once per match, so a pattern matching a
million posts costs a million calls. Engines parallelise those calls but never reduce how
many are made. Cost modelling of LLM operators in query plans is an open problem
[@yanDBMSLLMIntegrationStrategies2025].

## Gap

Semantic operator optimizers estimate each predicate separately and then combine the estimates
as if the predicates were independent. SemCEB [@zimmererSemCEBCardinalityEstimation2026]
benchmarks cardinality estimation but evaluates filters and joins in isolation. I found no
measurement of correlation between semantic predicates across an edge or a join.

## Idea

Semantic verdicts on connected nodes correlate, the way misinformation clusters in friend
groups or complaints cluster on a bad product. Graph statistics that engines already keep, such
as clustering coefficient and assortativity, predict that correlation before any LLM call.

The economics differ from classical optimization. Semantic predicates have no free statistics, so
the optimizer has to sample with the LLM anyway, and sampled verdicts can be cached and reused
during execution. That makes sampling along the pattern affordable here, where in a normal
database it would not be.

Contributions:

- measure verdict correlation along edge types, and how often independence picks the wrong plan
- extend PLOP-style placement to fixed-shape graph patterns, using degree-weighted sampling
- extend cascades so neighbour verdicts can escalate a node to the large model

## Research questions

1. How strongly do LLM verdicts correlate along edges, and how often does that change the
   best plan?
2. How much do the cost model and the escalation save at a fixed accuracy target?
3. Which graph properties predict when each technique helps?

## Why it was set aside

The correlation insight carried over to [Entity-aware cascades for semantic filters](Entity-aware%20cascades%20for%20semantic%20filters.md),
which is smaller and has baselines I can compare against directly. Reasons in
[2026-09-24](../Log/2026-09-24.md).

## References

- yanDBMSLLMIntegrationStrategies2025 - DBMS-LLM survey, names cost modelling of LLM operators as open
- zimmererSemCEBCardinalityEstimation2026 - SemCEB, cardinality estimation benchmark for semantic operators
- lotus - LOTUS, semantic operators and model cascades
- mangPLOPCostBasedPlacement2026 - PLOP, cost-based placement of semantic operators
- zhaoLarchLearnedQuery2026 - Larch, learned predicate ordering, states the independence assumption
- russoAbacusCostBasedOptimizer2026 - Abacus, cost-based optimizer for semantic operator systems
- qiSemaHighperformanceSystem2026 - Sema, system for LLM-based semantic query processing
- kumarasingheIPDBOptimizingSemantic2026 - iPDB, selective semantic operators first cuts later tokens
- leisHowGoodAre2015 - Join Order Benchmark, estimators fail on predicates correlated across joins
- liWanderJoinOnline2016 - Wander Join, degree-weighted random walks over joins
- chaudhuriRandomSamplingJoins1999 - random sampling over joins
- liuOptimizingLLMQueries2025 - reorders rows to reuse the KV cache prefix
- bestaGraphSeekNextGenerationGraph2026 - GraphSeek, LLM plans the query but never judges the data
- zhangLeveragingLargeLanguage2025 - label-free node classification from LLM annotations
- thapaliyaWhereLLMAnnotators2026 - homophilous neighbours share the same LLM mistake
- zhengMICROLightweightMiddleware2026 - cross-store cross-model graph-relation joins
- xuBridgingGapCardinality2026 - cardinality estimation for semantic queries on unstructured data
- urbanSelectivityEstimationSemantic2026 - selectivity estimation for semantic filters on images
