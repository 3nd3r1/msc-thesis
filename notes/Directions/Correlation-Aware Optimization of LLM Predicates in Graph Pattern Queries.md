# Correlation-Aware Optimization of LLM Predicates in Graph Pattern Queries

## Problem

Many questions over graph data depend on what text means, not just how things are connected. A trust & safety team wants posts spreading vaccine misinformation by people who are friends with healthcare workers. A compliance team wants suppliers linked to companies whose filings mention sanctions exposure. Each is a graph pattern with one or more semantic predicates: conditions only a language model can judge.

Graph databases already let you write these queries, but they evaluate them naively. Neo4j added the ai.text.completion() function to Cypher 25 in Neo4j 2025.11 for generating text with external AI providers, and the documentation shows it applied to each row, optionally with concurrent transactions or Cypher's parallel runtime. Relational platforms do the same: Databricks' AI Functions take over parallelization, retries and scaling, and recommend submitting the whole dataset in one query. That makes each call efficient, but it doesn't reduce how many calls are made. A pattern that matches a million posts costs a million LLM calls. That is too slow for interactive use and infeasible on local hardware with a fixed GPU budget.

Yan and Lu's survey of DBMS–LLM integration [1] identifies cost modeling of LLM operators in query plans as a central open problem. This thesis addresses it for graph pattern queries.

## Related work and the gap

Semantic operator systems. LOTUS [2] defines semantic operators and optimizes them with model cascades. It treats an optimization as correct if it lowers cost while keeping a statistical accuracy guarantee relative to a reference algorithm. PLOP [3] places semantic filters, joins and projections relative to relational operators using a cost model. Larch [4] learns the order in which to evaluate multiple semantic filters. Abacus [5], Palimpzest [6], iPDB [7] and Sema [8] choose models, techniques and operator orders. iPDB, for example, shows that putting an accurate, highly selective semantic operator first reduces the tuples and tokens the next operator sees.

All of these estimate each predicate separately and combine the estimates as if predicates were independent. Larch states this explicitly: once per-instance selectivities are estimated accurately, the best order can be computed exactly, but under a predicate-independence assumption.

Cardinality estimation for semantic operators. SemCEB [9] is the first benchmark for this problem. It notes that current systems estimate cardinality from naive uniform samples, and evaluates semantic filters and joins in isolation, finding that sampling is robust across predicate categories but expensive and hard to scale. Nobody has studied correlation between semantic predicates across a join or an edge.

Correlation in classical query optimization. The Join Order Benchmark [10] showed that all industrial cardinality estimators routinely make large errors, and none handle predicates that are correlated across joins accurately. MSCN [11] learns such correlations. Wander join [12] estimates over joins with degree-weighted random walks, and join sampling has been studied since Chaudhuri et al. [13]. These methods assume cheap predicates. None account for each sample costing an LLM call.

Prefix caching for LLM queries. Liu et al. [14] reorder rows and columns to maximize KV-cache reuse and deduplicate redundant inference requests, implemented in Spark with vLLM as the serving backend. Their running example joins Amazon reviews with products and puts the product description first, so many reviews of the same product share a cached prefix. This thesis builds on their technique and does not claim it.

LLMs and graphs. Text-to-Cypher and agentic systems such as GraphSeek [15] use the LLM to plan queries. GraphSeek separates LLM planning from deterministic query execution over the full dataset, so the LLM never judges the data itself. In graph machine learning, LLM-GNN has an LLM annotate a small share of nodes and trains a GNN on those labels to predict the rest, with follow-ups Cella/Locle [17] and CANE [18]. These are offline node classification over fixed classes, not query-time predicates with accuracy guarantees. CANE also warns that homophilous neighbors can share the same LLM mistake, so a mislabeled node can still look reliable.

The gap. No system optimizes LLM predicates inside graph pattern queries. No study measures whether LLM verdicts are correlated along edges in real graphs, or how much that correlation misleads independence-based semantic optimizers.

## Key insight

Semantic verdicts on connected nodes are correlated. Misinformation clusters in friend groups, and complaints cluster on bad products. Graph statistics that engines already maintain, such as degree distribution, clustering coefficient and assortativity, predict how strong and how costly this correlation is, before any LLM call is made.

The economics also differ from classical optimization. Regular databases avoid join sampling because per-column histograms are free and bad plans are cheap. Semantic predicates have no free statistics, so an optimizer must sample with the LLM anyway. A bad plan costs many LLM calls, and sampled verdicts can be cached and reused during execution. Sampling along the pattern therefore becomes affordable where it rarely is in classical systems.

## Approach

C1. Empirical study of verdict correlation. Measure how strongly LLM verdicts correlate along different edge types (same-type edges such as knows, replyOf and cites, and cross-type edges such as review–product), and how often this leads independence-based optimizers to choose the wrong plan. A negative or mixed result is still a finding.

C2. Graph-aware cost model and placement (core). Extend PLOP-style placement to fixed-shape graph patterns. Selectivity and fan-out are estimated by degree-weighted sampling along the pattern [12]. Graph statistics decide when joint sampling is worth its LLM cost, and sampled verdicts are cached for reuse during execution.

C3. Graph-aware escalation in cascades (core, minimal version). Extend LOTUS-style cascades so that neighbors' verdicts can send a node to the large model when the small model's answer disagrees with its neighborhood. Neighbors never assign a verdict on their own. At equal cost, recall can only improve relative to the cascade baseline. Stretch: richer routing, and structural pruning with cheap predicates on parent nodes.

Infrastructure (not claimed as novel). A verdict cache keyed by (prompt, model version, node), and prefix-sharing batch order following Liu et al. [14].

## Worked example

The query "misinformation posts by friends of healthcare workers" can be run posts-first (check all 5M posts) or people-first (check 100k bios, then only the healthcare workers' friends' posts). An independence-based estimate assumes the 5k healthcare workers have about 100k distinct friends, which is everyone, so people-first looks no cheaper. In reality healthcare workers mostly know each other, so their friend lists overlap and they might have 25k distinct friends. People-first then costs about 1.35M calls instead of 5M. The clustering coefficient warns of this overlap in advance, and sampling along the knows edges measures it. (Numbers are illustrative.)

## Research questions

RQ1: How strongly are LLM verdicts correlated along edges in real graphs, and how often does this change the best plan?
RQ2: How much do the graph-aware cost model and escalation save at a fixed accuracy target?
RQ3: Which graph properties (assortativity, clustering, degree skew) predict when each technique helps?

## Evaluation

TODO
## Risks

TODO

## References

> **TODO** — the note cites [1]–[18] but the bibliography was never written.
> Numbers below are recovered from the names used in the text; `[16]` is never
> cited and its entry is unknown.

1. Yan and Lu — DBMS–LLM integration survey (arXiv 2507.19254).
2. LOTUS — *Semantic Operators and Their Optimization*, Patel et al., PVLDB 18(11), 2025.
3. PLOP — TODO
4. Larch — TODO
5. Abacus — TODO
6. Palimpzest — TODO
7. iPDB — TODO
8. Sema — TODO
9. SemCEB — TODO
10. Join Order Benchmark — TODO
11. MSCN — TODO
12. Wander join — TODO
13. Chaudhuri et al., join sampling — TODO
14. Liu et al., prefix caching / KV-cache reuse for LLM queries — TODO
15. GraphSeek — TODO
16. *(never cited in the text — likely LLM-GNN)* TODO
17. Cella / Locle — TODO
18. CANE — TODO
