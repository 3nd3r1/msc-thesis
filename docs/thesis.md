# Thesis context

Background for anyone (human or Claude) working in this repo. `CLAUDE.md` has the short version.

## People

- **Prof. Jiaheng Lu** — supervisor, UDBMS (Unified Database Management Systems) group, University of Helsinki. Works on categorical unification of multi-model databases and on DBMS–LLM integration.
- **Zhengtong Yan** — PhD student in the group; AI for databases, reinforcement learning for query optimization. Co-authored the DBMS–LLM survey with Lu (arXiv 2507.19254).
- **Irena Holubová (Prague)** — long-standing collaborator; her group built the MM-cat tooling for multi-model data.

## How the topic was chosen

Viljami's background is implementation-heavy: infrastructure and data integration work in industry (Ericsson, Palantir). He prefers building systems over surveys or pure empirical comparisons.

Lu's steer: semantic operators over multi-model or multi-modal data is the direction he finds most promising. He pointed to two papers on multi-modal query processing with LLMs: **M2EX** and **CAESURA**.

The multi-modal variant was dropped because LOTUS already supports multimodal data. The multi-model variant, specifically graph data, was chosen instead.

Other directions considered and set aside: an executor for Lu's categorical algebra, cost models for LLM operators, schema routing over heterogeneous databases, LLM-assisted access-path specification for MM-cat, incremental view maintenance across heterogeneous sources, and LLM-assisted schema mapping. Kept here in case Lu redirects.

## Key concepts

**Semantic operators.** Relational-style operators (filter, join, map, aggregate, top-k) whose condition is written in natural language and evaluated by an LLM. Example: keep reviews where "the review complains about quality". Each evaluation is an LLM call, so cost and latency dominate, and query optimization is mostly about making fewer or cheaper calls.

**LOTUS.** The existing semantic-operator system this thesis builds on. It adds semantic operators to pandas DataFrames (`sem_filter`, `sem_join`, `sem_map`, ...) and has optimizations such as model cascades.

**SemCEB.** A benchmark for cardinality estimation of semantic operators: predicting how many rows pass an LLM predicate without running it on everything. Its gap for this thesis: it treats predicates as independent.

**Graph pattern queries.** Queries that match a fixed shape in a graph, e.g. `(user)-[wrote]->(review)-[about]->(product)`, with LLM filters on the nodes. Variable-length paths are out of scope.

## The pitch

1. LOTUS gives semantic operators over tables.
2. Cardinality estimation for them (SemCEB) ignores correlation between predicates.
3. On graph data, LLM verdicts are correlated along edges: reviews of the same product, or by the same user, tend to get the same verdict. That structure could improve selectivity estimates, filter placement, and let the system skip or batch LLM calls.

The first step is checking that the correlation actually exists (experiment 01).

## Thesis scope (tentative, pending Lu)

- Fixed-shape pattern queries with LLM filters.
- Optimizations: filter placement, caching, batching, correlation-aware estimation.
- Extend LOTUS; do not build a new engine.
- Data: Amazon Reviews 2023 (McAuley Lab) as the user–review–product graph.

## Helsinki thesis process

- Programme code MH50_009.
- PreThesis system: thesis plan (1–2 pages, submitted by the supervisor) → supervision agreement → seminar presentation → submission via Sisu / E-thesis.
- Seminar: no course enrolment needed; book a presentation slot via the e-form on the seminar page at least two weeks ahead.
- Graded 0–5 by two examiners.
- The thesis is written in LaTeX on Overleaf; figures and tables are generated here and uploaded there.

## Useful sources

- arXiv HTML pages (`arxiv.org/html/<id>`) render full papers better than abstract pages.
- DBLP and arXiv are more current than institutional group pages.
