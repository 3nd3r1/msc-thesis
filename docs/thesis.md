# Thesis context

Background for anyone (human or Claude) working in this repo: what the topic is and why.
`CLAUDE.md` has the short version and the working rules. Current status and results are
in `LOG.md`, not here — this file describes the topic, not where it stands.

**Working title:** Entity-aware cascades for semantic filters.

## People

- **Prof. Jiaheng Lu** — supervisor, UDBMS (Unified Database Management Systems) group, University of Helsinki. Works on categorical unification of multi-model databases and on DBMS–LLM integration.
- **Zhengtong Yan** — PhD student in the group; AI for databases, reinforcement learning for query optimization. Co-authored the DBMS–LLM survey with Lu (arXiv 2507.19254).
- **Irena Holubová (Prague)** — long-standing collaborator; her group built the MM-cat tooling for multi-model data.

## Key concepts

**Semantic operators.** Relational-style operators (filter, join, map, aggregate, top-k) whose condition is written in natural language and evaluated by an LLM. Example: keep reviews where "the review complains about quality". Each evaluation is an LLM call, so cost and latency dominate, and optimization is mostly about making fewer or cheaper calls.

**LOTUS.** The existing semantic-operator system this thesis builds on. It adds semantic operators to pandas DataFrames (`sem_filter`, `sem_join`, `sem_map`, ...) and already implements model cascades. The thesis extends its cascade rather than building a new engine.

**Model cascade.** The standard way to make a semantic filter affordable. A cheap *proxy* model scores every row; only rows whose score is uncertain are escalated to the expensive *oracle* model. Thresholds are calibrated so the final result meets a user-specified accuracy target. The saving comes from rows the proxy can settle on its own.

**Entity key.** A column already in the schema that groups rows belonging to the same real-world thing: product ID for reviews, author ID for posts, thread ID for messages. Grouping by it is exact and costs nothing, unlike inferring groups from embeddings.

**Oracle / proxy.** Oracle = the expensive, trusted model whose verdict is treated as ground truth. Proxy = the cheap model (or the same model with a smaller variant) used to avoid oracle calls.

## Motivation

A semantic filter evaluates a natural-language predicate on every row with an LLM. Without optimization, a filter over a million rows means a million model calls, which makes it the dominant cost in semantic query processing.

Cascades score each row from its own text and treat rows as independent. Real data often violates this. Rows belong to entities: reviews of a product, posts by a user, messages in a thread. A defective product attracts many reviews describing the same failure. Once a few of its reviews are confirmed as injury reports, the rest carry strong evidence before the model sees them. Current cascades pay full price for that evidence.

## Gap

Recent work exploits similarity between rows, but only through text.

- **CSV** (Hou et al., VLDB 2026) clusters rows by embedding, labels a sample per cluster with the oracle, and propagates the majority label.
- **Adaptive Two-Phase** (Kim et al., 2026) runs this clustering first and escalates to a trained proxy when clusters stay mixed.

In both, grouping is *inferred* from embeddings, because the methods were designed for unstructured corpora with no schema. In dataframe and SQL systems the data already carries entity identifiers, but `sem_filter` only sees the columns its predicate references. Entity grouping is free and exact. Whether it is informative depends on the predicate, and no existing method tests or uses it.

## Idea

Extend the cascade with entity-aware evaluation. Partition rows by an entity key from the schema and spend the oracle budget adaptively across entities rather than uniformly across rows. Then use confirmed verdicts within an entity to decide its remaining rows. Candidate mechanisms:

- per-entity sampling and voting (CSV with cluster = entity);
- neighbour verdicts as a feature of the proxy;
- per-entity calibration thresholds.

Embedding clustering remains the fallback for rows without a useful key, and the two can be combined. An entity key is a one-hop relationship, so the approach extends naturally to foreign keys and graph edges in multi-model data. The thesis scopes to single keys.

## Research questions

1. How strongly are semantic-filter verdicts correlated within schema entities, and how does this compare with embedding clusters? How does the entity-size distribution limit the achievable savings?
2. How many oracle calls does entity-aware cascading save at a fixed accuracy target, compared with a standard cascade and with clustering-based cascades?
3. Can accuracy guarantees be kept when evidence is shared within an entity? Per-entity voting can likely reuse CSV-style sampling bounds. A proxy that uses neighbour verdicts produces correlated errors, which breaks the independence assumptions behind existing calibrations.
4. How can the method detect weak grouping and fall back, so that it never does worse than the baselines?

## Evaluation

Two or three datasets with entity keys (tentatively product reviews with product IDs), several predicates each. Each dataset is labelled once with the oracle, recording token logprobs, so all later comparisons run offline. The logprobs also give per-query Bayes error, which can be used to measure query difficulty and a lower bound on oracle calls, following Kim et al.

- **Primary metric:** oracle calls at a fixed corpus-accuracy target, for direct comparison with CSV and Two-Phase. Precision and recall are reported as well.
- **Baselines:** full oracle run, the LOTUS cascade, CSV, Two-Phase, and CSV with the entity key concatenated into the embedded text.
- **Implementation:** extends LOTUS's existing cascade.

## Risks

- Verdicts may be only weakly correlated within entities.
- Embedding clusters may already capture the same structure.
- Entity sizes are typically long-tailed. Most entities have only a few rows, too few to sample and vote on, so savings depend on the large entities.

All three can be measured in the first experiment, before committing to the method.

## How the topic was chosen

Viljami's background is implementation-heavy: infrastructure and data integration work in industry (Ericsson, Palantir). He prefers building systems over surveys or pure empirical comparisons.

Lu's steer: semantic operators over multi-model or multi-modal data is the direction he finds most promising. He pointed to two papers on multi-modal query processing with LLMs: **M2EX** and **CAESURA**.

Routes taken and then narrowed:

- The multi-modal variant was dropped because LOTUS already supports multimodal data.
- The multi-model variant over graph data was taken next: LLM filters inside fixed-shape graph pattern queries, with correlation along edges improving *cardinality estimation* (the gap in **SemCEB**, a benchmark for cardinality estimation of semantic operators that treats predicates as independent).
- That narrowed again to the present topic. The correlation insight survived; the vehicle changed from cardinality estimation over graph patterns to cascades over a single entity key, which is a smaller, more measurable contribution with directly comparable baselines. Graph edges become a natural extension rather than the scope.

Other directions considered and set aside: an executor for Lu's categorical algebra, general cost models for LLM operators, schema routing over heterogeneous databases, LLM-assisted access-path specification for MM-cat, incremental view maintenance across heterogeneous sources, and LLM-assisted schema mapping. Kept here in case Lu redirects.

## Helsinki thesis process

- Programme code MH50_009.
- PreThesis system: thesis plan (1–2 pages, submitted by the supervisor) → supervision agreement → seminar presentation → submission via Sisu / E-thesis.
- Seminar: no course enrolment needed; book a presentation slot via the e-form on the seminar page at least two weeks ahead.
- Graded 0–5 by two examiners.
- The thesis is written in LaTeX on Overleaf; figures and tables are generated here and uploaded there.

## Useful sources

- arXiv HTML pages (`arxiv.org/html/<id>`) render full papers better than abstract pages.
- DBLP and arXiv are more current than institutional group pages.
- LOTUS: *Semantic Operators and Their Optimization*, Patel et al., PVLDB 18(11), 2025.
