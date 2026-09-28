# Entity-aware cascades for semantic filters

**Status:** active — current direction.

## Problem

A semantic filter evaluates a natural-language predicate on every row with an LLM. Without optimization, a filter over a million rows means a million model calls, which makes it the dominant cost in semantic query processing. The standard remedy is a model cascade. A cheap proxy scores each row, only uncertain rows are escalated to the expensive model (the oracle), and thresholds are calibrated so the result meets a user-specified accuracy target.

## Related work and the gap

Recent work exploits similarity between rows, but only through text. CSV (Hou et al., VLDB 2026) clusters rows by embedding, labels a sample per cluster with the oracle, and propagates the majority label. The adaptive Two-Phase method (Kim et al., 2026) runs this clustering first and escalates to a trained proxy when clusters stay mixed. In both, grouping is inferred from embeddings, because the methods were designed for unstructured corpora with no schema. In dataframe and SQL systems the data already carries entity identifiers, but `sem_filter` only sees the columns its predicate references. Entity grouping is free and exact. Whether it is informative depends on the predicate, and no existing method tests or uses it.

## Key insight

Cascades score each row from its own text and treat rows as independent. Real data often violates this. Rows belong to entities: reviews of a product, posts by a user, messages in a thread. A defective product attracts many reviews describing the same failure. Once a few of its reviews are confirmed as injury reports, the rest carry strong evidence before the model sees them. Current cascades pay full price for that evidence.

## Approach

Extend the cascade with entity-aware evaluation. Partition rows by an entity key from the schema and spend the oracle budget adaptively across entities rather than uniformly across rows. Then use confirmed verdicts within an entity to decide its remaining rows. Candidate mechanisms:

- per-entity sampling and voting (CSV with cluster = entity);
- neighbour verdicts as a feature of the proxy;
- per-entity calibration thresholds.

Embedding clustering remains the fallback for rows without a useful key, and the two can be combined. An entity key is a one-hop relationship, so the approach extends naturally to foreign keys and graph edges in multi-model data. The thesis scopes to single keys.

## Worked example

> **TODO** — a concrete query with numbers, the way the graph direction has one:
> a dataset, a predicate, entity sizes, and oracle calls saved versus a plain cascade.

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

## References

> **TODO** — full citations.

- **CSV** — Hou et al., VLDB 2026.
- **Adaptive Two-Phase** — Kim et al., 2026.
- **LOTUS** — *Semantic Operators and Their Optimization*, Patel et al., PVLDB 18(11), 2025.
