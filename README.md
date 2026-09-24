# MSC Thesis - Semantic operators over graph data

Code, results and research log for Viljami's MSc thesis (computer science, University of Helsinki, supervised by Prof. Jiaheng Lu, UDBMS group).

**TLDR:** LLM-evaluated predicates inside fixed-shape graph pattern queries (e.g. `user–review–product`).
Existing cardinality estimation for semantic operators assumes predicates are independent, but on graphs LLM verdicts are likely correlated along edges — reviews of the same product tend to get the same verdict.
An optimizer that knows this should estimate better and make fewer LLM calls. Built as an extension of [LOTUS](https://github.com/lotus-data/lotus), not a new engine.

## Layout

```
CLAUDE.md        working rules for this repo
LOG.md           dated research log and current status
docs/thesis.md   research context, related work, open questions
data/            raw data (gitignored)
experiments/     NN_short_name/run.py + results/
```

Start with `docs/thesis.md` for the full context.

## Status

In the top entry of [`LOG.md`](LOG.md).
