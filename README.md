# MSC Thesis - Entity-aware cascades for semantic filters

Code, results and research log for Viljami's MSc thesis (computer science, University of Helsinki, supervised by Prof. Jiaheng Lu, UDBMS group).

**TLDR:** A semantic filter runs an LLM predicate on every row, so a filter over a million rows means a million model calls. Cascades cut that by escalating only uncertain rows to the expensive model — but they treat rows as independent.
Rows belong to entities: reviews of a product, posts by a user. Verdicts correlate within them, so once a few of a product's reviews are confirmed, the rest carry strong evidence for free.
Prior work exploits this only through embedding clusters; dataframe and SQL data already carries exact entity keys and nothing uses them. Built as an extension of [LOTUS](https://github.com/lotus-data/lotus)'s cascade, not a new engine.

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
