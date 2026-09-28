# MSC Thesis - Entity-aware cascades for semantic filters

Code, results and research log for Viljami's MSc thesis (computer science, University of Helsinki, supervised by Prof. Jiaheng Lu, UDBMS group).

**TLDR:** A semantic filter runs an LLM predicate on every row, so a filter over a million rows means a million model calls. Cascades cut that by escalating only uncertain rows to the expensive model — but they treat rows as independent.
Rows belong to entities: reviews of a product, posts by a user. Verdicts correlate within them, so once a few of a product's reviews are confirmed, the rest carry strong evidence for free.
Prior work exploits this only through embedding clusters; dataframe and SQL data already carries exact entity keys and nothing uses them. Built as an extension of [LOTUS](https://github.com/lotus-data/lotus)'s cascade, not a new engine.

## Layout

```
CLAUDE.md        working rules for this repo
notes/           Obsidian vault: all markdown notes, start at notes/Thesis.md
data/            raw data (gitignored)
experiments/     NN_short_name/run.py + results/
```

Start with [`notes/Thesis.md`](notes/Thesis.md), or [`notes/Thesis context.md`](notes/Thesis%20context.md) for the full context.

## Status

In the newest file in [`notes/Log/`](notes/Log).
