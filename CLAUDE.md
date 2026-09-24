# CLAUDE.md

Code and experiments for Viljami's MSc thesis in computer science at the University of Helsinki (graduating 2027), supervised by Prof. Jiaheng Lu (UDBMS group). The thesis itself is written in LaTeX on Overleaf; this repo holds code, results and the research log only.

Full research context, related work and open questions: `docs/thesis.md`. Read it before proposing new experiments or changing direction.

## The thesis in short

Semantic operators over graph data: LLM-evaluated predicates (e.g. "review complains about quality") inside graph pattern queries.

- Scope: fixed-shape pattern queries such as user–review–product, with LLM filters. Focus on filter placement, caching, batching, and cost/cardinality estimation. Variable-length paths are out of scope.
- Build on an existing semantic-operator system (LOTUS) and extend it. Do not build a new engine.
- Core idea: cardinality estimation for semantic operators (SemCEB) assumes predicates are independent. On graphs, LLM verdicts are likely correlated along edges (reviews of the same product tend to get the same verdict). An optimizer that knows this can estimate better and skip LLM calls.

## Current status

- Direction emailed to Lu for feedback (late Sep 2026); waiting for a reply. Nothing is committed until he agrees.
- Next deliverable after agreement: a 1–2 page thesis plan submitted through PreThesis.
- Experiment 01 tests the core idea: do `sem_filter` verdicts cluster by product in Amazon Reviews 2023, also after conditioning on star rating?

## Repo layout

```
CLAUDE.md
LOG.md              dated research log: what ran, results, decisions
docs/thesis.md      research context and background
requirements.txt
data/               gitignored raw data (downloads)
experiments/
  NN_short_name/
    run.py
    results/        small outputs, committed
```

## Working rules

- Experiments are numbered and never overwritten. A changed predicate, model or sample means a new experiment folder.
- Every run records its predicate, model, sample sizes and seed, in the code as constants and in `LOG.md`.
- After a result or decision, add a short dated entry to `LOG.md`: what was run, the key numbers, what it means. This log is the source for the thesis's methods and results chapters, so keep it accurate.
- LLM calls cost money. Cache verdicts to disk and reuse them; ask before anything that would make more than a few thousand calls.
- LOTUS changes go in a separate fork pinned in `requirements.txt`, not copied into this repo.
- Keep code simple and readable. Python, pandas, plain scripts over frameworks.

## How to communicate

- Concise and direct. No filler, no buzzwords.
- Plain-language explanations when a concept is new; don't assume background in areas outside databases and systems.
- Say plainly when something is wrong or a result doesn't support the idea.
