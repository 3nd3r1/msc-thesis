# CLAUDE.md

Code and experiments for Viljami's MSc thesis in computer science at the University of
Helsinki (graduating 2027), supervised by Prof. Jiaheng Lu (UDBMS group). The thesis
itself is written in LaTeX on Overleaf; this repo holds code, results and the notes.

## Layout

- `notes/` — all markdown notes, an Obsidian vault. **`notes/Thesis.md` is the only way
  in: read it first and follow its links.** Do not map the vault's structure here.
- `experiments/` — one folder per experiment, `NN_short_name/run.py` plus `results/`
  for small committed outputs.
- `data/` — raw data downloads, gitignored.
- `shell.nix` + `.envrc` — nix dev environment, entered via direnv.
- `README.md` — repo front page.

## The thesis in short

**Entity-aware cascades for semantic filters.** A semantic filter runs an LLM predicate
on every row. The standard way to make that affordable is a model cascade: a cheap proxy
scores each row, only uncertain rows are escalated to the expensive oracle, and
thresholds are calibrated to hit an accuracy target.

- Core idea: cascades treat rows as independent, but rows belong to entities (reviews of
  a product, posts by a user) and verdicts correlate within them. Spend the oracle budget
  per entity rather than per row, then settle an entity's remaining rows from the verdicts
  already confirmed within it.
- Gap: prior work (CSV, adaptive Two-Phase) exploits row similarity but infers groups from
  embeddings, because it targets corpora with no schema. Dataframe and SQL data already
  carries entity keys — exact and free — and no method uses them.
- Scope: single entity keys, `sem_filter` only. Foreign keys and graph edges are a stated
  extension, not the scope.
- Build on LOTUS and extend its existing cascade. Do not build a new engine.
- Primary metric: oracle calls at a fixed accuracy target. Baselines: full oracle run,
  LOTUS cascade, CSV, Two-Phase.
- Label each dataset once with the oracle, recording logprobs, so every later comparison
  runs offline.

## Working rules

- Experiments are numbered and never overwritten. A changed predicate, model or sample
  means a new experiment folder.
- Every run records its predicate, model, sample sizes and seed, in the code as constants
  and in the log.
- After a result or decision, add a dated entry to the work log — `notes/Thesis.md` links
  it. Record what was run, the key numbers, what it means. This log is the source for the
  thesis's methods and results chapters, so keep it accurate. It is the only place status
  is tracked — do not restate status in other files.
- LLM calls cost money. Cache verdicts to disk and reuse them; ask before anything that
  would make more than a few thousand calls.
- LOTUS changes go in a separate fork pinned in `requirements.txt`, not copied into this
  repo.
- Keep code simple and readable. Python, pandas, plain scripts over frameworks.

## How to communicate

- Concise and direct. No filler, no buzzwords.
- Plain-language explanations when a concept is new; don't assume background in areas
  outside databases and systems.
- Say plainly when something is wrong or a result doesn't support the idea.
