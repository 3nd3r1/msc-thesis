# Playground

Exploratory work that isn't an experiment yet, like screening a dataset before deciding to use it.

- One folder per question, scripts and `results/` inside, no README or step structure needed.
- It is committed, so a log entry can point at a script or a result here.
- Scripts can import from `experiments/lib` and from the experiments.
- When a result is needed in the thesis or as part of the fixed design, it moves to a numbered experiment in `experiments/`.

## Contents

- `synergy_screen/` - screens the SYNERGY reviews for citation and coupling graphs against a matched cosine kNN, see the 2026-10-22 and 2026-10-23 logs.
