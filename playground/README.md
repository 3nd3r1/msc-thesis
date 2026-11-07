# Playground

Exploratory work that isn't an experiment yet, like screening a dataset before deciding to use it.

- One folder per question, scripts and `results/` inside, no README or step structure needed.
- It is committed, so a log entry can point at a script or a result here.
- Scripts can import from `experiments/lib` and from the experiments.
- When a result is needed in the thesis or as part of the fixed design, it moves to a numbered experiment in `experiments/`.

## Contents

- `synergy_screen/` - screens the SYNERGY reviews for citation and coupling graphs against a matched cosine kNN, picks the replay reviews and checks whether citation reaches the positives the embedding order finds last and compares the degree of included and excluded papers, see the 2026-10-22, 2026-10-23, 2026-11-04 and 2026-11-06 logs.
- `synergy_proxy/` - checks whether a shorter claim or the 70B rescues the SYNERGY proxy, see the 2026-11-07 log.
- `cad_screen/` - screens CAD's reply, thread and author graphs against a matched cosine kNN,, whether they add to the embedding in a classifier and whether that beats fold noise and a shuffled graph, see the 2026-11-07 log.
