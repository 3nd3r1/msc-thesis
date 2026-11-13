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
- `context_screen/` - runs the CAD screen on other datasets whose labels were given in context, Switchboard dialogue acts, MELD emotions and the Reddit and Gab hate conversations, see the 2026-11-12 log.
- `meld_proxy/` - checks whether the 8B proxy ranks MELD utterances by emotion on a balanced sample, and whether a prompt that says to judge the line alone stops its refusals, see the 2026-11-12 log.
- `switch_tuning/` - sweeps the decay and a proxy head start of `proxy+switch` against the proxy on every dataset with a proxy, see the 2026-11-12 log.
- `residual_screen/` - checks whether the correlation of the label or of the proxy residual along the edges tracks what links add to the proxy in step 4 of 03, see the 2026-11-13 log.
- `refusal_screen/` - checks whether how often the 8B refuses under the plain LOTUS prompt on a random sample tracks what links add to the proxy in step 4 of 03, with the pass rule fixed in the script before the run, see the 2026-11-13 log.
