# Playground

Quick tests and screening, like checking a dataset or a signal before deciding to test it properly.

- One folder per question, scripts and `results/` inside, no README or step structure needed.
- It is committed, so a log entry can point at a script or a result here.
- Scripts import shared code from `experiments/lib`, never from an experiment.
- When a quick test suggests a hypothesis worth testing, it becomes a numbered experiment with the hypothesis locked in its README, see `CLAUDE.md`.

## Contents

- `synergy_screen/` - screens the SYNERGY reviews for citation and coupling graphs against a matched cosine kNN, picks the replay reviews and checks whether citation reaches the positives the embedding order finds last and compares the degree of included and excluded papers, see the 2026-10-22, 2026-10-23, 2026-11-04 and 2026-11-06 logs.
- `synergy_proxy/` - checks whether a shorter claim or the 70B rescues the SYNERGY proxy, see the 2026-11-07 log.
- `cad_screen/` - screens CAD's reply, thread and author graphs against a matched cosine kNN,, whether they add to the embedding in a classifier and whether that beats fold noise and a shuffled graph, see the 2026-11-07 log.
- `context_screen/` - runs the CAD screen on other datasets whose labels were given in context, Switchboard dialogue acts, MELD emotions and the Reddit and Gab hate conversations, see the 2026-11-12 log.
- `meld_proxy/` - checks whether the 8B proxy ranks MELD utterances by emotion on a balanced sample, and whether a prompt that says to judge the line alone stops its refusals, see the 2026-11-12 log.
- `switch_tuning/` - sweeps the decay and a proxy head start of `proxy+switch` against the proxy on every dataset with a proxy, see the 2026-11-12 log.
- `residual_screen/` - checks whether the correlation of the label or of the proxy residual along the edges tracks what links add to the proxy in step 4 of 03, see the 2026-11-13 log.
- `refusal_screen/` - checks whether how often the 8B refuses under the plain LOTUS prompt on a random sample tracks what links add to the proxy in step 4 of 03, with the pass rule fixed in the script before the run, see the 2026-11-13 log.
- `shape_screen/` - checks whether how hard a predicate is to find from the text, the AP of a classifier on the embeddings, alone and times the lift, tracks what links add to the proxy in step 4 of 03, with the pass rules fixed in the scripts before the run, see the 2026-11-16 and 2026-11-17 logs.
- `bargain_screen/` - runs BARGAIN's recall target query on Cora, Stack, CAD and MELD with the 8B as the proxy and our labels as the oracle, so no calls, see the 2026-11-17 log, and `links.py` puts a first phase of search on the proxy, a cosine kNN, rewired links or the links before it at the same budget over a budget grid, see the 2026-11-18 log, and `wide.py` widens the budget grid of 06 to half the rows against QBCB from 06, see the 2026-11-19 log.
- `tar_screen/` - the TAR target method and QBCB stopping rules over the proxy, kNN, rewired and link orders against BARGAIN, written by agents, k not fixed before the run, see the 2026-11-18 log.
- `cal_screen/` - continuous active learning, a logistic regression on the proxy logit and the embedding refit on the verdicts, with and without counts of confirmed neighbours, as the order under QBCB against wide BARGAIN and the 06 orders, see the 2026-11-19 log.
- `bootstrap_screen/` - cluster bootstrap over threads and dialogues for intervals on the 06 gaps on CAD and MELD, the kNN rebuilt on each draw, resumable from `results/gaps_draws.jsonl`, see the 2026-11-19 and 2026-11-23 logs.
