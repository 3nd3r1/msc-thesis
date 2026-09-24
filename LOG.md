# Research log

Dated entries, newest first. What was run, the key numbers, what was decided and why.
The top entry is the current status — nowhere else in the repo tracks it.

Every experiment entry records its predicate, model, sample sizes and seed.

---

## 2026-09-24 — pivot: entity-aware cascades instead of graph cardinality estimation

Narrowed the topic. Working title is now **entity-aware cascades for semantic filters**.
Full writeup in `docs/thesis.md`.

What changed. The correlation insight is unchanged: rows that belong to the same entity
(reviews of a product, posts by a user) tend to get the same LLM verdict. What changed is
what we do with it. Before: use correlation to improve *cardinality estimation* for LLM
filters inside graph pattern queries, against the SemCEB gap. Now: use it inside a *model
cascade*, spending the oracle budget per entity instead of per row, and deciding an
entity's remaining rows from the verdicts already confirmed within it.

Why. The cascade framing has a primary metric that is directly comparable to existing work
(oracle calls at a fixed accuracy target), concrete baselines to beat (LOTUS's own cascade,
CSV, Two-Phase), and it extends code that already exists in LOTUS. Cardinality estimation
over graph patterns needed more machinery to say anything measurable. Graph edges are now
a stated extension — an entity key is a one-hop relationship — rather than the scope.

The gap it claims: CSV (Hou et al., VLDB 2026) and adaptive Two-Phase (Kim et al., 2026)
both exploit row similarity, but infer groups from embeddings because they target corpora
with no schema. Dataframe and SQL data already carries entity identifiers; `sem_filter`
just never looks at them. Exact grouping for free, and nobody has tested whether it helps.

Consequences for experiment 01. It is still the go/no-go on within-entity correlation, but
it now has to answer more, matching RQ1 and the three risks:
- correlation within product IDs (as before);
- the same measurement for embedding clusters, to show the entity key adds something
  embeddings do not already capture;
- the entity-size distribution, since long-tailed sizes cap the achievable saving —
  entities with 2–3 rows are too small to sample and vote on.
It should also record token logprobs while labelling, so the labelled set can be reused
offline for every later comparison and for the Bayes-error difficulty measure.
`experiments/01_edge_correlation/run.py` predates the pivot and covers only the first of
these.

Still no reply from Lu, and the pivot has not been sent to him.

## 2026-09-24 — repo set up, waiting on Lu

Repo scaffolded: `CLAUDE.md` (working rules), `docs/thesis.md` (topic and background),
this log, `README.md`. `shell.nix` for the dev environment. No code yet.

Direction was emailed to Lu in late September and there is no reply yet. Nothing is
committed until he agrees to the direction. After agreement the next deliverable is a
1–2 page thesis plan through PreThesis.

Downloaded `data/All_Beauty.jsonl.gz` (Amazon Reviews 2023, McAuley Lab) as the first
slice to work with — smallest category, enough to prototype on before touching a
larger one.

Next: experiment 01. Do `sem_filter` verdicts cluster by product, and does the
clustering survive conditioning on star rating? If verdicts are just a proxy for the
star rating, there is no new signal to exploit and the thesis pitch loses its core.

## Earlier — topic chosen

Lu's steer was semantic operators over multi-model or multi-modal data, pointing at
M2EX and CAESURA. The multi-modal variant was dropped because LOTUS already handles
multimodal data; the multi-model variant over graph data was taken instead. Directions
considered and set aside are listed in `docs/thesis.md`.
