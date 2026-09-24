# Research log

Dated entries, newest first. What was run, the key numbers, what was decided and why.
The top entry is the current status — nowhere else in the repo tracks it.

Every experiment entry records its predicate, model, sample sizes and seed.

---

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
