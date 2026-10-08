# CLAUDE.md

Code and experiments for Viljami's MSc thesis in computer science at the University of
Helsinki (graduating 2027), supervised by Prof. Jiaheng Lu (UDBMS group). The thesis
itself is written in LaTeX on Overleaf; this repo holds code, results and the notes.

## Layout

- `docs/` - all markdown notes.
  - `docs/Thesis.md` is the way in
- `experiments/` - one folder per experiment, `NN_short_name/run.py` plus `results/`
- `labels/` - oracle labels per dataset, shared between experiments
- `data/` - raw data downloads, gitignored.
- `README.md` - repo front page.

## Working rules

- Never write useless markdown accents like `*` or `_` in notes.
- In markdown write one sentence per line. Do not wrap sentences.
- Do not repeat information in multiple places.
- After a result or decision ask to add a short entry to docs/Log.
- Keep code extremely simple by default. No useless comments.
- Never run experiment scripts unless asked. Verifying a piece in isolation is fine.
- A result is a BLUF line, the table, and the file it came from. Nothing else.
- Do not explain or interpret a result unless asked. If an explanation is needed it goes in docs/Log.
- Do not restate rows of a table in prose.
- Do not write numbers that the code already determines, like call counts or seed counts.

## How to communicate

- Concise and direct. No filler, no buzzwords.
- No em dashes.
- No ;
- Say plainly when something is wrong or a result doesn't support the idea.
