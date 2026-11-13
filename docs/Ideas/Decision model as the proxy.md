---
date: 2026-11-13
---

# Decision model as the proxy

Jev from TypeSafe AI is a non-generative decision model that gives a calibrated probability for a boolean, a choice or a score in one forward pass, instead of generating True or False.
JEVDB [@wangPruneFirstDecide2026] puts it into a semantic database as the fast path and sends only the rows inside an uncertainty band to a frontier LLM.
The band is set on 400 calibration items per predicate so the recall and precision loss stay under a tolerance.
It is evaluated on SemBench and on joins over TPC-DS.
Jev alone at a 0.5 threshold ranges from 24 to 89% F1 between queries, so like our 8B it depends strongly on the predicate.
JEVDB uses relational structure only to prune rows with semijoins, never a neighbour's label as evidence about a row.

## What carries over

- Jev fills the proxy slot of 03 step 4, a calibrated replacement for the 8B.
- Jev still reads one row, so on CAD and MELD links should still add to it. Showing that would make the RQ2 result hold for a stronger proxy than the 8B.
- Jev cannot refuse, so the refusal check in `playground/refusal_screen` does not transfer to it. The equivalent signal is the share of a random sample inside the uncertainty band, which is closer to what RQ3 means, a proxy that does not know.
- The 8B cannot stand in for this, its mean p is 0.84 to 0.996 on every predicate in the `proxy.txt` files of 03, so its uncertainty band is nearly empty.

## Open

- Jev is a closed commercial API while the rest of the setup is open Llama, and pricing and access are not checked.
- It is a scope change after the design lock, so it needs Lu.
- A first try would be Jev on the same 300 row samples as the refusal screen, the share inside the band against the step 4 gains, with the pass rule fixed before the run.

Related: [Learning from PROBE](Learning%20from%20PROBE.md).
