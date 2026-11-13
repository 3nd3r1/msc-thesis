---
date: 2026-11-13
---

# Learning from PROBE

[@zhaoPROBELearnedProxyBased2026] labels a sample with the LLM, trains a logistic regression proxy, corrects it over a kNN graph on embeddings and sends the uncertain rows to the LLM.
The correction is Correct and Smooth [@huangCombiningLabelPropagation2020], bias diffusion of the residual, LLM label minus proxy score, then smoothing of the scores.
Bias diffusion does most of the work in their ablation.
It is evaluated on SemBench and LRO-Bench against LOTUS, BARGAIN, ScaleDoc, CSV and UQE, and the code is public.

## What carries over

- A `proxy+diffuse` order for 03, diffusing the residuals of the oracle verdicts over the real links. It uses negatives as well as positives and normalises by degree, which is what fixed `blend/deg` on SYNERGY.
- Diffusion over links against over the cosine kNN against over both is the fair comparison with PROBE, and the same split as the screens in step 3.
- PROBE says its refinement weakens when positives and negatives do not separate in embedding space. Links are structure outside that space, so the claim to test is that links add to methods like PROBE when the label depends on context outside the row.
- Their pilot sample with a confidence bound that picks the sampling mode is a template for an RQ3 rule that picks links or not.

## What did not carry over

Their refinement assumes the proxy's errors correlate along edges.
Neither that nor label correlation along the edges tracks what links add on top of the proxy, see the 2026-11-13 log.

## Comparing with existing methods

- Their methods on our datasets, PROBE or BARGAIN on CAD and MELD, keeps what makes the datasets interesting.
- Our method on their datasets needs row to row links. Not checked whether SemBench or LRO-Bench tables have foreign keys that give them, like reviews of the same movie, which would be the [Entity key](../Concepts/Entity%20key.md) idea again.
- They report F1 against the share of rows sent to the LLM and 03 reports calls to a recall target, so one side needs converting.

Related: [Structure outside the row](Structure%20outside%20the%20row.md).
