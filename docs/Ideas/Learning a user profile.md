---
date: 2026-11-04
---

# Learning a user profile

Learn a user profile from a few samples, like past purchases, so a vague request can be turned into exact choices.

Someone says "do my shopping".
The LLM picks the exact products they would want from the store, based on what they have bought before.

## Link to the recall problem

This is the recall side of semantic operators.
A sem_filter goes through a large set of products and says yes or no to each.
The profile learned from the few samples is what drives the filter.

## Open questions

- What is the profile, a written summary the LLM keeps, a set of examples in the prompt, or something learned?
- How few samples are enough?
- How to measure it, recall of the products the person actually buys next?
- Is there a dataset with purchase histories?
- How does it connect to the recommender tricks in [Learning from verdicts](Learning%20from%20verdicts.md)?

Related: the meta thoughts and the "let my yes and no on a few rows define the predicate" direction in the [Oct 21 log](../Log/2026-10-21.md).
