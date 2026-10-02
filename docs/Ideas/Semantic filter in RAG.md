---
date: 2026-10-01
---
RAG retrieves by vector similarity, which tells you what a chunk looks like and not whether it answers the question.
A semantic filter over the candidates answers the second one.
It only pays off if a cascade gets the per-row cost low enough, which is the number [02](../../experiments/02_cascade_replay) is measuring anyway.

LLM rerankers already do something close, so the open part is what a filter with an accuracy target adds over a reranker with a score cutoff.
