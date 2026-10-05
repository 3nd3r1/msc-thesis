# 03 link guided search

## Question

For a rare predicate, does ordering oracle calls by links between rows find the positives in fewer calls than ordering by embedding similarity or by a small-LLM proxy score?

## Method

Replay on fully labelled data, so no new oracle calls.
Each order picks the next row to send to the oracle, and I count calls until 90% of the positives are found.

Orders compared:

- random
- embedding similarity
- small-LLM proxy score, the LOTUS and BARGAIN way, steps 2 and 3 only
- link expansion from confirmed positives
- links and embeddings together

Proxy calls are counted separately from oracle calls.

## Data

- Cora
- BIRD codebase_community, the Stack Exchange posts and comments, where the edges have to be built from the columns

## Models

- Llama 3.1 70B as the oracle,
- Llama 3.1 8B as the proxy

## Steps

Each step is `python run.py <step>` and writes `results/<step>.txt`

### 1. Graph benchmarks

Use the graphs' own class labels as the oracle, one predicate per class, on Cora, WikiCS, PubMed and ogbn-arxiv.
No LLM calls.
This is a positive control, since these graphs are benchmarks because their edges work.
PubMed should show no gain.

### 2. Cora with an LLM oracle

All 2,708 papers, 5 to 6 predicates that are not its topic classes.
Predicates go in `predicates.md` before labelling.

### 3. Ordinary tables

Generate candidate edges from every non-text column, same value, close in time, or references inside the text.
Run on All_Beauty with the existing labels, and on the posts and comments of BIRD codebase_community.

## Reading it

Stop after step 1 if links plus embeddings do not beat embeddings alone on Cora.

Step 3 counts as working if links plus embeddings need at least 20% fewer oracle calls than the best baseline on at least half of the non-control predicates.

Every edge type and predicate is reported, including the ones with no lift.
