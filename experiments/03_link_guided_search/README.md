# 03 link guided search

## Question

For a rare predicate, does ordering oracle calls by links between rows find the positives in fewer calls than ordering by embedding similarity or by a small-LLM proxy score?

## Method

Replay on fully labelled data, so no new oracle calls.
Each order picks the next row to send to the oracle, and what I measure is the calls needed to find 90% of the positives, per predicate and order, as a share of all rows.
Lower is better, and random is the reference since it needs about 90% of the rows by construction.
Mean and sd over 5 seeds, because the orders break ties at random.

Orders compared:

- random
- embedding similarity
- small-LLM proxy score, the LOTUS and BARGAIN way, steps 2 and 3 only
- link expansion from confirmed positives
- links and embeddings together

Every order starts with no labels, so the first positive is found at random and only then do links and similarity have anything to point at.
Proxy calls in steps 2 and 3 are reported on their own, since an 8B call is not a 70B call.

## Data

- Cora
- BIRD codebase_community, the Stack Exchange posts and comments, where the edges have to be built from the columns

## Models

- Llama 3.1 70B as the oracle,
- Llama 3.1 8B as the proxy

## Steps

Each step is `python run.py <step>` and writes `results/<step>.txt`

### 1. Cora classes as predicates

Use the graph's own class labels as the oracle, one predicate per class.
No LLM calls.
This is a positive control, since Cora is a benchmark because its edges work.

#### Result

Links help on all 7 classes, and links plus embeddings is the best order on every one of them.

Calls to reach 90% recall, as a share of the 2,708 papers, mean of 5 seeds:

| class                  | rate  | random | embeddings | links | links+embeddings |
| ---------------------- | ----- | ------ | ---------- | ----- | ---------------- |
| Neural_Networks        | 30.2% | 89.8%  | 86.0%      | 57.0% | 47.3%            |
| Probabilistic_Methods  | 15.7% | 90.2%  | 80.5%      | 43.2% | 28.3%            |
| Genetic_Algorithms     | 15.4% | 90.5%  | 70.3%      | 17.1% | 16.4%            |
| Theory                 | 13.0% | 90.1%  | 68.2%      | 34.3% | 29.4%            |
| Case_Based             | 11.0% | 89.9%  | 57.7%      | 35.0% | 16.1%            |
| Reinforcement_Learning | 8.0%  | 89.2%  | 44.2%      | 29.1% | 12.5%            |
| Rule_Learning          | 6.6%  | 90.2%  | 59.6%      | 38.9% | 35.5%            |

Random lands at 90% of the rows on every class, which is the sanity check that the metric is right.

Two things to keep in mind before reading too much into it.
The embeddings baseline here is the binary word vector Cora ships, not a sentence embedding, and those are the same features a GNN would use to predict these classes.
Rule_Learning, the rarest class, is also the weakest result and the noisiest, 35.5% with an sd of 10.8% against 38.9% for links alone.

Go for step 2.

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
