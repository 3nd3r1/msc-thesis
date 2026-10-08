# 03 link guided search

## Question

For a predicate, does ordering oracle calls by links between rows find the positives in fewer calls than ordering by embedding similarity or by a small-LLM proxy score?

## Method

Replay on fully labelled data, so no new oracle calls.
Each order picks the next row to send to the oracle, and what I measure is the calls needed to find 90% of the positives, per predicate and order, as a share of all rows.
Lower is better, and random is the reference since it needs about 90% of the rows by construction.
Reported as mean and sd over seeds, because the orders break ties at random.

Orders compared:

- random
- embedding similarity
- link expansion from confirmed positives
- links and embeddings together, which in practice means links first and similarity as the tiebreak, since a confirmed neighbour counts 1 and a cosine is about 0.2
- small-LLM proxy score, the LOTUS and BARGAIN way
- the proxy score with link expansion on top

Every order starts with no labels, so the first positive is found at random and only then do links and similarity have anything to point at.
The two proxy orders sit in their own step, since an 8B call on every row is a cost the others never pay, and comparing them against each other holds it fixed.

## Data

- Cora, the Graph-COM text attributed copy, which is the LINQS graph plus titles and abstracts, verified to have the same nodes and edges
- BIRD codebase_community, the Stack Exchange posts and comments, where the edges have to be built from the columns

## Models

- Llama 3.1 70B Instruct Turbo as the oracle,
- Llama 3.1 8B as the proxy

## Steps

Each step is `python run.py <step>` and writes `results/<step>.txt`.

### 1. Cora classes as predicates

Step `classes`.
Use the graph's own class labels as the oracle.
This is a positive control, since Cora is a benchmark because its edges work.

#### Result

Links help on all 7 classes, and links plus embeddings is the best order on every one of them.

Calls to reach 90% recall, as a share of the 2,708 papers, mean of 50 seeds:

| class                  | rate  | random | embeddings | links | links+embeddings |
| ---------------------- | ----- | ------ | ---------- | ----- | ---------------- |
| Neural_Networks        | 30.2% | 90.1%  | 86.1%      | 56.1% | 47.3%            |
| Probabilistic_Methods  | 15.7% | 90.0%  | 80.5%      | 39.1% | 28.1%            |
| Genetic_Algorithms     | 15.4% | 90.1%  | 70.3%      | 17.2% | 16.5%            |
| Theory                 | 13.0% | 89.9%  | 68.1%      | 38.5% | 29.3%            |
| Case_Based             | 11.0% | 89.4%  | 57.4%      | 36.8% | 16.1%            |
| Reinforcement_Learning | 8.0%  | 89.8%  | 44.7%      | 36.2% | 12.4%            |
| Rule_Learning          | 6.6%  | 89.7%  | 60.0%      | 38.8% | 37.0%            |

### 2. Cora with an LLM oracle

Step `label` and then step `predicates`.

- `label` makes gets the result of every predicate on every paper from the oracle.
- `predicates` replays the orders on those labels and makes no oracle calls.

#### Predicates

| key          | kind        | claim                                                                  |
| ------------ | ----------- | ---------------------------------------------------------------------- |
| proof        | non-topical | the abstract says the paper proves a theorem or derives a formal bound |
| language     | topical     | the paper is about natural language, text or speech                    |
| biology      | topical     | the abstract says the method is applied to biological or medical data  |
| robotics     | topical     | the paper is about robots or controlling a physical device             |
| first person | control     | the abstract is written in the first person singular, using I or my    |
| markov       | non-topical | the abstract says the paper uses a Markov model or Markov process      |
| unsupervised | topical     | the paper is about unsupervised learning or clustering                 |

#### Result

BLUF: Links and embeddings beat embeddings alone on 2 of the 6 real predicates, markov and language.

Calls to reach 90% recall as a share of the rows, mean of 50 seeds, from `results/predicates.txt`:

| predicate    | rate | random | embeddings | links | links+embeddings |
| ------------ | ---- | ------ | ---------- | ----- | ---------------- |
| robotics     | 4.3% | 89.4%  | 12.3%      | 74.6% | 14.2%            |
| markov       | 3.6% | 89.1%  | 18.5%      | 74.9% | 13.7%            |
| language     | 3.2% | 89.0%  | 39.3%      | 73.5% | 31.0%            |
| unsupervised | 3.6% | 88.8%  | 47.9%      | 81.2% | 45.3%            |
| biology      | 4.8% | 90.1%  | 64.1%      | 83.0% | 64.1%            |
| proof        | 6.6% | 90.0%  | 74.7%      | 81.8% | 75.0%            |
| first person | 2.8% | 88.6%  | 90.7%      | 86.6% | 90.7%            |

### 3. How much the citations correlate the positives

Step `lift`

Of the edges leaving a positive the share that land on a positive over the positive rate.
1.0x is no correlation.
Also the share of positives with at least one positive neighbour, which grows with degree, and the share with no neighbour at all, which links can never reach.

### 4. Oracle calls under a proxy

Step `proxy` and then step `cascade`.

- `proxy` gets p(yes) on every paper and predicate from the small model.
- `cascade` replays the proxy order and the proxy with link expansion on top, and makes no oracle calls.

### 5. Ordinary tables

Generate candidate edges from every non-text column, same value, close in time, or references inside the text.
Run on All_Beauty with the existing labels, and on the posts and comments of BIRD codebase_community.
