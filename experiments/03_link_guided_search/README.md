# 03 link guided search

## Question

For a predicate, does ordering oracle calls by links between rows find the positives in fewer calls than ordering by embedding similarity or by a small-LLM proxy score?

## Method

Replay on fully labelled data, so no new oracle calls.
Each order picks the next row to send to the oracle, and what I measure is the calls needed to find a target share of the positives, per predicate and order, as a share of all rows.
Lower is better, and random is the reference since it needs about the target share of the rows by construction.
Targets are 50, 80, 90 and 95% in steps 2 and 4, since links can only matter once the proxy has used up its good rows.
Above 95% the predicates have too few positives for one more to mean less than a percent of recall.
Reported as mean and sd over seeds, because the orders break ties at random.

Orders compared:

- random
- embedding similarity
- link expansion from confirmed positives
- links and embeddings together, which in practice means links first and similarity as the tiebreak, since a confirmed neighbour counts 1 and a cosine is about 0.2
- positive minus negative confirmed neighbours, since the three orders above drop every negative the oracle returns
- `blend a`, which is `a * (positive - negative neighbours) + mean cosine`, for a in 1.0, 0.3 and 0.1, so the two signals sit on one scale instead of links deciding
- small-LLM proxy score, the LOTUS and BARGAIN way
- the proxy score with link expansion on top

Every order starts with no labels, so the first positive is found at random and only then do links and similarity have anything to point at.
`blend 1.0` isolates the negatives and `blend 0.3` and `blend 0.1` isolate the scaling, so each order differs from the one above it by one thing.
With no edges `blend 1.0` is the same order as links and embeddings together.
The two proxy orders sit in their own step, since an 8B call on every row is a cost the others never pay, and comparing them against each other holds it fixed.

## Data

- Cora, the Graph-COM text attributed copy, which is the LINQS graph plus titles and abstracts, verified to have the same nodes and edges
- Stack Exchange, the Cross Validated dump from BIRD `codebase_community` as the datasets `stack` and `stack_author`, which are 5,243 questions with two edge sets that share the labels, postLinks where a human marked two questions as related at 6,460 edges and same author at 6,295

The pool is the postLinks giant component, mean degree 2.46 against Cora's 3.90.
Over all 42,912 questions the degree is 0.47 and only a quarter have any neighbour, so a random sample would have almost no edges left.
Shared tag edges are out.
A tag is a topic label, so they leak any topical predicate the way Cora's classes do.

## Models

- Llama 3.1 70B Instruct Turbo as the oracle,
- Llama 3.1 8B as the proxy

## Steps

Each step is `python run.py <dataset> <step>` and writes `results/<dataset>/<step>.txt`.
Datasets are `cora`, `stack` and `stack_author`.
Every loader returns rows with a title and a text column plus the edges, so only `classes` is tied to one dataset.

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

### 2. Predicates with an LLM oracle

Step `label` and then step `predicates`.

- `label` gets the verdict of every predicate on every row from the oracle.
- `predicates` replays the orders on those labels and makes no oracle calls.

#### Predicates

##### Cora

| key          | kind        | claim                                                                  |
| ------------ | ----------- | ---------------------------------------------------------------------- |
| proof        | non-topical | the abstract says the paper proves a theorem or derives a formal bound |
| language     | topical     | the paper is about natural language, text or speech                    |
| biology      | topical     | the abstract says the method is applied to biological or medical data  |
| robotics     | topical     | the paper is about robots or controlling a physical device             |
| first person | control     | the abstract is written in the first person singular, using I or my    |
| markov       | non-topical | the abstract says the paper uses a Markov model or Markov process      |
| unsupervised | topical     | the paper is about unsupervised learning or clustering                 |

##### Stack

| key         | role      | claim                                                               |
| ----------- | --------- | ------------------------------------------------------------------- |
| spss        | lexical   | the question mentions SPSS or Stata                                 |
| rcode       | lexical   | the question includes R code                                        |
| choose      | high rate | the question asks which statistical method or test to use           |
| assumptions | topical   | the question is about whether an assumption of a method is violated |
| interaction | topical   | the question is about an interaction or moderation effect           |
| interpret   | row local | the question asks how to interpret output a statistics program made |
| ownstudy    | author    | the question describes data that the asker collected themselves     |
| econ        | field     | the question is about data from economics or finance                |
| survey      | field     | the question is about data from a questionnaire or survey of people |
| biomed      | field     | the question is about data from biology or medicine                 |
| we          | control   | the question is written in the first person plural, using we or our |

#### Result

##### Cora

BLUF: links and embeddings beat embeddings alone on 2 of the 6 real predicates at a 90% target and on 3 at 95%, and which ones changes with the target.

Calls to reach 90% recall as a share of the rows, mean of 50 seeds, from `results/cora/predicates.txt`:

| predicate    | rate | random | embeddings | links | links+embeddings |
| ------------ | ---- | ------ | ---------- | ----- | ---------------- |
| robotics     | 4.3% | 89.4%  | 12.3%      | 74.6% | 14.2%            |
| markov       | 3.6% | 89.1%  | 18.5%      | 74.9% | 13.7%            |
| language     | 3.2% | 89.0%  | 39.3%      | 73.5% | 31.0%            |
| unsupervised | 3.6% | 88.8%  | 47.9%      | 81.2% | 45.3%            |
| biology      | 4.8% | 90.1%  | 64.1%      | 83.0% | 64.1%            |
| proof        | 6.6% | 90.0%  | 74.7%      | 81.8% | 75.0%            |
| first person | 2.8% | 88.6%  | 90.7%      | 86.6% | 90.7%            |

Links and embeddings minus embeddings at each target, from the same file, so negative means links help:

| predicate    | 50%  | 80%  | 90%  | 95%   |
| ------------ | ---- | ---- | ---- | ----- |
| robotics     | +1.7 | +1.4 | +1.9 | -11.5 |
| markov       | +0.5 | +0.1 | -4.8 | -0.2  |
| language     | -1.0 | -3.1 | -8.3 | -4.4  |
| unsupervised | -1.8 | -7.0 | -2.6 | -3.0  |
| biology      | -5.1 | +1.0 | 0.0  | -8.0  |
| proof        | +0.3 | +2.6 | +0.3 | +0.5  |
| first person | -0.4 | +0.3 | 0.0  | 0.0   |

##### Stack

BLUF: links and embeddings beat embeddings alone on 3 of the 10 real predicates on postLinks, assumptions, rcode and econ, and on none of them at every target, and putting the two signals on one scale never beats embeddings on postLinks and beats it only on the control on same author.

Calls to reach 90% recall as a share of the rows, postLinks edges, pool with junk and repeated texts dropped, from `results/stack/predicates.txt`:

| predicate   | rate  | random | embeddings | links | links+embeddings |
| ----------- | ----- | ------ | ---------- | ----- | ---------------- |
| spss        | 3.7%  | 89.8%  | 33.8%      | 88.8% | 34.2%            |
| rcode       | 12.5% | 89.8%  | 38.2%      | 85.6% | 37.9%            |
| choose      | 13.7% | 90.1%  | 38.5%      | 83.5% | 41.0%            |
| assumptions | 8.7%  | 90.0%  | 54.7%      | 83.0% | 50.7%            |
| interaction | 3.2%  | 89.3%  | 14.8%      | 84.2% | 14.4%            |
| interpret   | 10.4% | 89.6%  | 30.5%      | 84.0% | 32.5%            |
| ownstudy    | 5.8%  | 90.0%  | 62.4%      | 88.9% | 61.9%            |
| econ        | 7.0%  | 89.5%  | 60.0%      | 87.8% | 59.4%            |
| survey      | 9.3%  | 89.7%  | 40.0%      | 84.0% | 39.8%            |
| biomed      | 11.0% | 89.9%  | 51.8%      | 88.2% | 53.8%            |
| we          | 2.7%  | 88.8%  | 87.9%      | 88.3% | 88.2%            |

Links and embeddings minus embeddings at each target, so negative means links help:

| predicate   | 50%  | 80%  | 90%  | 95%  |
| ----------- | ---- | ---- | ---- | ---- |
| spss        | +2.5 | +2.5 | +0.4 | -1.2 |
| rcode       | +3.1 | +4.2 | -0.3 | -6.7 |
| choose      | +2.5 | +1.0 | +2.5 | +1.1 |
| assumptions | +0.6 | -5.4 | -4.0 | 0.0  |
| interaction | +0.7 | +0.3 | -0.4 | -0.3 |
| interpret   | +2.2 | +2.7 | +2.0 | +1.0 |
| ownstudy    | +1.5 | -0.5 | -0.5 | +0.3 |
| econ        | +0.2 | -3.3 | -0.6 | -2.7 |
| survey      | +1.4 | +0.6 | -0.2 | +1.7 |
| biomed      | +1.7 | +1.8 | +2.0 | +0.4 |
| we          | +1.7 | +0.3 | +0.3 | +0.3 |

Links alone never beats embeddings on any predicate or target.
`blend 0.1`, the best of the three blends and the smallest a in the grid, loses in 43 of the 44 postLinks cells, by +0.1 to +42.9 points.

Same author, from `results/stack_author/predicates.txt`:

| predicate   | 50%  | 80%  | 90%  | 95%  |
| ----------- | ---- | ---- | ---- | ---- |
| spss        | +0.8 | +0.6 | -1.1 | -1.9 |
| rcode       | +1.6 | +0.6 | -0.1 | -3.9 |
| choose      | +3.6 | +3.3 | +3.2 | +2.8 |
| assumptions | +3.1 | -1.5 | +1.8 | +1.7 |
| interaction | +2.7 | +3.4 | +3.4 | +2.9 |
| interpret   | +4.0 | +3.9 | +3.9 | +3.3 |
| ownstudy    | +0.3 | -0.5 | -1.2 | +0.6 |
| econ        | -1.7 | -3.2 | -0.1 | -1.5 |
| survey      | +1.7 | +2.0 | +1.6 | +0.8 |
| biomed      | +0.1 | +0.2 | +0.1 | -0.6 |
| we          | +1.0 | +0.2 | +0.5 | +0.2 |

econ is the only predicate negative at every target, and it clears one sd at 80 and 95%.
The other entries that clear one sd are ownstudy at 90% and spss and rcode at 95%.

On same author `blend 0.1` clears one sd only on we at 80 and 90% and on biomed at 50%, so the predicate it helps most is the control.

### 3. How much the links correlate the positives

Step `lift`

Of the edges leaving a positive the share that land on a positive over the positive rate.
1.0x is no correlation.
Also the share of positives with at least one positive neighbour, which grows with degree, and the share with no neighbour at all, which links can never reach.

#### Result

##### Cora

BLUF: positives sit next to positives on all 6 real predicates, 4.5 to 12.1x above the base rate, and the control is lower but not absent at 3.5x.

From `results/cora/lift.txt`, the pool with junk and repeated texts dropped:

| predicate    | rate | edge% | lift  | nbr+  | alone |
| ------------ | ---- | ----- | ----- | ----- | ----- |
| language     | 3.2% | 39.1% | 12.1x | 63.9% | 1.4%  |
| markov       | 3.6% | 32.6% | 9.0x  | 63.0% | 0.0%  |
| robotics     | 4.3% | 38.6% | 8.9x  | 69.1% | 4.1%  |
| unsupervised | 3.6% | 28.2% | 7.8x  | 46.9% | 2.5%  |
| biology      | 4.8% | 36.3% | 7.5x  | 50.9% | 4.6%  |
| proof        | 6.6% | 29.7% | 4.5x  | 56.1% | 2.7%  |
| first person | 2.8% | 9.7%  | 3.5x  | 25.8% | 6.5%  |

##### Stack

BLUF: the control sits mid table on postLinks at 2.2x and only interaction, assumptions and survey clear it, while on same author the control is 1.0x and the field and tool predicates correlate but 44 to 65% of positives have no neighbour at all.

From `results/stack/lift.txt` and `results/stack_author/lift.txt`, the pool with junk and repeated texts dropped, sorted by postLinks:

| predicate   | rate  | postLinks | author | alone, postLinks | alone, author |
| ----------- | ----- | --------- | ------ | ---------------- | ------------- |
| interaction | 3.2%  | 10.2x     | 3.0x   | 6.3%             | 58.9%         |
| assumptions | 8.7%  | 4.8x      | 2.2x   | 2.8%             | 54.1%         |
| survey      | 9.3%  | 4.1x      | 3.7x   | 1.8%             | 59.4%         |
| interpret   | 10.4% | 3.4x      | 1.6x   | 2.9%             | 57.1%         |
| choose      | 13.7% | 2.8x      | 1.2x   | 3.0%             | 59.9%         |
| rcode       | 12.5% | 2.4x      | 3.3x   | 2.0%             | 44.5%         |
| spss        | 3.7%  | 2.3x      | 4.1x   | 2.8%             | 61.9%         |
| econ        | 7.0%  | 2.3x      | 3.2x   | 5.5%             | 60.3%         |
| we          | 2.7%  | 2.2x      | 1.0x   | 3.1%             | 58.8%         |
| biomed      | 11.0% | 1.8x      | 2.7x   | 2.6%             | 55.7%         |
| ownstudy    | 5.8%  | 1.6x      | 2.4x   | 3.8%             | 64.8%         |

Six of the eleven postLinks lifts are at or below the control.
Cora's weakest real predicate is 4.5x against a 3.5x control, so only interaction stands clear of its own control by that margin.
Lift does not order the step 2 winners.
interaction has the highest lift here and gains nothing, and rcode and econ win there while sitting at the control.

### 4. Oracle calls under a proxy

Step `proxy` and then step `cascade`.

- `proxy` gets p(yes) on every row and predicate from the small model.
- `cascade` replays the proxy order and the proxy with link expansion on top, and makes no oracle calls.

Both orders pay one small-model call per row, so the column below is oracle calls alone.

#### Result

##### Cora

BLUF: the proxy beats every order in step 2, and link expansion on top of it is worse on every predicate except language at a 95% target.

Calls to reach 90% recall as a share of the rows, from `results/cora/cascade.txt`, the pool with junk and repeated texts dropped:

| predicate    | rate | proxy | proxy+links |
| ------------ | ---- | ----- | ----------- |
| language     | 3.2% | 3.6%  | 8.2%        |
| markov       | 3.6% | 5.5%  | 9.1%        |
| robotics     | 4.3% | 5.5%  | 9.4%        |
| unsupervised | 3.6% | 9.2%  | 9.2%        |
| first person | 2.8% | 10.3% | 14.8%       |
| proof        | 6.6% | 11.8% | 20.0%       |
| biology      | 4.8% | 13.5% | 15.7%       |

proxy+links minus proxy at each target, from the same file:

| predicate    | 50%  | 80%  | 90%  | 95%  |
| ------------ | ---- | ---- | ---- | ---- |
| language     | +1.6 | +4.4 | +4.6 | -2.2 |
| markov       | +2.9 | +4.1 | +3.6 | +4.3 |
| robotics     | +2.0 | +3.3 | +3.9 | +4.0 |
| unsupervised | +1.7 | +3.0 | 0.0  | +1.2 |
| first person | +6.2 | +6.9 | +4.5 | +1.4 |
| proof        | +6.9 | +9.4 | +8.2 | +7.1 |
| biology      | +3.8 | +3.8 | +2.2 | +3.5 |
