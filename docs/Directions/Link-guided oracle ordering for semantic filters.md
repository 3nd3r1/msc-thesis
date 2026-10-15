# Link-guided oracle ordering for semantic filters

A semantic filter with a recall target can find its positives in fewer oracle calls by following links out of the rows it has already confirmed.

## Angle

Under a recall target the cost is set by the order in which rows go to the oracle.
LOTUS [@lotus] and BARGAIN [@bargain] order rows by a small model's score of each row's own text.
Many tables also store links between rows, and positives tend to link to positives, so a confirmed positive says something about its neighbours that their own text does not.
Open is which predicates and edge types this adds anything on over embeddings and the proxy, and how a system can tell before it pays for the calls.

## Gap

CSV [@csv] and adaptive Two-Phase [@kimFastLLMBasedSemantic2026] group rows by embedding and take an accuracy target, which a rare predicate meets by answering no to every row.
PaSa [@pasa] follows citations with an LLM judge, but as an agent on the open web, with no fixed table and no recall target.
I have not found a semantic operator that uses links between rows.

## Research questions

1. Which predicates correlate along which edge types?
2. When they do, does link-guided ordering in LOTUS `sem_filter` reach a recall target in fewer oracle calls than embedding or proxy ordering?
3. Can a cheap check tell whether links will help, and fall back to the default order when they will not?

## Evaluation

Offline replay on fully labelled data, in [03](../../experiments/03_link_guided_search).

## Risks

- The proxy already ranks well. On Cora, link expansion on top of it lost on 6 of the 7 predicates, step 4 of 03.
- Only some predicates correlate along links, and I cannot yet predict which.

## References

- lotus - LOTUS, `sem_filter` and the proxy ordering this extends
- bargain - BARGAIN, proxy scores with recall guarantees
- csv - CSV, embedding clusters and an accuracy target
- kimFastLLMBasedSemantic2026 - adaptive Two-Phase, clustering then a trained proxy
- pasa - PaSa, follows citations with an LLM judge
