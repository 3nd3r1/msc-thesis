---
date: 2026-11-09
---

# Picking the edge from the schema

Ask which entity a predicate is a property of, and use the edges that connect rows sharing that entity.
A predicate about the author should agree along author edges, and one about the topic along topic links.
This would let the system decide which links to use, or none, before any oracle call, which is RQ3.

## Where it comes from

NVIDIA's [Auto Ontology](https://github.com/NVIDIA/auto-ontology) is a semantic layer for text-to-SQL agents.
It spends LLM effort once to name the business terms in a database, map them to columns and infer the joins, and a human certifies the result.
Each question is then resolved against that layer instead of the raw schema.
It has a cheap `check_answerable` call that grades whether the layer covers a question before the expensive agent runs, and it checks filter values against the data before trusting the SQL.
The README reports no accuracy numbers.

The two parts that carry over are the pre-flight check and checking the guess against the data.

## First hint in existing results

The predicate kinds in Stack were set before any lift was measured.

| predicate   | kind    | lift postLinks | lift same author |
| ----------- | ------- | -------------- | ---------------- |
| interaction | topical | 13.1x          | 3.1x             |
| assumptions | topical | 5.0x           | 2.3x             |
| ownstudy    | author  | 1.6x           | 2.5x             |
| spss        | lexical | 2.5x           | 4.2x             |
| rcode       | lexical | 2.5x           | 3.4x             |
| econ        | field   | 2.4x           | 3.3x             |
| survey      | field   | 4.5x           | 3.8x             |

From `experiments/03_link_guided_search/results/stack/lift.txt` and `stack_author/lift.txt`, all rows.
The lexical ones are about a tool the asker uses, so they count as author properties.

## Test

1. Give an LLM the predicate and a one line description of each edge type, like reply to parent, same thread, same author or cites.
2. Ask which entity the predicate is about and whether rows sharing it would agree.
3. Score the answer against the measured lift and the calls saved in step 2 of 03, on Stack, CAD and Cora.

No oracle calls are needed, the labels and lift tables exist.
CAD is the best test, with three edge types for each predicate.
The score has to be calls saved and not only lift, since high lift did not become fewer calls on Stack.

The data side is a second check.
After the first few positives, measure how often their neighbours agree and fall back to the default order when they do not.
Metadata only, sample only and both together is a clean comparison for RQ3.

## Open questions

- Is the entity a predicate is about something an LLM gets right, or does it need the human certification step Auto Ontology has?
- Which edges count as an entity, a thread yes, a citation probably not?
- Implication between predicates, like a slur post being abuse, is the ontology's term hierarchy and would let verdicts carry over between predicates, but that is a separate direction.

Related: [Structure outside the row](Structure%20outside%20the%20row.md).
