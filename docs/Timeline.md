# Timeline

Written 2026-10-09.
Dates after November are estimates and will move.
The steps themselves are in [Process](Process.md).

## Fixed dates

| Date        | What                                                             |
| ----------- | ---------------------------------------------------------------- |
| Oct 16      | Thesis plan drafted and sent to Lu                               |
| Oct 23      | Stack Exchange test done, decide whether links have a case       |
| Feb         | Seminar presentation, date open                                  |
| Spring 2027 | Submission, exact date TODO, check with the programme            |

## Phases

### Oct 9 to Oct 23, plan and the last exploratory test

- Write the thesis plan around three questions:
  1. Which predicates correlate along links?
  2. Does link-ordered oracle calling save calls in LOTUS `sem_filter` when they do?
  3. Can the system tell cheaply whether links will help, and fall back to embeddings when they will not?
- Stack Exchange, same author and same thread edges, predicates fixed up front, no prompt tuning.
- Start the related work chapter, LOTUS, cascades, CSV, Two-Phase and PaSa.

### Oct 26 to the end of November, lock the design

- Plan agreed with Lu, then the PreThesis and supervision agreement steps.
- Fix the experiment design, datasets, edge types, predicates, baselines and metrics.
- Draft the background chapter.

### Dec to Jan, main experiments

- Run the fixed design. No new datasets or directions unless Lu asks.
- Implement the fallback check from question 3.
- Write the method and setup chapters while the runs go.

### Feb to Mar, seminar, results and writing

- Slides from the results, then the presentation.
- Final runs, figures and tables.
- Results and discussion chapters.
- Full draft to Lu by the end of March.

### Apr to the deadline, revise and submit

- Lu's comments, revision, language check.
- Submit.

## Rules

- After Oct 23 the direction is fixed. New ideas go on a future work list, not into the schedule.
- Every week has some writing in it, even during experiments.
- Weekly three line update to Lu, done, next, blocked.
