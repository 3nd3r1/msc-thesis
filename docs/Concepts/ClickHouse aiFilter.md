# ClickHouse aiFilter

A semantic filter built into ClickHouse as a SQL function, `aiFilter(text, condition[, params])`, returning 1 or 0 so it can sit in `WHERE`, `PREWHERE` or `JOIN ... ON`.
Added in 26.8, beta, announced 2026-09-11, in private preview on ClickHouse Cloud.
It belongs to a family of AI functions with `aiClassify`, `aiExtract`, `aiGenerate`, `aiTranslate`, `aiRedact`, `aiEmbed` and `aiSimilarity`.

Example from the docs: `SELECT * FROM reviews WHERE aiFilter(body, 'the customer is angry about shipping')`.

## How it works

- One HTTP request to the provider per row, so a call budget is a row budget. Only the embedding functions batch.
- The model is told to answer `true` or `false`, temperature 0 by default. Anything other than `true` maps to 0, so unparseable replies and refusals drop the row.
- No probability or confidence comes back, only the 0 or 1.
- Providers are OpenAI and Anthropic, and any OpenAI compatible endpoint such as vLLM or Ollama.
- Requests run from a server wide thread pool with several in flight per query thread, and transient failures are retried.
- A join with `aiFilter` in `ON` calls the LLM once per candidate pair.

## Cost controls

- Per query caps on API calls, input tokens and output tokens.
- By default hitting a cap aborts the query. With `ai_function_throw_on_quota_exceeded = 0` the rows past the cap get 0 and the query succeeds.
- The blog's advice is to run on a `LIMIT 100` sample first and to put cheap predicates in a subquery before the LLM predicate.

## What it does not do

- No cascade, no proxy model, no sampling and no recall or precision guarantee.
- No caching or deduplication of identical inputs inside ClickHouse.
- The docs warn to use it only where false positives and false negatives are acceptable.

## Relevance

- It is the naive plan the LOTUS, SUPG and BARGAIN line of work optimises, one oracle call per row, now shipped in a production database.
- With the soft cap the query answers from whichever rows were evaluated before the budget ran out, so the order rows are fed in decides the recall at a fixed budget. That is the budgeted setting where the link order of 03 helps.
- A refusal maps to 0, the same silent drop that the 8B's refusals on CAD and MELD would cause.

Sources: https://clickhouse.com/docs/sql-reference/functions/ai-functions and https://clickhouse.com/blog/ai-functions-in-clickhouse, read 2026-11-18.
