# Semantic operators

Relational-style operators (filter, join, map, aggregate, top-k) whose condition is written in natural language and evaluated by an LLM. Example: keep reviews where "the review complains about quality". Each evaluation is an LLM call, so cost and latency dominate, and optimization is mostly about making fewer or cheaper calls.
