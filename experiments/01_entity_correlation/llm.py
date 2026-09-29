import math
import os
import sys

from openai import OpenAI


BASE_URL = "https://api.deepinfra.com/v1/openai"
ORACLE = "meta-llama/Meta-Llama-3.1-70B-Instruct"
PROXY = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# LOTUS sem_filter format.
SYSTEM = """The user will provide a claim and some relevant context.
Your job is to determine whether the claim is true for the given context.
Use the following format to provide your answer:
Answer: <Your answer here. The answer should be either True or False>"""

PROMPT = """Context:
[Title]: «{title}»
[Text]: «{text}»

Claim: {predicate}"""

VERDICTS = {"true": "yes", "false": "no"}


def client():
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("set DEEPINFRA_API_KEY, see .env.example")
    return OpenAI(base_url=BASE_URL, api_key=key)


def p_yes(verdict, p):
    return p if verdict == "yes" else 1 - p


def judge(api, model, predicate, row):
    choice = api.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(predicate=predicate, **row)},
        ],
        max_tokens=8,
        temperature=0,
        logprobs=True,
    ).choices[0]

    for t in choice.logprobs.content if choice.logprobs else []:
        word = t.token.strip().lower().strip(".")
        if word in VERDICTS:
            return VERDICTS[word], math.exp(t.logprob)
    return None, None
