import math
import os
import sys

from openai import OpenAI


BASE_URL = "https://api.deepinfra.com/v1/openai"
ORACLE = "meta-llama/Meta-Llama-3.1-70B-Instruct"
PROXY = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

PROMPT = """Claim: {predicate}

Review title: {title}
Review text: {text}

Does the claim hold for this review? Answer with one word, Yes or No."""


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
            {"role": "user", "content": PROMPT.format(predicate=predicate, **row)}
        ],
        max_tokens=8,
        temperature=0,
        logprobs=True,
    ).choices[0]

    for t in choice.logprobs.content if choice.logprobs else []:
        word = t.token.strip().lower().strip(".")
        if word in ("yes", "no"):
            return word, math.exp(t.logprob)
    return None, None
