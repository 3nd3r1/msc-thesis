import os
import sys

from openai import OpenAI


BASE_URL = "https://api.deepinfra.com/v1/openai"
ORACLE = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # Similar to Lotus
PROXY = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


def client():
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        sys.exit("set DEEPINFRA_API_KEY, see .env.example")
    return OpenAI(base_url=BASE_URL, api_key=key)
