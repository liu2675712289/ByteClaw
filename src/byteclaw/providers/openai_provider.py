"""OpenAI-backed chat model factory."""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI


def create_model() -> ChatOpenAI:
    """Create a ChatOpenAI model using values loaded from ``.env``."""

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing; add it to .env or the environment")

    return ChatOpenAI(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
