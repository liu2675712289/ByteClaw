"""OpenAI-backed chat model factory."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI


_PROJECT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_DEFAULT_MODEL = "gpt-4o-mini"


def _load_environment() -> None:
    """Load cwd configuration, then fill missing values from the project."""

    cwd_env = find_dotenv(usecwd=True)
    cwd_env_path = Path(cwd_env).resolve() if cwd_env else None
    if cwd_env_path is not None:
        load_dotenv(cwd_env_path)

    project_env_path = _PROJECT_ENV_FILE.resolve()
    if project_env_path.is_file() and project_env_path != cwd_env_path:
        load_dotenv(project_env_path)


def get_model_name() -> str:
    """Return the configured model name after loading ByteClaw's environment."""

    _load_environment()
    return os.getenv("OPENAI_MODEL", _DEFAULT_MODEL)


def create_model() -> ChatOpenAI:
    """Create a ChatOpenAI model using values loaded from ``.env``."""

    _load_environment()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing; add it to .env or the environment")

    return ChatOpenAI(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", _DEFAULT_MODEL),
        temperature=0,
    )
