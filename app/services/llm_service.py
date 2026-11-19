
import logging

from google import genai
from google.genai import types, errors
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from app.config import settings


logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=settings.gemini_api_key)


def is_retryable_error(exc: BaseException) -> bool:
    """Retry only temporary Gemini API errors."""

    if isinstance(exc, errors.APIError):
        return exc.code in (429, 500, 502, 503, 504)

    return False


@retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
@traceable(name="Gemini_LLM_Generate", run_type="llm")
def _generate(
    system_prompt: str,
    user_message: str,
    model: str,
    temperature: float,
    json_mode: bool = False,
) -> dict:
    """Generate text or structured JSON using Gemini."""

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_message,
        config=config,
    )

    text = response.text or ""

    usage_metadata = response.usage_metadata
    prompt_tokens = usage_metadata.prompt_token_count if usage_metadata else 0
    completion_tokens = usage_metadata.candidates_token_count if usage_metadata else 0
    total_tokens = usage_metadata.total_token_count if usage_metadata else 0

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.extra.setdefault("metadata", {}).update(
            {
                "ls_provider": "google_genai",
                "ls_model_name": model,
                "ls_model_type": "chat",
                "ls_temperature": temperature,
                "json_mode": json_mode,
            }
        )

    return {
        "text": text,
        "usage": usage,
        "usage_metadata": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def generate(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Generate a normal text response."""

    if model is None:
        model = settings.llm_model_answer

    return _generate(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        temperature=temperature,
    )


def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Generate a JSON-formatted response."""

    if model is None:
        model = settings.llm_model_grader

    return _generate(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        temperature=temperature,
        json_mode=True,
    )
