
import logging
import re
import threading
import time

from google import genai
from google.genai import errors
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
from app.services.query_cache_service import query_cache


logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=settings.gemini_api_key)

VECTOR_SIZE = 1536
MAX_BATCH_SIZE = 16

# Keep requests comfortably below Gemini's free-tier rate limit.
REQUEST_INTERVAL_SECONDS = 1.2

_request_lock = threading.Lock()
_last_request_time = 0.0


def _pace_request() -> None:
    """Ensure embedding API requests are spaced apart."""
    global _last_request_time

    with _request_lock:
        now = time.monotonic()
        wait_time = (
            _last_request_time + REQUEST_INTERVAL_SECONDS - now
        )

        if wait_time > 0:
            time.sleep(wait_time)

        _last_request_time = time.monotonic()


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in (429, 500, 502, 503, 504)

    return False


def _wait_for_retry(retry_state) -> float:
    """
    Respect Gemini's suggested retry delay when available.
    Otherwise, use exponential backoff.
    """
    exc = retry_state.outcome.exception()

    if isinstance(exc, errors.APIError):
        error_message = str(exc)

        # Match messages such as:
        # "Please retry in 27.963s"
        # 'retryDelay": "27s"'
        delays = re.findall(
            r"retry in\s+([\d.]+)s|retryDelay['\"]?\s*:\s*['\"]([\d.]+)s",
            error_message,
            re.IGNORECASE,
        )

        parsed_delays = [
            float(value)
            for match in delays
            for value in match
            if value
        ]

        if parsed_delays:
            delay = max(parsed_delays) + 1.0
            logger.warning(
                "Gemini requested a retry delay of %.2f seconds",
                delay,
            )
            return delay

    # Fallback for errors without a usable retry delay.
    return wait_exponential(
        multiplier=2,
        min=2,
        max=30,
    )(retry_state)


@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(4),
    wait=_wait_for_retry,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
@traceable(name="Gemini_Embed_Batch", run_type="embedding")
def _embed_batch(
    texts: list[str],
    model: str,
) -> list[list[float]]:

    # Applied to every API attempt, including retries.
    _pace_request()

    response = gemini_client.models.embed_content(
        model=model,
        contents=texts,
        config={"output_dimensionality": VECTOR_SIZE},
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no embeddings")

    if len(response.embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, "
            f"received {len(response.embeddings)}"
        )

    vectors = []

    for item in response.embeddings:
        vector = list(item.values or [])

        if len(vector) != VECTOR_SIZE:
            raise ValueError(
                f"Invalid embedding dimension: "
                f"expected {VECTOR_SIZE}, received {len(vector)}"
            )

        vectors.append(vector)

    return vectors


@traceable(name="Embed_Texts_With_Cache", run_type="embedding")
def embed_texts(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:

    if not texts:
        return []

    model = model or settings.embedding_model

    results: list[list[float] | None] = [None] * len(texts)

    miss_indices = []
    miss_texts = []

    # Retrieve cached embeddings.
    for index, text in enumerate(texts):
        cached = query_cache.get_embedding(text, model)

        if cached is not None and len(cached) == VECTOR_SIZE:
            results[index] = cached
        else:
            miss_indices.append(index)
            miss_texts.append(text)

    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.extra.setdefault("metadata", {}).update(
            {
                "embedding_model": model,
                "total_texts": len(texts),
                "cache_hits": len(texts) - len(miss_texts),
                "cache_misses": len(miss_texts),
            }
        )

    # Generate missing embeddings in controlled batches.
    for start in range(0, len(miss_texts), MAX_BATCH_SIZE):
        batch_texts = miss_texts[
            start:start + MAX_BATCH_SIZE
        ]
        batch_indices = miss_indices[
            start:start + MAX_BATCH_SIZE
        ]

        vectors = _embed_batch(batch_texts, model)

        for original_index, text, vector in zip(
            batch_indices,
            batch_texts,
            vectors,
            strict=True,
        ):
            results[original_index] = vector
            query_cache.set_embedding(text, vector, model)

        logger.info(
            "Embedding progress: %d/%d cache misses processed",
            min(start + len(batch_texts), len(miss_texts)),
            len(miss_texts),
        )

    if any(result is None for result in results):
        raise RuntimeError("Embedding generation incomplete")

    return [
        result
        for result in results
        if result is not None
    ]
