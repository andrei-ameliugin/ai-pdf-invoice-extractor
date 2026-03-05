"""Async integration with the OpenAI Responses API."""

import json
import logging

import httpx

from config import Settings
from schemas import ExtractedData

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert document parser. "
    "Extract the following fields from the provided document text.\n\n"
    "Return ONLY valid JSON in the following format:\n"
    "{\n"
    '  "company": "",\n'
    '  "invoice_number": "",\n'
    '  "date": "",\n'
    '  "total_amount": ""\n'
    "}\n\n"
    "Fill each field with the value found in the document. "
    "If a field is not found, use an empty string. "
    "Do not include any commentary, explanation, or markdown — return raw JSON only."
)


# ── Custom exceptions ───────────────────────────────────────────────────────

class ExtractionError(Exception):
    """Base exception for the extraction pipeline."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class OpenAIConnectionError(ExtractionError):
    """Raised when the HTTP request to OpenAI fails (network / timeout)."""


class OpenAIAPIError(ExtractionError):
    """Raised when OpenAI returns a non-200 status code."""


class ResponseParsingError(ExtractionError):
    """Raised when the model output cannot be parsed into ExtractedData."""


# ── Core extraction logic ───────────────────────────────────────────────────

async def extract_fields(
    text: str,
    settings: Settings,
    client: httpx.AsyncClient,
) -> tuple[ExtractedData, str, int]:
    """Send *text* to OpenAI and return parsed fields + metadata.

    Returns:
        A tuple of ``(ExtractedData, model_used, total_tokens)``.

    Raises:
        OpenAIConnectionError: On network or timeout failures.
        OpenAIAPIError: When OpenAI returns a non-200 status.
        ResponseParsingError: When the model output is not valid JSON.
    """
    text_preview = text[:80].replace("\n", " ")
    logger.info("Starting extraction (input length=%d, preview='%s…')", len(text), text_preview)

    payload = {
        "model": settings.openai_model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    # ── Call OpenAI ──────────────────────────────────────────────────────
    try:
        response = await client.post(
            settings.openai_api_url,
            json=payload,
            headers=headers,
            timeout=settings.openai_timeout,
        )
    except httpx.TimeoutException as exc:
        logger.error("OpenAI request timed out after %.1fs", settings.openai_timeout)
        raise OpenAIConnectionError(
            f"OpenAI request timed out after {settings.openai_timeout}s"
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("OpenAI request failed: %s", exc)
        raise OpenAIConnectionError(f"OpenAI request failed: {exc}") from exc

    if response.status_code != 200:
        logger.error("OpenAI API error %d: %s", response.status_code, response.text[:200])
        raise OpenAIAPIError(
            f"OpenAI API returned {response.status_code}: {response.text[:200]}",
            status_code=502,
        )

    # ── Parse API response ───────────────────────────────────────────────
    try:
        body = response.json()
        content: str = body["output"][0]["content"][0]["text"]
        model_used: str = body.get("model", settings.openai_model)
        total_tokens: int = body.get("usage", {}).get("total_tokens", 0)
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected OpenAI response structure: %s", exc)
        raise ResponseParsingError(
            f"Unexpected OpenAI response structure: {exc}"
        ) from exc

    logger.debug("Raw model output: %s", content)

    # ── Validate extracted data ──────────────────────────────────────────
    try:
        data = ExtractedData.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse model output as JSON: %s", exc)
        raise ResponseParsingError(
            f"Failed to parse model output as JSON: {exc}"
        ) from exc

    logger.info(
        "Extraction succeeded (model=%s, tokens=%d, company='%s')",
        model_used, total_tokens, data.company,
    )

    return data, model_used, total_tokens
