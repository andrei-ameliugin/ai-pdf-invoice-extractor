"""FastAPI application — document data extraction service."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import configure_logging, get_settings
from extractor import ExtractionError, extract_fields
from pdf_utils import extract_text_from_pdf
from schemas import (
    ErrorResponse,
    ExtractionRequest,
    ExtractionResponse,
)

logger = logging.getLogger(__name__)

# ── Lifespan — shared async HTTP client ──────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[dict]:
    """Create and tear down a shared httpx.AsyncClient."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Starting AI Document Extractor (model=%s, timeout=%.0fs)",
        settings.openai_model,
        settings.openai_timeout,
    )

    async with httpx.AsyncClient() as client:
        yield {"http_client": client}

    logger.info("Shutting down AI Document Extractor")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Document Extractor",
    description=(
        "Extract structured invoice data from raw text using the "
        "OpenAI Responses API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Exception handlers ──────────────────────────────────────────────────────


@app.exception_handler(ExtractionError)
async def extraction_error_handler(
    _request: Request,
    exc: ExtractionError,
) -> JSONResponse:
    """Return a clean JSON error for any extraction failure."""
    logger.warning("Extraction error: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.message).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a 422 with human-readable validation details."""
    errors = exc.errors()
    messages = [
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    ]
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail="; ".join(messages)).model_dump(),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        502: {"model": ErrorResponse, "description": "Upstream API failure"},
    },
    summary="Extract structured data from a document",
)
async def extract(body: ExtractionRequest, request: Request) -> ExtractionResponse:
    """Accept raw document text and return structured invoice fields."""
    settings = get_settings()
    client: httpx.AsyncClient = request.state.http_client

    data, model_used, total_tokens = await extract_fields(
        text=body.text,
        settings=settings,
        client=client,
    )

    return ExtractionResponse(
        data=data,
        model_used=model_used,
        usage_tokens=total_tokens,
    )


@app.post(
    "/extract-pdf",
    response_model=ExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or empty PDF"},
        502: {"model": ErrorResponse, "description": "Upstream API failure"},
    },
    summary="Extract structured data from a PDF invoice",
)
async def extract_pdf(file: UploadFile, request: Request) -> ExtractionResponse:
    """Accept a PDF file upload and return structured invoice fields."""
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Expected a PDF file, got '{file.content_type}'.",
        )

    text = await extract_text_from_pdf(file)

    settings = get_settings()
    client: httpx.AsyncClient = request.state.http_client

    data, model_used, total_tokens = await extract_fields(
        text=text,
        settings=settings,
        client=client,
    )

    return ExtractionResponse(
        data=data,
        model_used=model_used,
        usage_tokens=total_tokens,
    )
