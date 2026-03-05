"""Pydantic models for request / response validation."""

from pydantic import BaseModel, Field


# Maximum characters accepted in a single extraction request.
MAX_TEXT_LENGTH = 15_000


# ── Request ──────────────────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    """Body accepted by POST /extract."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Raw text of the invoice or document to analyse.",
        json_schema_extra={
            "example": (
                "Invoice #104122\n"
                "Date: 2026-03-02\n"
                "From: Flowers Corp\n"
                "Total: $1,250.00"
            )
        },
    )


# ── Extracted fields ─────────────────────────────────────────────────────────

class ExtractedData(BaseModel):
    """Structured fields returned by the language model."""

    company: str = Field(default="", description="Name of the issuing company.")
    invoice_number: str = Field(default="", description="Invoice identifier.")
    date: str = Field(default="", description="Invoice date (as written in the document).")
    total_amount: str = Field(default="", description="Total amount including currency symbol.")


# ── Response ─────────────────────────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    """Successful extraction response."""

    data: ExtractedData
    model_used: str = Field(..., description="OpenAI model that produced the result.")
    usage_tokens: int = Field(..., ge=0, description="Total tokens consumed by the request.")


class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str
