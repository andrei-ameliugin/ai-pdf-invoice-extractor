"""Utility helpers for extracting text from PDF files."""

import io
import logging

import pdfplumber
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


async def extract_text_from_pdf(file: UploadFile) -> str:
    """Read an uploaded PDF and return the combined text of all pages.

    Raises:
        HTTPException 400: If the file cannot be read as a PDF or contains
            no extractable text.
    """
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pdf = pdfplumber.open(io.BytesIO(contents))
    except Exception as exc:
        logger.error("Failed to open PDF: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        ) from exc

    pages_text: list[str] = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    pdf.close()

    combined = "\n".join(pages_text).strip()

    if not combined:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no extractable text.",
        )

    logger.info(
        "Extracted text from PDF (%d pages, %d characters)",
        len(pdf.pages),
        len(combined),
    )

    return combined
