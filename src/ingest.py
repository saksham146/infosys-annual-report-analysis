from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "Data" / "annual_report.pdf"


def ingest_annual_report(pdf_path=PDF_PATH, chunk_size=700, overlap=100):
    """Extract report text into page-aware, overlapping chunks."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Annual report PDF not found: {pdf_path}")
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size >= 100 and 0 <= overlap < chunk_size.")

    chunks = []
    with pymupdf.open(pdf_path) as document:
        print(f"Reading {pdf_path.name} ({len(document)} pages)…")
        step = chunk_size - overlap
        for page_index, page in enumerate(document):
            text = " ".join(page.get_text("text", sort=True).split())
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    boundary = text.rfind(" ", start + chunk_size // 2, end)
                    if boundary > start:
                        end = boundary
                excerpt = text[start:end].strip()
                if excerpt:
                    chunks.append({"page": page_index + 1, "text": excerpt})
                if end >= len(text):
                    break
                start = max(start + 1, end - overlap)
    print(f"Prepared {len(chunks):,} passages.")
    return chunks
