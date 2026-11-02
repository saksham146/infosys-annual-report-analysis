# Infosys Report Analyst

A Streamlit RAG assistant for asking evidence-based questions about the Infosys Integrated Annual Report 2023–24. It retrieves relevant passages from a local ChromaDB index and streams a Gemini answer with report page references.

## Features

- Page-aware PDF extraction and overlapping passage chunking
- Local `all-MiniLM-L6-v2` embeddings with a persistent ChromaDB index
- Gemini streaming answers grounded in retrieved passages
- Agent status, evidence count, suggested questions, and source page display
- Local credentials and report data kept out of Git

## Setup

Use Python 3.11 or newer. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the annual report PDF at `Data/annual_report.pdf`. The report and generated vector database are intentionally ignored by Git. Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key
```

Alternatively, configure `GEMINI_API_KEY` in Streamlit secrets. Optional model settings are `GEMINI_MODEL` and `EMBEDDING_MODEL`.

Build the local report index (downloads the embedding model on first use):

```powershell
python main.py
```

To replace the existing index after embeddings are ready:

```powershell
python main.py --rebuild
```

Run the agent:

```powershell
python -m streamlit run app.py
```

## GitHub and deployment

Keep `.env`, `key.env`, `Data/annual_report.pdf`, `Data/chroma_db/`, `.venv/`, and `.streamlit/secrets.toml` private. For Streamlit Community Cloud, add `GEMINI_API_KEY` under the app's Secrets settings. Build or provide the report index as part of deployment setup; the local ChromaDB directory is not committed.

## Project layout

- `app.py` — Streamlit analyst interface
- `src/ingest.py` — PDF text extraction and chunking
- `src/Embed.py` — local embedding generation
- `src/Vector_store.py` — ChromaDB index build/update command
- `src/rag_pipeline.py` — retrieval, prompt construction, and Gemini streaming
- `main.py` — report index CLI

## Project manual

[Download the user and developer manual](output/pdf/Infosys_Annual_Report_Analyst_Manual.pdf) for the full setup guide, agent diagrams, and file-by-file walkthrough.
