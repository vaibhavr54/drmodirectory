# Media Intelligence Pipeline Demo

This project turns the provided architecture diagram into a runnable demo. The Flask app simulates ingestion, preprocessing, tagging, face clustering, and event grouping for uploaded images.

## Features
- Image validation, normalization, and quality scoring.
- Lightweight embedding generation and cosine similarity search.
- Face-like clustering (heuristic) and hourly event grouping.
- Web UI to upload images, review clusters, browse events, and run face search.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open <http://localhost:5000>.

## Notes
- This is a demo implementation designed to be easy to run without heavyweight ML dependencies.
- Embeddings are based on a small RGB histogram, so similarity is approximate.
