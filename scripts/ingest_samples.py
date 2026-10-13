"""Script to ingest sample documents into the running API.

Usage:
    python scripts/ingest_samples.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description="Ingest sample documents")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL of the running RAG API",
    )
    parser.add_argument(
        "--data-dir", default="data/sample_documents",
        help="Path to sample documents directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    client = httpx.Client(base_url=args.base_url, timeout=60)

    # Check health
    try:
        r = client.get("/health")
        if r.status_code != 200:
            print(f"ERROR: API not healthy: {r.status_code}")
            sys.exit(1)
        print(f"API is healthy: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to API at {args.base_url}: {e}")
        sys.exit(1)

    # Ingest all documents
    print(f"\nIngesting documents from {data_dir}...")
    for filepath in sorted(data_dir.glob("*")):
        if filepath.suffix.lower() in (".md", ".markdown", ".txt", ".pdf"):
            with open(filepath, "rb") as f:
                content = f.read()
            files = {"file": (filepath.name, content, "text/plain")}
            r = client.post("/documents/upload", files=files)
            if r.status_code == 201:
                data = r.json()
                print(f"  [OK] {filepath.name} -> doc_id={data['document_id']} ({data['chunk_count']} chunks)")
            else:
                print(f"  [FAIL] {filepath.name}: {r.status_code} {r.text}")

    # List documents
    r = client.get("/documents")
    if r.status_code == 200:
        data = r.json()
        print(f"\nTotal documents in store: {data['total']}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
