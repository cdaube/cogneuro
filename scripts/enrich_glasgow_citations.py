"""
Enrich Glasgow researcher abstracts with citation counts from OpenAlex.

Mirrors scripts/enrich_citations.py but targets data/glasgow_abstracts.csv.

Usage:
    uv run python scripts/enrich_glasgow_citations.py
"""

import pandas as pd
import requests
import time
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "glasgow_citations.csv")

OPENALEX_BASE = "https://api.openalex.org/works"
BATCH_SIZE = 50
POLITE_EMAIL = "christoph.daube@gmail.com"


def fetch_citations_batch(pmids: list[str]) -> dict[str, int]:
    """Fetch citation counts for a batch of PMIDs from OpenAlex."""
    filter_str = "|".join(pmids)
    params = {
        "filter": f"ids.pmid:{filter_str}",
        "select": "ids,cited_by_count",
        "per_page": BATCH_SIZE,
        "mailto": POLITE_EMAIL,
    }

    resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = {}
    for work in data.get("results", []):
        pmid_url = work.get("ids", {}).get("pmid", "")
        pmid = pmid_url.rstrip("/").split("/")[-1] if pmid_url else None
        if pmid:
            results[pmid] = work.get("cited_by_count", 0)

    return results


def enrich():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} abstracts from {INPUT_CSV}")

    pmids = df["pmid"].astype(str).tolist()

    # Resume support
    already_done = set()
    existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        already_done = set(existing["pmid"].astype(str))
        existing_rows = existing.to_dict("records")
        print(f"Resuming: {len(already_done)} PMIDs already fetched.")

    remaining = [p for p in pmids if p not in already_done]
    print(f"Fetching citations for {len(remaining)} PMIDs...")

    all_results = list(existing_rows)
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    for batch in tqdm(batches, desc="OpenAlex"):
        for attempt in range(3):
            try:
                citations = fetch_citations_batch(batch)
                for pmid in batch:
                    all_results.append({
                        "pmid": pmid,
                        "cited_by_count": citations.get(pmid, 0),
                    })
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if len(all_results) % (BATCH_SIZE * 100) < BATCH_SIZE:
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

        time.sleep(0.1)

    out = pd.DataFrame(all_results)
    out.drop_duplicates(subset="pmid", inplace=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Citation data for {len(out)} papers saved to {OUTPUT_CSV}")
    print(f"Mean citations: {out['cited_by_count'].mean():.1f}")
    print(f"Median citations: {out['cited_by_count'].median():.0f}")
    print(f"Max citations: {out['cited_by_count'].max()}")


if __name__ == "__main__":
    enrich()
