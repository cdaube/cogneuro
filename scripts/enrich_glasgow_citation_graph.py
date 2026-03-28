"""
Build a citation graph among Glasgow researcher papers using OpenAlex.

For each paper in glasgow_abstracts.csv, fetches its references from OpenAlex,
then filters to only keep edges where both citing and cited paper are in our dataset.

Outputs:
  data/glasgow_citation_graph.csv  — edge list (citing_pmid, cited_pmid)

Usage:
    uv run python scripts/enrich_glasgow_citation_graph.py
"""

import pandas as pd
import requests
import time
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "glasgow_abstracts.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "glasgow_citation_graph.csv")
OPENALEX_MAP_CACHE = os.path.join(DATA_DIR, ".glasgow_openalex_map.csv")

OPENALEX_BASE = "https://api.openalex.org/works"
BATCH_SIZE = 50
POLITE_EMAIL = "christoph.daube@gmail.com"


def fetch_batch(pmids: list[str]) -> list[dict]:
    """Fetch OpenAlex IDs and referenced_works for a batch of PMIDs."""
    filter_str = "|".join(pmids)
    params = {
        "filter": f"ids.pmid:{filter_str}",
        "select": "id,ids,referenced_works",
        "per_page": BATCH_SIZE,
        "mailto": POLITE_EMAIL,
    }
    resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for work in data.get("results", []):
        pmid_url = work.get("ids", {}).get("pmid", "")
        pmid = pmid_url.rstrip("/").split("/")[-1] if pmid_url else None
        openalex_id = work.get("id", "")
        refs = work.get("referenced_works", []) or []
        if pmid:
            results.append({
                "pmid": pmid,
                "openalex_id": openalex_id,
                "referenced_works": refs,
            })
    return results


def build_graph():
    df = pd.read_csv(INPUT_CSV)
    pmids = df["pmid"].astype(str).tolist()
    print(f"Loaded {len(pmids)} papers from {INPUT_CSV}")

    # Step 1: Fetch OpenAlex IDs + references for all papers
    if os.path.exists(OPENALEX_MAP_CACHE):
        print("Loading cached OpenAlex mappings...")
        cached = pd.read_csv(OPENALEX_MAP_CACHE)
        already_done = set(cached["pmid"].astype(str))
        all_records = cached.to_dict("records")
        # Parse stringified lists back
        for r in all_records:
            if isinstance(r["referenced_works"], str):
                r["referenced_works"] = r["referenced_works"].split("|") if r["referenced_works"] else []
    else:
        already_done = set()
        all_records = []

    remaining = [p for p in pmids if p not in already_done]
    print(f"Fetching OpenAlex data for {len(remaining)} papers...")

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    for batch in tqdm(batches, desc="OpenAlex refs"):
        for attempt in range(3):
            try:
                results = fetch_batch(batch)
                all_records.extend(results)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # Checkpoint every 200 batches
        if len(all_records) % (BATCH_SIZE * 200) < BATCH_SIZE:
            _save_cache(all_records)

        time.sleep(0.1)

    _save_cache(all_records)

    # Step 2: Build mapping from OpenAlex ID → PMID (for our papers only)
    oa_to_pmid = {}
    pmid_refs = {}
    for rec in all_records:
        oa_id = rec["openalex_id"]
        pmid = rec["pmid"]
        oa_to_pmid[oa_id] = pmid
        pmid_refs[pmid] = rec["referenced_works"]

    print(f"Mapped {len(oa_to_pmid)} papers to OpenAlex IDs")

    # Step 3: Build edge list — only keep edges where both ends are in our dataset
    our_oa_ids = set(oa_to_pmid.keys())
    edges = []
    for pmid, refs in pmid_refs.items():
        for ref_oa_id in refs:
            if ref_oa_id in our_oa_ids:
                cited_pmid = oa_to_pmid[ref_oa_id]
                if cited_pmid != pmid:  # no self-citations
                    edges.append({"citing_pmid": pmid, "cited_pmid": cited_pmid})

    edge_df = pd.DataFrame(edges).drop_duplicates()
    edge_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(edge_df)} citation edges (within dataset) saved to {OUTPUT_CSV}")

    # Clean up cache
    if os.path.exists(OPENALEX_MAP_CACHE):
        os.remove(OPENALEX_MAP_CACHE)


def _save_cache(records):
    """Save intermediate results with referenced_works as pipe-separated strings."""
    rows = []
    for r in records:
        rows.append({
            "pmid": r["pmid"],
            "openalex_id": r["openalex_id"],
            "referenced_works": "|".join(r["referenced_works"]) if r["referenced_works"] else "",
        })
    pd.DataFrame(rows).to_csv(OPENALEX_MAP_CACHE, index=False)


if __name__ == "__main__":
    build_graph()
