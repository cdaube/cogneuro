"""
Enrich scraped PubMed abstracts with author institution data from OpenAlex.

For each paper we store every unique institution name (semicolon-separated).
Results are saved to data/institutions.csv.

Usage:
    uv run python scripts/enrich_institutions.py
"""

import pandas as pd
import requests
import time
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_CSV = os.path.join(DATA_DIR, "neuro_abstracts.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "institutions.csv")

OPENALEX_BASE = "https://api.openalex.org/works"
BATCH_SIZE = 50
POLITE_EMAIL = "christoph.daube@gmail.com"


def fetch_institutions_batch(pmids: list[str]) -> dict[str, str]:
    """Fetch institution names for a batch of PMIDs from OpenAlex.

    Returns {pmid: "Inst A; Inst B; ..."} for each found paper.
    """
    filter_str = "|".join(pmids)
    params = {
        "filter": f"ids.pmid:{filter_str}",
        "select": "ids,authorships",
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
        if not pmid:
            continue

        # Collect unique institution display names across all authors
        institutions = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name")
                if name:
                    institutions.add(name)

        results[pmid] = "; ".join(sorted(institutions)) if institutions else ""

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
    print(f"Fetching institutions for {len(remaining)} PMIDs...")

    all_results = list(existing_rows)
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="OpenAlex institutions")):
        for attempt in range(3):
            try:
                institutions = fetch_institutions_batch(batch)
                for pmid in batch:
                    all_results.append({
                        "pmid": pmid,
                        "institutions": institutions.get(pmid, ""),
                    })
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # Checkpoint every 100 batches (~5000 papers)
        if (batch_idx + 1) % 100 == 0:
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
            tqdm.write(f"  Checkpointed {len(all_results)} rows.")

        time.sleep(0.1)

    # Final save
    out = pd.DataFrame(all_results)
    out.drop_duplicates(subset="pmid", inplace=True)
    out.to_csv(OUTPUT_CSV, index=False)

    # Quick stats
    n_with = (out["institutions"] != "").sum()
    print(f"\nDone! Institution data for {len(out)} papers saved to {OUTPUT_CSV}")
    print(f"  Papers with ≥1 institution resolved: {n_with} ({100*n_with/len(out):.1f}%)")

    # Check Glasgow specifically
    glasgow_mask = out["institutions"].str.contains("Glasgow", case=False, na=False)
    print(f"  Papers with a Glasgow affiliation: {glasgow_mask.sum()}")


if __name__ == "__main__":
    enrich()
