"""
PubMed scraper for neuroimaging abstracts (1980-2025).

Strategy:
  1. Use esearch to get ALL matching PMIDs (paginated in chunks of 10k)
  2. Use efetch to retrieve article details by PMID (batches of 200)

This avoids the WebEnv session-expiry problem that plagues usehistory+retstart.

Usage:
    uv run python scripts/scrape_pubmed.py
"""

from Bio import Entrez
import pandas as pd
from tqdm import tqdm
import time
import os
import json

# -- Configuration --
Entrez.email = "christoph.daube@gmail.com"
Entrez.api_key = "83596db810992244490dd34f950166198307"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "neuro_abstracts.csv")
PMID_CACHE = os.path.join(DATA_DIR, ".pmid_cache.json")
FETCH_BATCH = 200
SEARCH_BATCH = 10000
SAVE_EVERY = 5000
MAX_RETRIES = 5

# -- Journal filter --
JOURNALS = [
    "Neuron",
    "Science",
    "Nature",
    "Nat Neurosci",
    "Nat Commun",
    "J Neurosci",
    "Elife",
    "PLoS Biol",
    "Proc Natl Acad Sci U S A",
    "Neuroimage",
    "PLoS Comput Biol",
    "Nat Hum Behav",
    "Hum Brain Mapp",
    "Imaging Neurosci",
    "Brain",
    "Cereb Cortex",
    "J Neurophysiol",
    "Magn Reson Med",
    "Trends Cogn Sci",
    "J Cogn Neurosci",
    "Trends Neurosci",
    "Curr Biol",
    "Cell Rep",
    "Cortex",
    "Sci Adv",
    "Psychophysiology",
    "Netw Neurosci",
    "J Neurosci Methods",
    "Neuroinformatics",
]

MODALITY_TERMS = [
    '"Magnetic Resonance Imaging"[MeSH]',
    '"functional MRI"',
    '"fMRI"',
    '"Diffusion Tensor Imaging"[MeSH]',
    '"DTI"',
    '"Electroencephalography"[MeSH]',
    '"EEG"',
    '"Magnetoencephalography"[MeSH]',
    '"MEG"',
    '"Electrocorticography"[MeSH]',
    '"ECoG"',
    '"Spectroscopy, Near-Infrared"[MeSH]',
    '"fNIRS"',
    '"Transcranial Magnetic Stimulation"[MeSH]',
    '"TMS"',
    '"Transcranial Direct Current Stimulation"[MeSH]',
    '"tDCS"',
    '"Transcranial Alternating Current Stimulation"',
    '"tACS"',
    '"Local Field Potentials"[MeSH]',
    '"LFP"',
    '"stereo-EEG"',
    '"sEEG"',
    '"simultaneous EEG-fMRI"',
    '"simultaneous fMRI-EEG"',
]


def build_query():
    """Build the full PubMed query string."""
    modalities = " OR ".join(MODALITY_TERMS)
    journals = " OR ".join(f'"{j}"[ta]' for j in JOURNALS)
    query = (
        f"({modalities}) "
        f'AND "Humans"[MeSH] '
        f"AND ({journals}) "
        f'AND ("1980/01/01"[Date - Publication] : "2025/12/31"[Date - Publication])'
    )
    return query


def get_all_pmids(query):
    """Retrieve all PMIDs by searching year-by-year to stay under the 10k retstart limit."""
    if os.path.exists(PMID_CACHE):
        with open(PMID_CACHE) as f:
            pmids = json.load(f)
        print(f"Loaded {len(pmids)} cached PMIDs.")
        return pmids

    # Strip the existing date filter from the query so we can add per-year filters
    base_query = query.split(' AND ("1980')[0]

    all_pmids = []
    for year in tqdm(range(1980, 2026), desc="Fetching PMIDs by year"):
        year_query = (
            f'{base_query} AND ("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication])'
        )
        for attempt in range(MAX_RETRIES):
            try:
                handle = Entrez.esearch(
                    db="pubmed", term=year_query,
                    retmax=9999,
                )
                results = Entrez.read(handle)
                handle.close()
                year_count = int(results["Count"])
                pmids = results["IdList"]
                if year_count > 9999:
                    tqdm.write(f"  WARNING: {year} has {year_count} results (capped at 9999). Some may be missed.")
                all_pmids.extend(pmids)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Search error for {year}: {e}. Retry in {wait}s...")
                time.sleep(wait)
        time.sleep(0.11)  # Rate limiting

    all_pmids = list(dict.fromkeys(all_pmids))
    print(f"Retrieved {len(all_pmids)} unique PMIDs across all years.")

    with open(PMID_CACHE, "w") as f:
        json.dump(all_pmids, f)

    return all_pmids


def extract_article_data(article):
    """Extract relevant fields from a PubMed article record."""
    try:
        medline = article["MedlineCitation"]
        art = medline["Article"]

        title = str(art.get("ArticleTitle", ""))

        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
        if not abstract_parts:
            return None
        abstract = " ".join(str(p) for p in abstract_parts)

        pub_date = art["Journal"]["JournalIssue"]["PubDate"]
        year = pub_date.get("Year") or pub_date.get("MedlineDate", "Unknown")[:4]

        journal = art["Journal"].get("Title", "Unknown")
        pmid = str(medline["PMID"])

        mesh_list = medline.get("MeshHeadingList", [])
        mesh_terms = []
        for mh in mesh_list:
            descriptor = mh.get("DescriptorName", "")
            mesh_terms.append(str(descriptor))
        mesh_str = "; ".join(mesh_terms)

        doi = ""
        for eid in art.get("ELocationID", []):
            if str(eid.attributes.get("EIdType", "")) == "doi":
                doi = str(eid)
                break

        return {
            "pmid": pmid,
            "year": year,
            "journal": journal,
            "title": title,
            "abstract": abstract,
            "mesh_terms": mesh_str,
            "doi": doi,
        }
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def fetch_articles_by_pmid(pmids):
    """Fetch article details for a list of PMIDs."""
    id_str = ",".join(pmids)
    handle = Entrez.efetch(
        db="pubmed",
        id=id_str,
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()

    results = []
    for article in records.get("PubmedArticle", []):
        row = extract_article_data(article)
        if row:
            results.append(row)
    return results


def scrape():
    query = build_query()
    print(f"Query (first 200 chars):\n{query[:200]}...\n")

    # Step 1: Get all PMIDs
    all_pmids = get_all_pmids(query)

    # Step 2: Check which PMIDs we already fetched
    already_fetched = set()
    all_data = []
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        already_fetched = set(existing["pmid"].astype(str))
        all_data = existing.to_dict("records")
        print(f"Resuming: {len(already_fetched)} articles already fetched.")

    remaining = [p for p in all_pmids if p not in already_fetched]
    print(f"Remaining to fetch: {len(remaining)}")

    if not remaining:
        print("All articles already fetched!")
        return

    # Step 3: Fetch in batches by PMID
    batches = [remaining[i:i + FETCH_BATCH] for i in range(0, len(remaining), FETCH_BATCH)]

    pbar = tqdm(batches, desc="Fetching articles")
    for i, batch in enumerate(pbar):
        for attempt in range(MAX_RETRIES):
            try:
                articles = fetch_articles_by_pmid(batch)
                all_data.extend(articles)
                pbar.set_postfix(total=len(all_data))
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                tqdm.write(f"  Error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    tqdm.write(f"  FAILED batch {i}. Skipping.")

        # Periodic save
        if (i + 1) % (SAVE_EVERY // FETCH_BATCH) == 0:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset="pmid", inplace=True)
            df.to_csv(OUTPUT_CSV, index=False)
            tqdm.write(f"  Checkpoint: {len(df)} unique abstracts saved.")

    # Final save
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset="pmid", inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! {len(df)} unique abstracts saved to {OUTPUT_CSV}")

    # Clean up PMID cache
    if os.path.exists(PMID_CACHE):
        os.remove(PMID_CACHE)


if __name__ == "__main__":
    scrape()
