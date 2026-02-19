# CogNeuro Figures + Interactive Explorer

## Generate static figures/animation (memory-aware)

```bash
uv run python scripts/make_figures.py
```

On lower-memory machines (e.g. 18 GB), this now auto-switches to lighter settings.

## Generate interactive browser explorer

```bash
uv run python scripts/make_interactive_explorer.py
```

Output:

- `figures/umap_explorer.html`

Features:

- Hover for quick metadata
- Click a point to pin full details + abstract in a side panel
- Links to PubMed (`pmid`) and DOI (when available)

## Open in browser

From the project root:

```bash
uv run python -m http.server 8000
```

Then open:

- `http://localhost:8000/figures/umap_explorer.html`

This uses only cached local files in `data/` (no scraping, no re-embedding).
