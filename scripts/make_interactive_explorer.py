"""
Build an interactive UMAP explorer as a single HTML file.

Features:
- Hover over a point to preview paper metadata
- Click a point to open full paper details + abstract in a side panel
- Uses existing cached files only (no scraping, no embeddings)

Usage:
    uv run python scripts/make_interactive_explorer.py
"""

import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _mesh(row, term):
    mesh = str(row.get("mesh_terms", ""))
    return term.lower() in mesh.lower()


def _kw(row, pattern):
    text = str(row.get("title", "")) + " " + str(row.get("abstract", ""))
    return bool(re.search(pattern, text, re.IGNORECASE))


MODALITY_RULES = [
    (
        "EEG-fMRI",
        lambda r: _kw(
            r, r"simultaneous.*(EEG|fMRI).*(EEG|fMRI)|EEG.fMRI|fMRI.EEG"
        ),
    ),
    (
        "sEEG",
        lambda r: _mesh(r, "stereoelectroencephalography")
        or _kw(r, r"\bsEEG\b|stereo.?EEG|intracranial EEG"),
    ),
    (
        "ECoG",
        lambda r: _mesh(r, "Electrocorticography")
        or _kw(r, r"\bECoG\b|electrocorticogra"),
    ),
    (
        "LFP",
        lambda r: _mesh(r, "Local Field Potential")
        or _kw(r, r"\bLFP\b|local field potential"),
    ),
    (
        "fNIRS",
        lambda r: _mesh(r, "Near-Infrared")
        or _kw(r, r"\bfNIRS\b|functional near.infrared|near.infrared spectroscop"),
    ),
    (
        "MEG",
        lambda r: _mesh(r, "Magnetoencephalography")
        or _kw(r, r"\bMEG\b|magnetoencephalogra"),
    ),
    ("tACS", lambda r: _kw(r, r"\btACS\b|transcranial alternating current")),
    (
        "tDCS",
        lambda r: _mesh(r, "Transcranial Direct Current")
        or _kw(r, r"\btDCS\b|transcranial direct current"),
    ),
    (
        "TMS",
        lambda r: _mesh(r, "Transcranial Magnetic Stimulation")
        or _kw(r, r"\bTMS\b|transcranial magnetic stimulat"),
    ),
    (
        "EEG",
        lambda r: _mesh(r, "Electroencephalography")
        or _kw(r, r"\bEEG\b|electroencephalogra"),
    ),
    (
        "DTI",
        lambda r: _mesh(r, "Diffusion Tensor Imaging")
        or _kw(r, r"\bDTI\b|diffusion tensor|diffusion.weighted|tractograph"),
    ),
    (
        "fMRI",
        lambda r: _kw(
            r,
            r"\bfMRI\b|functional MRI|functional magnetic resonance|BOLD|blood.oxygen.level",
        ),
    ),
    (
        "MRI",
        lambda r: _mesh(r, "Magnetic Resonance Imaging")
        or _kw(r, r"\bMRI\b|magnetic resonance imaging"),
    ),
]


MODALITY_COLORS = {
    "fMRI": "#dc2626",
    "MRI": "#ef4444",
    "fNIRS": "#e85d04",
    "DTI": "#b91c1c",
    "EEG": "#2563eb",
    "MEG": "#60a5fa",
    "sEEG": "#facc15",
    "ECoG": "#fde047",
    "LFP": "#eab308",
    "tDCS": "#16a34a",
    "tACS": "#4ade80",
    "TMS": "#15803d",
    "EEG-fMRI": "#8b5cf6",
    "Other": "#78716c",
}


def assign_modality(row):
    for label, rule in MODALITY_RULES:
        if rule(row):
            return label
    return "Other"


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "neuro_abstracts.csv"))

    coords = np.load(os.path.join(DATA_DIR, "umap_coords.npy"))
    if len(df) != len(coords):
        raise ValueError(
            "Length mismatch between neuro_abstracts.csv and umap_coords.npy."
        )

    df["x"] = coords[:, 0].astype(np.float32)
    df["y"] = coords[:, 1].astype(np.float32)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")

    print("Tagging modalities...")
    df["modality"] = df.apply(assign_modality, axis=1)

    citations_file = os.path.join(DATA_DIR, "citations.csv")
    if os.path.exists(citations_file):
        cit = pd.read_csv(citations_file)
        df = df.merge(cit[["pmid", "cited_by_count"]], on="pmid", how="left")
        df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
    else:
        df["cited_by_count"] = 0

    institutions_file = os.path.join(DATA_DIR, "institutions.csv")
    if os.path.exists(institutions_file):
        inst = pd.read_csv(institutions_file)
        inst = inst[["pmid", "institutions"]].copy()
        inst["pmid_str"] = inst["pmid"].astype(str)
        inst = inst.drop(columns=["pmid"]).drop_duplicates(subset=["pmid_str"], keep="first")
        df["pmid_str"] = df["pmid"].astype(str)
        df = df.merge(inst, on="pmid_str", how="left")
    else:
        df["institutions"] = ""

    df["title_clean"] = df["title"].map(clean_text)
    df["journal_clean"] = df["journal"].map(clean_text)
    df["abstract_clean"] = df["abstract"].map(clean_text)
    df["doi_clean"] = df["doi"].map(clean_text)
    df["inst_clean"] = df["institutions"].map(clean_text)
    df["abstract_preview"] = df["abstract_clean"].str.slice(0, 500)
    df["inst_preview"] = df["inst_clean"].str.slice(0, 250)
    df["is_glasgow"] = df["inst_clean"].str.contains("Glasgow", case=False, na=False)
    df["year_label"] = df["year_int"].fillna(-1).astype(int).astype(str)
    df.loc[df["year_label"] == "-1", "year_label"] = "Unknown"
    df["pmid_label"] = df["pmid"].astype(str)
    all_count = int(len(df))
    glasgow_count = int(df["is_glasgow"].sum())

    fig = go.Figure()
    all_trace_indices = []
    glasgow_trace_indices = []

    def _customdata(sub_df):
        return np.stack(
            [
                sub_df["pmid_label"],
                sub_df["title_clean"],
                sub_df["year_label"],
                sub_df["journal_clean"],
                sub_df["doi_clean"],
                sub_df["abstract_preview"],
                sub_df["inst_preview"],
                sub_df["cited_by_count"].astype(str),
                sub_df["modality"],
            ],
            axis=-1,
        )

    modality_order = list(MODALITY_COLORS.keys())
    for mod in modality_order:
        sub = df[df["modality"] == mod]
        if sub.empty:
            continue

        fig.add_trace(
            go.Scattergl(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=mod,
                customdata=_customdata(sub),
                marker=dict(size=4, opacity=0.5, color=MODALITY_COLORS[mod]),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Year: %{customdata[2]}<br>"
                    "Journal: %{customdata[3]}<br>"
                    "PMID: %{customdata[0]}<br>"
                    "Modality: %{customdata[8]}<extra></extra>"
                ),
            )
        )
        all_trace_indices.append(len(fig.data) - 1)

        sub_glasgow = sub[sub["is_glasgow"]]
        if sub_glasgow.empty:
            continue

        gla_cites = sub_glasgow["cited_by_count"].to_numpy(dtype=float)
        gla_log = np.log1p(gla_cites)
        if np.isclose(gla_log.max(), gla_log.min()):
            gla_sizes = np.full(len(sub_glasgow), 8.0)
        else:
            gla_sizes = 5.0 + 10.0 * (gla_log - gla_log.min()) / (gla_log.max() - gla_log.min())

        fig.add_trace(
            go.Scattergl(
                x=sub_glasgow["x"],
                y=sub_glasgow["y"],
                mode="markers",
                name=f"{mod} (Glasgow)",
                showlegend=False,
                visible=False,
                customdata=_customdata(sub_glasgow),
                marker=dict(size=gla_sizes, opacity=0.9, color=MODALITY_COLORS[mod]),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Year: %{customdata[2]}<br>"
                    "Journal: %{customdata[3]}<br>"
                    "PMID: %{customdata[0]}<br>"
                    "Modality: %{customdata[8]}<extra></extra>"
                ),
            )
        )
        glasgow_trace_indices.append(len(fig.data) - 1)

    fig.update_layout(
        width=1200,
        height=760,
        plot_bgcolor="white",
        title="Neuroimaging UMAP Explorer (hover + click)",
        legend=dict(itemsizing="constant", itemwidth=40, font=dict(size=11)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    html_div = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="umap-plot",
    )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>UMAP Explorer</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; }}
        .wrap {{ display: grid; grid-template-columns: 1fr 380px; height: 100vh; gap: 0; transition: grid-template-columns 220ms ease; }}
        body.panel-hidden .wrap {{ grid-template-columns: 1fr 0px; }}
    .plot-col {{ background: #ffffff; border-right: 1px solid #e5e7eb; overflow: auto; }}
        .panel {{ padding: 16px; overflow-y: auto; background: #f8fafc; transition: transform 220ms ease, opacity 180ms ease; }}
        body.panel-hidden .panel {{ transform: translateX(100%); opacity: 0; pointer-events: none; }}
    .panel h2 {{ margin: 0 0 10px; font-size: 18px; }}
        .panel-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
        .panel-btn {{ border: 1px solid #d1d5db; background: #ffffff; border-radius: 8px; font-size: 12px; padding: 6px 10px; cursor: pointer; color: #111827; }}
        .panel-btn:hover {{ background: #f3f4f6; }}
        .panel-pop {{ position: fixed; right: 10px; top: 50%; transform: translateY(-50%); z-index: 20; border: 1px solid #d1d5db; background: #ffffff; border-radius: 999px; padding: 8px 12px; font-size: 12px; cursor: pointer; display: none; }}
        body.panel-hidden .panel-pop {{ display: block; }}
    .meta {{ font-size: 13px; color: #374151; margin-bottom: 10px; }}
    .meta div {{ margin-bottom: 6px; }}
    .abstract {{ font-size: 14px; line-height: 1.45; color: #111827; white-space: pre-wrap; }}
    .muted {{ color: #6b7280; }}
    .doi a {{ color: #2563eb; text-decoration: none; }}
    .doi a:hover {{ text-decoration: underline; }}
    .instructions {{ font-size: 13px; color: #4b5563; margin-bottom: 12px; }}
    @media (max-width: 1200px) {{ .wrap {{ grid-template-columns: 1fr; height: auto; }} .plot-col {{ border-right: none; border-bottom: 1px solid #e5e7eb; }} }}
  </style>
</head>
<body>
    <button id="panel-pop" class="panel-pop">Show details</button>
  <div class=\"wrap\">
    <div class=\"plot-col\">{html_div}</div>
        <aside class=\"panel\">
                        <div class="panel-head">
                                <h2>Paper Details</h2>
                                <button id="panel-toggle" class="panel-btn" type="button">Hide</button>
                        </div>
            <div class=\"instructions\">Hover points for quick info. Click a point to pin full details and abstract here.</div>
            <div style=\"margin-bottom: 12px; font-size: 13px; color: #374151;\">
                <label style=\"display: inline-flex; align-items: center; gap: 8px; cursor: pointer;\">
                    <input id=\"glasgow-only-toggle\" type=\"checkbox\" />
                    Show only Glasgow-affiliated papers
                </label>
                <div id=\"visible-count\" style=\"margin-top: 6px; color: #6b7280;\"></div>
            </div>
            <div id=\"paper-detail\" class=\"muted\">No paper selected yet.</div>
        </aside>
  </div>

  <script>
    const plot = document.getElementById('umap-plot');
    const detail = document.getElementById('paper-detail');
        const panelToggle = document.getElementById('panel-toggle');
        const panelPop = document.getElementById('panel-pop');
        const glasgowToggle = document.getElementById('glasgow-only-toggle');
        const visibleCount = document.getElementById('visible-count');
        const allTraceIndices = {json.dumps(all_trace_indices)};
        const glasgowTraceIndices = {json.dumps(glasgow_trace_indices)};
        const allCount = {all_count};
        const glasgowCount = {glasgow_count};

    function esc(value) {{
      if (value === null || value === undefined) return '';
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }}

    function renderPoint(cd) {{
      const pmid = esc(cd[0]);
      const title = esc(cd[1]);
      const year = esc(cd[2]);
      const journal = esc(cd[3]);
      const doi = esc(cd[4]);
    const abstractText = esc(cd[5]);
      const institutions = esc(cd[6]);
      const citedBy = esc(cd[7]);
      const modality = esc(cd[8]);

      const doiLink = doi ? `<a href=\"https://doi.org/${{doi}}\" target=\"_blank\" rel=\"noopener noreferrer\">${{doi}}</a>` : 'N/A';
      const pubmedLink = pmid && pmid !== 'nan'
        ? `<a href=\"https://pubmed.ncbi.nlm.nih.gov/${{pmid}}/\" target=\"_blank\" rel=\"noopener noreferrer\">${{pmid}}</a>`
        : 'N/A';

      detail.innerHTML = `
                <h3 style=\"margin:0 0 10px; font-size:16px; line-height:1.35;\">${{title || 'Untitled'}}</h3>
        <div class=\"meta\">
                    <div><strong>Year:</strong> ${{year || 'Unknown'}}</div>
                    <div><strong>Journal:</strong> ${{journal || 'Unknown'}}</div>
                    <div><strong>Modality:</strong> ${{modality || 'Unknown'}}</div>
                    <div><strong>Citations:</strong> ${{citedBy || '0'}}</div>
                    <div><strong>PMID:</strong> ${{pubmedLink}}</div>
                    <div class=\"doi\"><strong>DOI:</strong> ${{doiLink}}</div>
                    <div><strong>Institutions:</strong> ${{institutions || 'N/A'}}</div>
                </div>
                <div class="abstract">${{abstractText || 'No abstract available.'}}</div>
      `;
    }}

    function updateVisibleCount() {{
      const filterLabel = glasgowToggle.checked ? 'Glasgow only' : 'All papers';
      const totalVisible = glasgowToggle.checked ? glasgowCount : allCount;
      visibleCount.textContent = `Showing ${{totalVisible}} points (${{filterLabel}})`;
    }}

        function setPanelHidden(hidden) {{
            if (hidden) {{
                document.body.classList.add('panel-hidden');
                panelToggle.textContent = 'Show';
            }} else {{
                document.body.classList.remove('panel-hidden');
                panelToggle.textContent = 'Hide';
            }}
            setTimeout(() => Plotly.Plots.resize(plot), 230);
        }}

    function applyGlasgowFilter() {{
      const onlyGlasgow = glasgowToggle.checked;
      if (onlyGlasgow) {{
        Plotly.restyle(plot, {{ visible: false }}, allTraceIndices);
        Plotly.restyle(plot, {{ visible: true }}, glasgowTraceIndices);
      }} else {{
        Plotly.restyle(plot, {{ visible: true }}, allTraceIndices);
        Plotly.restyle(plot, {{ visible: false }}, glasgowTraceIndices);
      }}
      updateVisibleCount();
    }}

    glasgowToggle.addEventListener('change', applyGlasgowFilter);
        panelToggle.addEventListener('click', () => setPanelHidden(!document.body.classList.contains('panel-hidden')));
        panelPop.addEventListener('click', () => setPanelHidden(false));

    plot.on('plotly_click', function(event) {{
      if (!event || !event.points || !event.points.length) return;
      const point = event.points[0];
      if (!point.customdata) return;
            setPanelHidden(false);
      renderPoint(point.customdata);
    }});

    updateVisibleCount();
  </script>
</body>
</html>
"""

    output_path = os.path.join(FIG_DIR, "umap_explorer.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"Saved interactive explorer: {output_path}")


if __name__ == "__main__":
    main()
