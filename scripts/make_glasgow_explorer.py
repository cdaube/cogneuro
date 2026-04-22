"""
Build an interactive Glasgow UMAP explorer as a single self-contained HTML file.

Features:
- Scatter plot of Glasgow researcher abstracts in UMAP space
- Dropdown to switch colour mapping: School, College, Year, Citation network
- Collapsible side panel with full abstract + metadata (coloured by school)
- Citation edge overlays (cites → blue, cited-by → red)
- GitHub-Pages-friendly: one HTML file, Plotly loaded from CDN

Usage:
    uv run python scripts/make_glasgow_explorer.py
"""

import json
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = BASE_DIR  # put index-glasgow.html at repo root for GitHub Pages
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Distinguishable colours (port of T. E. Holy's MATLAB function)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b])


def distinguishable_colors(n_colors, bg=None):
    if bg is None:
        bg = np.array([[1.0, 1.0, 1.0]])
    bg = np.atleast_2d(bg).astype(float)

    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x, indexing="ij")
    rgb = np.column_stack([R.ravel(), G.ravel(), B.ravel()])

    lab = _rgb_to_lab(rgb)
    bglab = _rgb_to_lab(bg)

    mindist2 = np.full(len(rgb), np.inf)
    for i in range(len(bglab) - 1):
        d = np.sum((lab - bglab[i]) ** 2, axis=1)
        mindist2 = np.minimum(d, mindist2)

    colors = np.zeros((n_colors, 3))
    lastlab = bglab[-1]
    for i in range(n_colors):
        d = np.sum((lab - lastlab) ** 2, axis=1)
        mindist2 = np.minimum(d, mindist2)
        idx = np.argmax(mindist2)
        colors[i] = rgb[idx]
        lastlab = lab[idx]

    return colors


# ---------------------------------------------------------------------------

COLLEGE_COLORS = {
    "MVLS": "#2563eb",
    "CoSE": "#dc2626",
    "NHS": "#16a34a",
    "Arts & Humanities": "#a855f7",
}

SCHOOL_ORDER = [
    "School of Biodiversity, One Health & Veterinary Medicine",
    "School of Cancer Sciences",
    "School of Cardiovascular & Metabolic Health",
    "School of Health & Wellbeing",
    "School of Infection & Immunity",
    "School of Medicine, Dentistry & Nursing",
    "School of Molecular Biosciences",
    "School of Psychology & Neuroscience",
    "School of Mathematics and Statistics",
    "School of Physics and Astronomy",
    "School of Computing Science",
    "School of Chemistry",
    "James Watt School of Engineering",
    "School of Geographical and earth Sciences",
    "School of Biomedical Engineering",
]

MVLS_SCHOOLS = [
    "School of Biodiversity, One Health & Veterinary Medicine",
    "School of Cancer Sciences",
    "School of Cardiovascular & Metabolic Health",
    "School of Health & Wellbeing",
    "School of Infection & Immunity",
    "School of Medicine, Dentistry & Nursing",
    "School of Molecular Biosciences",
    "School of Psychology & Neuroscience",
]

BLUE_HUE_SCHOOLS = [
    "School of Mathematics and Statistics",
    "School of Physics and Astronomy",
    "School of Computing Science",
    "School of Chemistry",
    "James Watt School of Engineering",
    "School of Geographical and earth Sciences",
    "School of Biomedical Engineering",
]

SCHOOL_ORDER_INDEX = {school: idx for idx, school in enumerate(SCHOOL_ORDER)}
SCHOOL_NORMALIZATION = {
    "smdn": "School of Medicine, Dentistry & Nursing",
    "school of medicine, dentistry & nursing": "School of Medicine, Dentistry & Nursing",
    "school of medicine, dentistry and nursing": "School of Medicine, Dentistry & Nursing",
    "school of infection & immunology": "School of Infection & Immunity",
    "school of infection and immunology": "School of Infection & Immunity",
    "school of infection & immunity": "School of Infection & Immunity",
    "school of infection and immunity": "School of Infection & Immunity",
    "james watt school of engineering": "James Watt School of Engineering",
    "suerc": "James Watt School of Engineering",
    "school of geographical and earth sciences": "School of Geographical and earth Sciences",
    "school of humanities": "",
}

LEGACY_SCHOOL_COLORS = {
    "James Watt School of Engineering": "rgb(0,0,255)",
    "SMDN": "rgb(0,255,0)",
    "SUERC": "rgb(255,0,0)",
    "School of Biodiversity, One Health & Veterinary Medicine": "rgb(255,0,175)",
    "School of Biomedical Engineering": "rgb(255,211,8)",
    "School of Cancer Sciences": "rgb(0,131,246)",
    "School of Cardiovascular & Metabolic Health": "rgb(0,140,70)",
    "School of Chemistry": "rgb(175,105,61)",
    "School of Computing Science": "rgb(87,8,96)",
    "School of Geographical and earth Sciences": "rgb(0,140,167)",
    "School of Health & Wellbeing": "rgb(255,175,255)",
    "School of Humanities": "rgb(0,255,237)",
    "School of Infection & Immunology": "rgb(193,255,123)",
    "School of Mathematics and Statistics": "rgb(175,87,246)",
    "School of Medicine, Dentistry & Nursing": "rgb(202,0,70)",
    "School of Molecular Biosciences": "rgb(131,140,0)",
    "School of Physics and Astronomy": "rgb(140,105,131)",
    "School of Psychology & Neuroscience": "rgb(43,61,0)",
}

# Additional blue-ish hues selected from distinguishable_colors(100),
# reusing the existing four blue family colours requested by the user.
BLUE_PALETTE_EXTRA_INDICES = (35, 51, 94)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_school(value):
    cleaned = clean(value)
    if not cleaned:
        return ""
    return SCHOOL_NORMALIZATION.get(cleaned.lower(), cleaned)


def school_sort_key(value):
    return (SCHOOL_ORDER_INDEX.get(value, len(SCHOOL_ORDER)), value)


def rgb_to_css(rgb):
    return f"rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})"


def build_school_color_map():
    palette_100 = distinguishable_colors(100, bg=np.array([[1, 1, 1], [0, 0, 0]]))
    extra_blue_hues = [rgb_to_css(palette_100[idx]) for idx in BLUE_PALETTE_EXTRA_INDICES]
    mvls_hues = [
        LEGACY_SCHOOL_COLORS["School of Biodiversity, One Health & Veterinary Medicine"],
        LEGACY_SCHOOL_COLORS["SUERC"],
        LEGACY_SCHOOL_COLORS["School of Cardiovascular & Metabolic Health"],
        LEGACY_SCHOOL_COLORS["School of Biomedical Engineering"],
        LEGACY_SCHOOL_COLORS["School of Infection & Immunology"],
        LEGACY_SCHOOL_COLORS["School of Medicine, Dentistry & Nursing"],
        LEGACY_SCHOOL_COLORS["School of Molecular Biosciences"],
        LEGACY_SCHOOL_COLORS["School of Computing Science"],
    ]
    blue_hues = [
        extra_blue_hues[0],
        LEGACY_SCHOOL_COLORS["School of Cancer Sciences"],
        extra_blue_hues[1],
        LEGACY_SCHOOL_COLORS["School of Geographical and earth Sciences"],
        LEGACY_SCHOOL_COLORS["James Watt School of Engineering"],
        LEGACY_SCHOOL_COLORS["School of Humanities"],
        extra_blue_hues[2],
    ]

    school_color_map = dict(zip(MVLS_SCHOOLS, mvls_hues, strict=True))
    school_color_map.update(dict(zip(BLUE_HUE_SCHOOLS, blue_hues, strict=True)))

    missing = [school for school in SCHOOL_ORDER if school not in school_color_map]
    if missing:
        raise ValueError(f"Missing school colours for: {missing}")

    return school_color_map


def main():
    # ------------------------------------------------------------------
    # 1. Load & merge data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "glasgow_abstracts.csv"))
    authors = pd.read_csv(os.path.join(DATA_DIR, "glasgow_authors.csv"))
    coords = np.load(os.path.join(DATA_DIR, "glasgow_umap_coords.npy"))

    if len(df) != len(coords):
        raise ValueError("Length mismatch between abstracts and UMAP coords.")

    df["x"] = coords[:, 0].astype(np.float32)
    df["y"] = coords[:, 1].astype(np.float32)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df["pmid"] = df["pmid"].astype(str)
    authors["pmid"] = authors["pmid"].astype(str)
    authors["school"] = authors["school"].map(normalize_school)
    authors = authors[authors["school"].isin(SCHOOL_ORDER)].copy()

    if authors.empty:
        raise ValueError("No Glasgow authors remain after school normalization/filtering.")

    # Citations
    cit_file = os.path.join(DATA_DIR, "glasgow_citations.csv")
    if os.path.exists(cit_file):
        cit = pd.read_csv(cit_file)
        cit["pmid"] = cit["pmid"].astype(str)
        df = df.merge(cit[["pmid", "cited_by_count"]], on="pmid", how="left")
        df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
    else:
        df["cited_by_count"] = 0

    # Author/school/college aggregation
    agg = authors.groupby("pmid").agg(
        glasgow_authors=("author_name", lambda x: "; ".join(sorted(set(x)))),
        schools=("school", lambda x: "; ".join(sorted(set(x), key=school_sort_key))),
        colleges=("college", lambda x: "; ".join(sorted(set(x)))),
        primary_college=("college", "first"),
        primary_school=("school", "first"),
    ).reset_index()
    agg["pmid"] = agg["pmid"].astype(str)
    df = df.merge(agg, on="pmid", how="inner")
    print(f"  Retained {len(df)} papers across {agg['primary_school'].nunique()} schools")

    # Citation graph edges
    graph_file = os.path.join(DATA_DIR, "glasgow_citation_graph.csv")
    edges_json = "[]"
    if os.path.exists(graph_file):
        edges = pd.read_csv(graph_file, dtype=str)
        edges_json = edges.to_json(orient="values")
        print(f"  Citation graph: {len(edges)} edges")

    # ------------------------------------------------------------------
    # 2. Build colour maps
    # ------------------------------------------------------------------
    school_color_map = build_school_color_map()

    # ------------------------------------------------------------------
    # 3. Prepare JSON data blob (one row per paper)
    # ------------------------------------------------------------------
    records = []
    for _, row in df.iterrows():
        records.append({
            "x": float(row["x"]),
            "y": float(row["y"]),
            "pmid": clean(row["pmid"]),
            "title": clean(row.get("title", "")),
            "year": clean(row.get("year", "")),
            "journal": clean(row.get("journal", "")),
            "doi": clean(row.get("doi", "")),
            "abstract": clean(row.get("abstract", "")),
            "all_authors": clean(row.get("all_authors", "")),
            "glasgow_authors": clean(row.get("glasgow_authors", "")),
            "cited_by_count": int(row.get("cited_by_count", 0)),
            "school": clean(row.get("primary_school", "")),
            "college": clean(row.get("primary_college", "")),
            "year_int": int(row["year_int"]) if pd.notna(row["year_int"]) else None,
        })

    data_json = json.dumps(records, separators=(",", ":"))

    # ------------------------------------------------------------------
    # 4. Build HTML
    # ------------------------------------------------------------------
    print("Building HTML...")
    page = _build_html(
        data_json=data_json,
        edges_json=edges_json,
        school_color_map_json=json.dumps(school_color_map, separators=(",", ":")),
        school_order_json=json.dumps(SCHOOL_ORDER, separators=(",", ":")),
        college_color_map_json=json.dumps(COLLEGE_COLORS, separators=(",", ":")),
        n_papers=len(df),
    )

    out_path = os.path.join(OUT_DIR, "glasgow_explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Saved: {out_path}  ({len(page) / 1e6:.1f} MB)")


def _build_html(*, data_json, edges_json, school_color_map_json,
                school_order_json, college_color_map_json, n_papers):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Glasgow Research Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#ffffff;color:#1e293b;overflow:hidden}}
.wrap{{display:grid;grid-template-columns:1fr 400px;height:100vh;transition:grid-template-columns 200ms ease}}
body.panel-hidden .wrap{{grid-template-columns:1fr 0px}}
.plot-col{{position:relative;overflow:hidden}}
#umap-plot{{width:100%;height:100%}}

/* ---- side panel ---- */
.panel{{display:flex;flex-direction:column;overflow-y:auto;background:#f8fafc;border-left:1px solid #e2e8f0;transition:transform 200ms ease,opacity 180ms ease}}
body.panel-hidden .panel{{transform:translateX(100%);opacity:0;pointer-events:none}}
.panel-head{{display:flex;align-items:center;justify-content:space-between;padding:14px 16px 10px;border-bottom:1px solid #e2e8f0}}
.panel-head h2{{margin:0;font-size:16px;font-weight:600}}
.panel-toolbar{{display:flex;gap:6px;align-items:center}}
.btn{{border:1px solid #cbd5e1;background:#ffffff;color:#1e293b;border-radius:6px;font-size:12px;padding:5px 10px;cursor:pointer}}
.btn:hover{{background:#f1f5f9}}
select.btn{{padding-right:24px}}
.panel-pop{{position:fixed;right:10px;top:50%;transform:translateY(-50%);z-index:20;border:1px solid #cbd5e1;background:#ffffff;color:#1e293b;border-radius:999px;padding:8px 14px;font-size:12px;cursor:pointer;display:none}}
body.panel-hidden .panel-pop{{display:block}}

.panel-body{{flex:1;padding:16px;overflow-y:auto}}
.instructions{{font-size:13px;color:#64748b;margin-bottom:14px}}
.paper-card{{border-radius:10px;padding:14px;margin-bottom:10px;background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #cbd5e1}}
.paper-card h3{{margin:0 0 8px;font-size:15px;line-height:1.4;color:#0f172a}}
.meta-row{{font-size:12px;color:#475569;margin-bottom:4px}}
.meta-row strong{{color:#1e293b}}
.abstract-text{{font-size:13px;line-height:1.55;color:#334155;white-space:pre-wrap;margin-top:10px}}
a{{color:#2563eb;text-decoration:none}}
a:hover{{text-decoration:underline}}
.count-badge{{display:inline-block;background:#f1f5f9;color:#64748b;border-radius:9999px;padding:2px 8px;font-size:11px;margin-left:6px}}

/* legend at bottom-left of plot */
#colour-legend{{position:absolute;bottom:12px;left:12px;background:rgba(255,255,255,0.92);border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;max-height:50vh;overflow-y:auto;font-size:11px;z-index:10;max-width:260px}}
#colour-legend .leg-title{{font-weight:600;margin-bottom:6px;font-size:12px;color:#1e293b}}
.leg-item{{display:flex;align-items:center;gap:6px;margin-bottom:3px;cursor:pointer;opacity:0.9}}
.leg-item:hover{{opacity:1}}
.leg-swatch{{width:12px;height:12px;border-radius:3px;flex-shrink:0}}
.leg-label{{color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
</style>
</head>
<body>
<button id="panel-pop" class="panel-pop">&#9664; Details</button>
<div class="wrap">
  <div class="plot-col">
    <div id="umap-plot"></div>
    <div id="colour-legend"></div>
  </div>
  <aside class="panel">
    <div class="panel-head">
      <h2>Paper Details</h2>
      <div class="panel-toolbar">
        <select id="colour-mode" class="btn">
          <option value="school">School</option>
          <option value="college">College</option>
          <option value="year">Year</option>
          <option value="citations">Citation network</option>
        </select>
        <button id="panel-toggle" class="btn" type="button">Hide &#9654;</button>
      </div>
    </div>
    <div class="panel-body">
      <div class="instructions">Hover for preview &middot; Click to pin details &middot; Use the dropdown to change colours</div>
      <div id="paper-detail" style="color:#94a3b8;">Click a point to see its details.</div>
    </div>
  </aside>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────
const DATA = {data_json};
const EDGES = {edges_json};
const SCHOOL_COLORS = {school_color_map_json};
const SCHOOL_ORDER = {school_order_json};
const COLLEGE_COLORS = {college_color_map_json};
const N = DATA.length;
const presentSchools = new Set(DATA.map(d => d.school).filter(Boolean));
const presentColleges = new Set(DATA.map(d => d.college).filter(Boolean));

// pre-index
const pmidIdx = {{}};
DATA.forEach((d, i) => {{ pmidIdx[d.pmid] = i; }});

// citation adjacency
const citesOut = {{}};   // pmid -> [pmid, ...]
const citedBy = {{}};    // pmid -> [pmid, ...]
EDGES.forEach(e => {{
  const [a, b] = e;
  if (!citesOut[a]) citesOut[a] = [];
  citesOut[a].push(b);
  if (!citedBy[b]) citedBy[b] = [];
  citedBy[b].push(a);
}});

// year colour scale
const years = DATA.map(d => d.year_int).filter(y => y !== null);
const minYear = Math.min(...years);
const maxYear = Math.max(...years);
function yearColor(y) {{
  if (y === null) return 'rgb(80,80,80)';
  const t = (y - minYear) / Math.max(maxYear - minYear, 1);
  // viridis-ish: purple → teal → yellow
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + (0.5 - Math.abs(t - 0.5)) * 2 * (150 - 84) + t * (37 - 84));
  return `rgb(${{r}},${{g}},${{Math.max(0, Math.min(255, b))}})`;
}}

// citation connection count
const connCount = DATA.map(d => {{
  return (citesOut[d.pmid] || []).length + (citedBy[d.pmid] || []).length;
}});
const maxConn = Math.max(1, ...connCount);
function connColor(n) {{
  const t = Math.sqrt(n / maxConn);
  const r = Math.round(255 * t);
  const g = Math.round(255 * (1 - t) * 0.4);
  return `rgb(${{r}},${{g}},40)`;
}}

// ── colour assignment helpers ────────────────────────────────────
function getColors(mode) {{
  if (mode === 'school') return DATA.map(d => SCHOOL_COLORS[d.school] || 'rgb(80,80,80)');
  if (mode === 'college') return DATA.map(d => COLLEGE_COLORS[d.college] || 'rgb(80,80,80)');
  if (mode === 'year') return DATA.map(d => yearColor(d.year_int));
  if (mode === 'citations') return DATA.map(d => SCHOOL_COLORS[d.school] || 'rgb(180,180,180)');
  return DATA.map(() => 'rgb(100,100,100)');
}}

function getLegendItems(mode) {{
  if (mode === 'school') {{
    return SCHOOL_ORDER
      .filter(name => presentSchools.has(name))
      .map(name => [name, SCHOOL_COLORS[name]]);
  }}
  if (mode === 'college') {{
    return Object.entries(COLLEGE_COLORS)
      .filter(([name]) => presentColleges.has(name))
      .sort((a,b) => a[0].localeCompare(b[0]));
  }}
  if (mode === 'year') {{
    const steps = 6;
    const items = [];
    for (let i = 0; i <= steps; i++) {{
      const y = Math.round(minYear + i * (maxYear - minYear) / steps);
      items.push([String(y), yearColor(y)]);
    }}
    return items;
  }}
  if (mode === 'citations') {{
    return SCHOOL_ORDER
      .filter(name => presentSchools.has(name))
      .map(name => [name, SCHOOL_COLORS[name]]);
  }}
  return [];
}}

// ── build initial plot ───────────────────────────────────────────
const xs = DATA.map(d => d.x);
const ys = DATA.map(d => d.y);
const texts = DATA.map(d => (d.title || '').slice(0, 80) + '...');

const scatterTrace = {{
  x: xs, y: ys,
  mode: 'markers',
  type: 'scattergl',
  marker: {{ size: 4, opacity: 0.5, color: getColors('school') }},
  text: texts,
  hovertemplate: '<b>%{{text}}</b><extra></extra>',
  hoverinfo: 'text',
}};

const layout = {{
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  margin: {{ l: 5, r: 5, t: 5, b: 5 }},
  xaxis: {{ visible: false }},
  yaxis: {{ visible: false }},
  showlegend: false,
  hovermode: 'closest',
}};

Plotly.newPlot('umap-plot', [scatterTrace], layout, {{
  responsive: true,
  displayModeBar: false,
  scrollZoom: true,
}});

// ── edge traces management ───────────────────────────────────────
let edgeTraceCount = 0;
function clearEdges() {{
  if (edgeTraceCount > 0) {{
    const indices = [];
    const total = document.getElementById('umap-plot').data.length;
    for (let i = total - edgeTraceCount; i < total; i++) indices.push(i);
    Plotly.deleteTraces('umap-plot', indices);
    edgeTraceCount = 0;
  }}
}}

function drawEdges(pmid) {{
  clearEdges();
  const srcIdx = pmidIdx[pmid];
  if (srcIdx === undefined) return;
  const sx = DATA[srcIdx].x, sy = DATA[srcIdx].y;
  const traces = [];

  // outgoing (blue)
  const outs = citesOut[pmid] || [];
  if (outs.length) {{
    const ex = [], ey = [];
    outs.forEach(t => {{
      const ti = pmidIdx[t];
      if (ti !== undefined) {{
        ex.push(sx, DATA[ti].x, null);
        ey.push(sy, DATA[ti].y, null);
      }}
    }});
    if (ex.length) traces.push({{
      x: ex, y: ey, mode: 'lines', type: 'scatter',
      line: {{ color: 'rgba(59,130,246,0.7)', width: 1.5 }},
      hoverinfo: 'skip', showlegend: false,
    }});
  }}

  // incoming (red)
  const ins = citedBy[pmid] || [];
  if (ins.length) {{
    const ex = [], ey = [];
    ins.forEach(s => {{
      const si = pmidIdx[s];
      if (si !== undefined) {{
        ex.push(DATA[si].x, sx, null);
        ey.push(DATA[si].y, sy, null);
      }}
    }});
    if (ex.length) traces.push({{
      x: ex, y: ey, mode: 'lines', type: 'scatter',
      line: {{ color: 'rgba(239,68,68,0.7)', width: 1.5 }},
      hoverinfo: 'skip', showlegend: false,
    }});
  }}

  if (traces.length) {{
    Plotly.addTraces('umap-plot', traces);
    edgeTraceCount = traces.length;
  }}
}}

// ── legend ────────────────────────────────────────────────────────
const legendEl = document.getElementById('colour-legend');
function renderLegend(mode) {{
  const items = getLegendItems(mode);
  const modeLabel = {{ school: 'School', college: 'College', year: 'Year', citations: 'Citations' }}[mode] || mode;
  let html = `<div class="leg-title">${{modeLabel}}</div>`;
  items.forEach(([label, color]) => {{
    html += `<div class="leg-item"><span class="leg-swatch" style="background:${{color}}"></span><span class="leg-label">${{label}}</span></div>`;
  }});
  legendEl.innerHTML = html;
}}
renderLegend('school');

// ── colour mode switching ────────────────────────────────────────
const modeSelect = document.getElementById('colour-mode');
modeSelect.addEventListener('change', () => {{
  const mode = modeSelect.value;
  clearEdges();
  Plotly.restyle('umap-plot', {{ 'marker.color': [getColors(mode)] }}, [0]);
  renderLegend(mode);
}});

// ── panel toggling ───────────────────────────────────────────────
const panelToggle = document.getElementById('panel-toggle');
const panelPop = document.getElementById('panel-pop');
function setPanelHidden(h) {{
  document.body.classList.toggle('panel-hidden', h);
  panelToggle.innerHTML = h ? '&#9664; Show' : 'Hide &#9654;';
  setTimeout(() => Plotly.Plots.resize(document.getElementById('umap-plot')), 220);
}}
panelToggle.addEventListener('click', () => setPanelHidden(!document.body.classList.contains('panel-hidden')));
panelPop.addEventListener('click', () => setPanelHidden(false));

// ── detail rendering ─────────────────────────────────────────────
const detailEl = document.getElementById('paper-detail');
function esc(v) {{
  if (v == null) return '';
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function renderDetail(d) {{
  const schoolColor = SCHOOL_COLORS[d.school] || '#475569';
  const doi = d.doi ? `<a href="https://doi.org/${{esc(d.doi)}}" target="_blank" rel="noopener">${{esc(d.doi)}}</a>` : 'N/A';
  const pmLink = d.pmid && d.pmid !== 'nan'
    ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${{esc(d.pmid)}}/" target="_blank" rel="noopener">${{esc(d.pmid)}}</a>`
    : 'N/A';
  const nOut = (citesOut[d.pmid] || []).length;
  const nIn  = (citedBy[d.pmid] || []).length;

  detailEl.innerHTML = `
    <div class="paper-card" style="border-left-color:${{schoolColor}}">
      <h3>${{esc(d.title) || 'Untitled'}}</h3>
      <div class="meta-row"><strong>Year:</strong> ${{esc(d.year) || '?'}}</div>
      <div class="meta-row"><strong>Journal:</strong> ${{esc(d.journal) || '?'}}</div>
      <div class="meta-row"><strong>Authors:</strong> ${{esc(d.all_authors) || '?'}}</div>
      <div class="meta-row"><strong>Glasgow authors:</strong> ${{esc(d.glasgow_authors) || '?'}}</div>
      <div class="meta-row"><strong>School:</strong> <span style="color:${{schoolColor}};font-weight:600">${{esc(d.school) || '?'}}</span></div>
      <div class="meta-row"><strong>College:</strong> ${{esc(d.college) || '?'}}</div>
      <div class="meta-row"><strong>Total citations:</strong> ${{d.cited_by_count}}</div>
      <div class="meta-row"><strong>Cites in dataset:</strong> ${{nOut}} &nbsp; <strong>Cited by in dataset:</strong> ${{nIn}}</div>
      <div class="meta-row"><strong>PMID:</strong> ${{pmLink}}</div>
      <div class="meta-row"><strong>DOI:</strong> ${{doi}}</div>
      <div class="abstract-text">${{esc(d.abstract) || 'No abstract available.'}}</div>
    </div>`;
}}

// ── click + hover ────────────────────────────────────────────────
const plot = document.getElementById('umap-plot');

plot.on('plotly_click', ev => {{
  if (!ev || !ev.points || !ev.points.length) return;
  const i = ev.points[0].pointIndex;
  const d = DATA[i];
  setPanelHidden(false);
  renderDetail(d);
  drawEdges(d.pmid);
}});

plot.on('plotly_hover', ev => {{
  if (!ev || !ev.points || !ev.points.length) return;
  // lightweight highlight only; full detail on click
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
