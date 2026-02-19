"""
Generate neuroimaging abstract UMAP visualisations and animation.

Loads cached embeddings + UMAP coords, tags modalities & eras,
and saves 5 figures to figures/.

Usage:
    uv run python scripts/make_figures.py
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
import os
import re

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _detect_total_memory_gb():
    try:
        if os.name == "posix":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return None


def _resolve_run_config():
    parser = argparse.ArgumentParser(
        description="Generate UMAP figures and modality animation."
    )
    parser.add_argument(
        "--low-memory",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use reduced-memory rendering defaults.",
    )
    parser.add_argument(
        "--image-format",
        choices=["auto", "pdf", "png"],
        default="auto",
        help="Output format for static figures.",
    )
    args = parser.parse_args()

    total_mem_gb = _detect_total_memory_gb()
    if args.low_memory == "on":
        low_memory = True
    elif args.low_memory == "off":
        low_memory = False
    else:
        low_memory = total_mem_gb is not None and total_mem_gb < 24

    if args.image_format == "auto":
        image_format = "png" if low_memory else "pdf"
    else:
        image_format = args.image_format

    return {
        "low_memory": low_memory,
        "image_format": image_format,
        "total_mem_gb": total_mem_gb,
    }


CONFIG = _resolve_run_config()
LOW_MEMORY = CONFIG["low_memory"]
IMAGE_FORMAT = CONFIG["image_format"]

print("Run config:")
if CONFIG["total_mem_gb"] is None:
    print(f"  low_memory={LOW_MEMORY}, image_format={IMAGE_FORMAT} (RAM unknown)")
else:
    print(
        f"  RAM≈{CONFIG['total_mem_gb']:.1f} GB, low_memory={LOW_MEMORY}, image_format={IMAGE_FORMAT}"
    )


def save_plotly(fig, stem):
    output_path = os.path.join(FIG_DIR, f"{stem}.{IMAGE_FORMAT}")
    fig.write_image(output_path, scale=1)
    print(f"  Saved {output_path}")

# ── 1. Load data ──
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "neuro_abstracts.csv"))

citations_file = os.path.join(DATA_DIR, "citations.csv")
if os.path.exists(citations_file):
    cit = pd.read_csv(citations_file)
    df = df.merge(cit[["pmid", "cited_by_count"]], on="pmid", how="left")
    df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
else:
    df["cited_by_count"] = 0

print(f"  {len(df)} abstracts, {df['journal'].nunique()} journals")

# ── 2. Load UMAP coords ──
coords = np.load(os.path.join(DATA_DIR, "umap_coords.npy"))
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]
df["year_int"] = pd.to_numeric(df["year"], errors="coerce")

if LOW_MEMORY:
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)

# ── 3. Tag modalities ──
def _mesh(row, term):
    mesh = str(row.get("mesh_terms", ""))
    return term.lower() in mesh.lower()

def _kw(row, pattern):
    text = str(row.get("title", "")) + " " + str(row.get("abstract", ""))
    return bool(re.search(pattern, text, re.IGNORECASE))

MODALITY_RULES = [
    ("EEG-fMRI", lambda r: _kw(r, r"simultaneous.*(EEG|fMRI).*(EEG|fMRI)|EEG.fMRI|fMRI.EEG")),
    ("sEEG",     lambda r: _mesh(r, "stereoelectroencephalography") or _kw(r, r"\bsEEG\b|stereo.?EEG|intracranial EEG")),
    ("ECoG",     lambda r: _mesh(r, "Electrocorticography") or _kw(r, r"\bECoG\b|electrocorticogra")),
    ("LFP",      lambda r: _mesh(r, "Local Field Potential") or _kw(r, r"\bLFP\b|local field potential")),
    ("fNIRS",    lambda r: _mesh(r, "Near-Infrared") or _kw(r, r"\bfNIRS\b|functional near.infrared|near.infrared spectroscop")),
    ("MEG",      lambda r: _mesh(r, "Magnetoencephalography") or _kw(r, r"\bMEG\b|magnetoencephalogra")),
    ("tACS",     lambda r: _kw(r, r"\btACS\b|transcranial alternating current")),
    ("tDCS",     lambda r: _mesh(r, "Transcranial Direct Current") or _kw(r, r"\btDCS\b|transcranial direct current")),
    ("TMS",      lambda r: _mesh(r, "Transcranial Magnetic Stimulation") or _kw(r, r"\bTMS\b|transcranial magnetic stimulat")),
    ("EEG",      lambda r: _mesh(r, "Electroencephalography") or _kw(r, r"\bEEG\b|electroencephalogra")),
    ("DTI",      lambda r: _mesh(r, "Diffusion Tensor Imaging") or _kw(r, r"\bDTI\b|diffusion tensor|diffusion.weighted|tractograph")),
    ("fMRI",     lambda r: _kw(r, r"\bfMRI\b|functional MRI|functional magnetic resonance|BOLD|blood.oxygen.level")),
    ("MRI",      lambda r: _mesh(r, "Magnetic Resonance Imaging") or _kw(r, r"\bMRI\b|magnetic resonance imaging")),
]

def assign_modality(row):
    for label, rule in MODALITY_RULES:
        if rule(row):
            return label
    return "Other"

print("Tagging modalities...")
df["modality"] = df.apply(assign_modality, axis=1)

# ── 4. Tag eras ──
ERA_KEYWORDS = {
    "Cartographic / Blobology": [
        "subtraction", "contrast", "activation map", "localization", "localisation",
        "functional specialization", "functional localizer", "functional localiser",
        "block design", "region of interest", "ROI", "voxel-based morphometry",
        "FFA", "fusiform face area", "PPA", "parahippocampal place area",
        "statistical parametric mapping", "SPM", "general linear model",
        "whole-brain analysis", "cluster correction",
    ],
    "Multivariate / Information": [
        "MVPA", "multi-voxel pattern", "multivoxel pattern",
        "decoding", "classifier", "classification accuracy",
        "RSA", "representational similarity", "representational geometry",
        "encoding model", "voxelwise", "population receptive field",
        "searchlight", "cross-validated", "pattern analysis",
        "information content", "neural code", "neural representation",
    ],
    "Connectivity / Networks": [
        "functional connectivity", "effective connectivity",
        "resting state", "resting-state", "default mode",
        "graph theory", "network analysis", "hub", "modularity",
        "dynamic causal model", "DCM", "Granger causality",
        "connectome", "connectomics", "structural connectivity",
        "independent component analysis", "ICA",
    ],
    "Deep Learning": [
        "deep learning", "deep neural network", "convolutional neural network",
        "CNN", "recurrent neural network", "RNN", "LSTM",
        "autoencoder", "generative adversarial", "GAN",
        "transfer learning", "fine-tuning", "end-to-end",
        "brain decoding", "image reconstruction",
        "neural network model", "DNN",
    ],
    "Foundation / LLM / Transformer": [
        "foundation model", "large language model", "LLM",
        "transformer", "GPT", "BERT", "attention mechanism",
        "self-supervised", "pre-trained", "pretrained",
        "brain-to-text", "brain-to-image", "neural encoding",
        "contrastive learning", "CLIP", "vision transformer",
        "representation learning", "latent space",
    ],
}

def assign_era(row):
    text = str(row.get("title", "")) + " " + str(row.get("abstract", ""))
    text_lower = text.lower()
    scores = {era: sum(1 for kw in kws if kw.lower() in text_lower) for era, kws in ERA_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unclassified"

print("Tagging eras...")
df["era"] = df.apply(assign_era, axis=1)

# ── Shared layout settings ──
LAYOUT = dict(
    width=1000 if LOW_MEMORY else 1200,
    height=700 if LOW_MEMORY else 800,
    plot_bgcolor="white",
    font=dict(size=12),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    title=None,
)
MARKER = dict(size=2.0 if LOW_MEMORY else 2.5)

# ── Figure 1: Modality ──
print("Figure 1: Modality...")
MODALITY_COLORS = {
    # Red family — structural & hemodynamic imaging
    "fMRI":  "#dc2626",
    "MRI":   "#ef4444",
    "fNIRS": "#e85d04",   # red-orange (near-infrared)
    "DTI":   "#b91c1c",
    # Blue family — scalp electrophysiology
    "EEG":   "#2563eb",
    "MEG":   "#60a5fa",
    # Yellow family — invasive electrophysiology
    "sEEG":  "#facc15",
    "ECoG":  "#fde047",
    "LFP":   "#eab308",
    # Green family — brain stimulation
    "tDCS":  "#16a34a",
    "tACS":  "#4ade80",
    "TMS":   "#15803d",
    # Purple — hybrid (red MRI + blue EEG)
    "EEG-fMRI": "#8b5cf6",
    # Neutral
    "Other": "#78716c",
}
fig1 = px.scatter(
    df, x="x", y="y", color="modality",
    color_discrete_map=MODALITY_COLORS,
    category_orders={"modality": list(MODALITY_COLORS.keys())},
    opacity=0.35,
)
fig1.update_traces(marker=MARKER)
# Boost legend markers: bigger, fully opaque
for t in fig1.data:
    t.update(legendgrouptitle_font_size=12)
fig1.update_layout(
    **LAYOUT,
    legend=dict(itemsizing="constant", itemwidth=40,
                font=dict(size=11)),
)
fig1.for_each_trace(lambda t: t.update(marker=dict(
    size=t.marker.size, opacity=1.0,
    line=dict(width=0),
)))
# Re-set the scatter opacity only (legend inherits marker.opacity=1)
fig1.update_traces(opacity=0.35, marker_size=MARKER["size"])
save_plotly(fig1, "umap_modality")

# ── Figure 2: Publication Year (log-compressed) ──
print("Figure 2: Publication Year...")
df_yr = df.dropna(subset=["year_int"]).sort_values("year_int").copy()
min_year = df_yr["year_int"].min()
max_year = df_yr["year_int"].max()
# Power transform (exponent > 1) spreads out recent years,
# compresses old years — the opposite of log.
year_norm = (df_yr["year_int"] - min_year) / (max_year - min_year)  # 0-1
df_yr["year_pow"] = year_norm ** 3  # cubic: 2020-2025 much more spread than 1980-1990

fig2 = px.scatter(
    df_yr, x="x", y="y", color="year_pow",
    color_continuous_scale="Plasma",
    opacity=0.4,
)
fig2.update_traces(marker=MARKER)
tick_years = list(range(int(min_year), int(max_year) + 1, 5))
fig2.update_coloraxes(
    colorbar_tickvals=[((y - min_year) / (max_year - min_year)) ** 3 for y in tick_years],
    colorbar_ticktext=[str(y) for y in tick_years],
    colorbar_title="Year",
)
fig2.update_layout(**LAYOUT)
save_plotly(fig2, "umap_year")

# ── Figure 3: Conceptual Era ──
print("Figure 3: Conceptual Era...")
ERA_COLORS = {
    "Cartographic / Blobology": "#e41a1c",
    "Multivariate / Information": "#377eb8",
    "Connectivity / Networks": "#4daf4a",
    "Deep Learning": "#984ea3",
    "Foundation / LLM / Transformer": "#ff7f00",
    "Unclassified": "#cccccc",
}
fig3 = px.scatter(
    df, x="x", y="y", color="era",
    color_discrete_map=ERA_COLORS,
    category_orders={"era": list(ERA_COLORS.keys())},
    opacity=0.4,
)
fig3.update_traces(marker=MARKER)
fig3.update_layout(
    **LAYOUT,
    legend=dict(itemsizing="constant", itemwidth=40, font=dict(size=11)),
)
fig3.update_traces(opacity=0.35, marker_size=MARKER["size"])
save_plotly(fig3, "umap_era")

# ── Figure 4: Journal ──
print("Figure 4: Journal...")
_qual = pc.qualitative.Dark24 + pc.qualitative.Light24
top_journals = df["journal"].value_counts().index.tolist()
JOURNAL_COLORS = {j: _qual[i % len(_qual)] for i, j in enumerate(top_journals)}

fig4 = px.scatter(
    df, x="x", y="y", color="journal",
    color_discrete_map=JOURNAL_COLORS,
    category_orders={"journal": top_journals},
    opacity=0.4,
)
fig4.update_traces(marker=MARKER)
fig4.update_layout(
    **LAYOUT,
    legend=dict(itemsizing="constant", itemwidth=40, font=dict(size=11)),
)
fig4.update_traces(opacity=0.35, marker_size=MARKER["size"])
save_plotly(fig4, "umap_journal")

# ── Figure 5: Citation count (all papers, Viridis) ──
print("Figure 5: Citation count...")
if df["cited_by_count"].sum() > 0:
    df_cit = df.copy()
    df_cit["log_cites"] = np.log1p(df_cit["cited_by_count"])
    df_cit = df_cit.sort_values("log_cites")  # low-cited behind, high-cited on top

    fig5 = px.scatter(
        df_cit, x="x", y="y", color="log_cites",
        color_continuous_scale="Viridis",
        opacity=0.4,
    )
    fig5.update_traces(marker=dict(size=MARKER["size"]))
    # Readable colorbar with actual citation counts
    tick_vals = np.log1p([0, 5, 20, 50, 100, 500, 1000, 5000, 10000])
    tick_text = ["0", "5", "20", "50", "100", "500", "1k", "5k", "10k"]
    # Keep only ticks within our data range
    max_val = df_cit["log_cites"].max()
    keep = [i for i, v in enumerate(tick_vals) if v <= max_val * 1.05]
    fig5.update_coloraxes(
        colorbar_tickvals=[tick_vals[i] for i in keep],
        colorbar_ticktext=[tick_text[i] for i in keep],
        colorbar_title="Citations",
    )
    fig5.update_layout(**LAYOUT)
    save_plotly(fig5, "umap_citations")
else:
    print("  Skipped — no citation data.")

# ── Figure 6: University of Glasgow highlight (modality colours) ──
inst_file = os.path.join(DATA_DIR, "institutions.csv")
if os.path.exists(inst_file):
    print("Figure 6: Glasgow highlight...")
    inst = pd.read_csv(inst_file)
    inst["pmid"] = inst["pmid"].astype(str)
    df["pmid_str"] = df["pmid"].astype(str)
    glasgow_pmids = set(
        inst.loc[
            inst["institutions"].str.contains("Glasgow", case=False, na=False),
            "pmid",
        ]
    )
    df["is_glasgow"] = df["pmid_str"].isin(glasgow_pmids)
    n_gla = df["is_glasgow"].sum()
    print(f"  {n_gla} papers with Glasgow affiliation")

    df_bg = df[~df["is_glasgow"]]
    df_fg = df[df["is_glasgow"]].copy()

    import plotly.graph_objects as go
    fig6 = go.Figure()

    # Background: all non-Glasgow papers in faint grey
    fig6.add_trace(go.Scatter(
        x=df_bg["x"], y=df_bg["y"], mode="markers",
        marker=dict(size=1.5, color="#e5e7eb", opacity=0.08),
        name="Other", showlegend=False,
    ))

    # Foreground: Glasgow papers coloured by modality
    for mod in MODALITY_COLORS:
        sub = df_fg[df_fg["modality"] == mod]
        if len(sub) == 0:
            continue
        fig6.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(size=5, color=MODALITY_COLORS[mod], opacity=0.85,
                        line=dict(width=0)),
            name=mod,
        ))

    fig6.update_layout(
        **LAYOUT,
        legend=dict(itemsizing="constant", itemwidth=40, font=dict(size=11)),
        annotations=[dict(
            text=f"University of Glasgow ({n_gla} papers)",
            x=0.5, y=1.02, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="#333333"),
        )],
    )
    save_plotly(fig6, "umap_glasgow")
else:
    print("  Skipping Glasgow figure — run enrich_institutions.py first.")

# ── Figure 7: Year-by-year animation (MP4, 10 s) ──
print("Figure 7: Animation...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

df_anim = df.dropna(subset=["year_int"]).copy()
df_anim = df_anim.sort_values("year_int")
years_sorted = sorted(df_anim["year_int"].unique())

# Matplotlib-compatible colour map
_mpl_colors = {m: c for m, c in MODALITY_COLORS.items()}
df_anim["color"] = df_anim["modality"].map(_mpl_colors).fillna("#78716c")

# Fixed axis limits (with small padding)
x_pad = (df_anim["x"].max() - df_anim["x"].min()) * 0.03
y_pad = (df_anim["y"].max() - df_anim["y"].min()) * 0.03
xlims = (df_anim["x"].min() - x_pad, df_anim["x"].max() + x_pad)
ylims = (df_anim["y"].min() - y_pad, df_anim["y"].max() + y_pad)

anim_dpi = 110 if LOW_MEMORY else 150
anim_figsize = (12, 8)

fig_a, ax = plt.subplots(figsize=anim_figsize, dpi=anim_dpi)
ax.set_facecolor("white")
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.axis("off")

# Build legend (modality order)
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
           markersize=7, label=m, markeredgewidth=0)
    for m, c in MODALITY_COLORS.items()
]
legend = ax.legend(
    handles=legend_handles, loc="upper right", frameon=False,
    fontsize=9, handletextpad=0.3, borderpad=0.5,
)

# Year annotation above the legend
year_text = ax.annotate(
    "", xy=(1.0, 1.0), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=28, fontweight="bold",
    color="#333333",
)

scat = ax.scatter([], [], s=1.5, linewidths=0)

x_vals = df_anim["x"].to_numpy(dtype=np.float32)
y_vals = df_anim["y"].to_numpy(dtype=np.float32)
c_vals = df_anim["color"].to_numpy()
year_vals = df_anim["year_int"].to_numpy(dtype=np.int32)
frame_ends = np.searchsorted(year_vals, years_sorted, side="right")

def update(frame_idx):
    end_idx = frame_ends[frame_idx]
    scat.set_offsets(np.column_stack([x_vals[:end_idx], y_vals[:end_idx]]))
    scat.set_facecolors(c_vals[:end_idx])
    year_text.set_text(str(int(years_sorted[frame_idx])))
    return scat, year_text

n_frames = len(years_sorted)
target_duration = 10.0  # seconds
fps = max(1, round(n_frames / target_duration))

anim = FuncAnimation(fig_a, update, frames=n_frames, blit=True)

# Try MP4 first (requires ffmpeg), fall back to GIF
try:
    writer = FFMpegWriter(fps=fps, bitrate=1200 if LOW_MEMORY else 2000)
    anim_path = os.path.join(FIG_DIR, "umap_modality_anim.mp4")
    anim.save(anim_path, writer=writer)
    print(f"  Saved animation: {anim_path}")
except Exception as e:
    print(f"  ffmpeg not available ({e}), saving as GIF...")
    anim_path = os.path.join(FIG_DIR, "umap_modality_anim.gif")
    anim.save(anim_path, writer="pillow", fps=fps)
    print(f"  Saved animation: {anim_path}")

plt.close(fig_a)

# ── Figure 8: Glasgow static video (1 frame, 1 second) ──
print("Figure 8: Glasgow static video (1 second)...")
if "is_glasgow" not in df.columns:
    if os.path.exists(inst_file):
        inst = pd.read_csv(inst_file)
        inst["pmid"] = inst["pmid"].astype(str)
        df["pmid_str"] = df["pmid"].astype(str)
        glasgow_pmids = set(
            inst.loc[
                inst["institutions"].str.contains("Glasgow", case=False, na=False),
                "pmid",
            ]
        )
        df["is_glasgow"] = df["pmid_str"].isin(glasgow_pmids)
    else:
        df["is_glasgow"] = False

df_gla = df[df["is_glasgow"]].copy()
if len(df_gla) == 0:
    print("  Skipped — no Glasgow papers found.")
else:
    df_gla["color"] = df_gla["modality"].map(_mpl_colors).fillna("#78716c")

    fig_g, ax_g = plt.subplots(figsize=anim_figsize, dpi=anim_dpi)
    ax_g.set_facecolor("white")
    ax_g.set_xlim(xlims)
    ax_g.set_ylim(ylims)
    ax_g.axis("off")

    # Reuse modality legend styling
    ax_g.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        borderpad=0.5,
    )

    ax_g.annotate(
        "University of Glasgow",
        xy=(1.0, 1.0),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=24,
        fontweight="bold",
        color="#333333",
    )

    ax_g.scatter(
        df_gla["x"].to_numpy(dtype=np.float32),
        df_gla["y"].to_numpy(dtype=np.float32),
        s=6,
        c=df_gla["color"].to_numpy(),
        linewidths=0,
        alpha=0.9,
    )

    try:
        writer = FFMpegWriter(fps=1, bitrate=1200 if LOW_MEMORY else 2000)
        static_path = os.path.join(FIG_DIR, "umap_glasgow_static_1s.mp4")
        with writer.saving(fig_g, static_path, dpi=anim_dpi):
            writer.grab_frame()
        print(f"  Saved static video: {static_path}")
    except Exception as e:
        print(f"  Could not save Glasgow static MP4 ({e}).")

    plt.close(fig_g)

print(f"\nDone! All figures saved to {FIG_DIR}/")
