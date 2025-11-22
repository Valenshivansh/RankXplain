# Streamlit app: per-year NIRF dashboard with file upload support + fallback file
# Requirements: streamlit, pandas, numpy, plotly
# Optional (for PNG export): kaleido

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import io, traceback, os, re

st.set_page_config(page_title="NIRF — Per-year Dashboard (Upload)", layout="wide")

# ---------------------------
# UNIVERSAL NUMERIC CLEANER
# ---------------------------
def clean_numeric_string(s: str):
    if not isinstance(s, str):
        return s
    s2 = s.replace("\xa0", " ").replace(",", "").replace("%", "").replace("`", "").strip()
    s2 = re.sub(r"\s+", " ", s2)
    s2 = re.sub(r"[^0-9.\-]+$", "", s2)
    return s2

def parse_float_safe(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    try:
        s = clean_numeric_string(str(x))
        return float(s) if s not in ("", ".", "-", "-.") else np.nan
    except:
        return np.nan

# ---------------------------
# FILE READ HELPERS
# ---------------------------
def safe_read_csv_filelike(filelike):
    try:
        filelike.seek(0)
    except:
        pass
    try:
        return pd.read_csv(filelike)
    except:
        try: filelike.seek(0)
        except: pass
        return pd.read_csv(filelike, engine="python", on_bad_lines="skip")

def safe_read_csv_path(path: Path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")

# ---------------------------
# PRESERVE RAW & PARSE NUMERIC
# ---------------------------
def preserve_raw_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    orig_cols = list(df.columns)

    for c in orig_cols:
        raw_name = f"raw::{c}"
        df[raw_name] = df[c].astype(str).replace("nan", "", regex=False)

    for c in orig_cols:
        num_col = f"{c}_num"
        if pd.api.types.is_numeric_dtype(df[c]):
            df[num_col] = pd.to_numeric(df[c], errors="coerce").astype(float)
            continue

        sample = df[c].dropna().astype(str).head(20).tolist()
        if not sample:
            df[num_col] = np.nan
            continue

        digit_count = sum(1 for s in sample if re.search(r"[0-9]", s))
        if digit_count >= max(1, int(len(sample) * 0.3)):
            cleaned = df[c].astype(str).apply(clean_numeric_string)
            df[num_col] = cleaned.replace("", np.nan).map(parse_float_safe)
        else:
            df[num_col] = np.nan

    return df

# ---------------------------
# NORMALIZE
# ---------------------------
def normalize_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    amin, amax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(amin) or pd.isna(amax) or np.isclose(amax, amin):
        return pd.Series([50.0]*len(s), index=s.index)
    return (s - amin) / (amax - amin) * 100.0

# ---------------------------
# CONSTANTS
# ---------------------------
FALLBACK_COMBINED = Path("/mnt/data/nirf_combined_predicted.csv")

PILLAR_MAP = {
    "Teaching": ["SS", "FSR", "FQE", "FRU", "OE"],
    "Research": ["PU", "QP", "IPR", "FPPP", "GPHD"],
    "Graduation": ["GUE", "GPHD"],
    "Outreach": ["RD", "WD", "ESCS", "PCS"],
    "Perception": ["PR"]
}

# ---------------------------
# UPLOAD UI
# ---------------------------
st.sidebar.title("Upload per-year CSVs")
uploaded_files = st.sidebar.file_uploader("CSV files (multiple)", type="csv", accept_multiple_files=True)

year_dfs = {}
upload_errors = {}

if uploaded_files:
    for f in uploaded_files:
        name = f.name
        digits = re.findall(r"\d{4}", name)
        year = int(digits[0]) if digits else None
        try:
            df_raw = safe_read_csv_filelike(f)
            df_proc = preserve_raw_and_clean(df_raw)

            if year is None and "Year" in df_proc.columns:
                yr_vals = pd.to_numeric(
                    df_proc["Year"].astype(str).str.extract(r"(\d{4})", expand=False),
                    errors="coerce"
                ).dropna()
                if not yr_vals.empty:
                    year = int(yr_vals.iloc[0])

            if year is None:
                year = st.sidebar.number_input(
                    f"Year for {name}", 2000, 2100, 2022, 1, key=name
                )

            df_proc["Year"] = int(year)
            year_dfs[int(year)] = df_proc

        except Exception as e:
            upload_errors[name] = str(e)

# Fallback load
if not year_dfs and FALLBACK_COMBINED.exists():
    df_comb = safe_read_csv_path(FALLBACK_COMBINED)
    df_comb = preserve_raw_and_clean(df_comb)
    if "Year" in df_comb.columns:
        if "Year_num" not in df_comb.columns:
            df_comb["Year_num"] = (
                df_comb["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
            )
            df_comb["Year_num"] = pd.to_numeric(df_comb["Year_num"], errors="coerce")

        for y in df_comb["Year_num"].dropna().unique().astype(int):
            df_y = df_comb[df_comb["Year_num"] == y].copy()
            if not df_y.empty:
                year_dfs[int(y)] = df_y
    else:
        year_dfs[2022] = df_comb

if upload_errors:
    st.sidebar.error("Errors while reading files:")
    for n, err in upload_errors.items():
        st.sidebar.write(f"- {n}: {err}")

if not year_dfs:
    st.error("No valid data. Upload files or add fallback.")
    st.stop()

# ---------------------------
# YEAR + INSTITUTE SELECTION
# ---------------------------
years_available = sorted(year_dfs.keys())
sel_year = st.sidebar.selectbox("Select year", years_available, index=len(years_available)-1)
df_year = year_dfs[sel_year].copy()

# detect institute column
def detect_institute_col(df):
    candidates = [
        "Institute Name","Institution Name","Institute","Institution",
        "InstituteName","Institute_Name"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.startswith("raw::"):
            orig = c.replace("raw::","")
            if any(k in orig.lower() for k in ["instit","college","univer","iit"]):
                return orig
    for c in df.columns:
        if any(k in c.lower() for k in ["instit","univer","college"]):
            return c
    return None

inst_col = detect_institute_col(df_year)
if not inst_col:
    st.error("Cannot detect institute column")
    st.stop()

df_year[inst_col] = df_year[inst_col].astype(str).str.strip()
institutes = sorted(df_year[inst_col].dropna().unique().tolist())

default_choice = (
    "Indian Institute of Technology Madras"
    if "Indian Institute of Technology Madras" in institutes
    else institutes[0]
)

sel_inst = st.sidebar.selectbox("Select institute", institutes, index=institutes.index(default_choice))

st.title(f"NIRF Visuals — {sel_year}")
st.markdown(f"**Institute column:** `{inst_col}` · Selected: **{sel_inst}**")
st.markdown("---")

# ---------------------------
# NUMERIC COLUMNS
# ---------------------------
num_cols = [c for c in df_year.columns if c.endswith("_num")]
base_to_num = {c[:-4]: c for c in num_cols}

# ---------------------------
# PILLAR SCORE CALC
# ---------------------------
pillars = list(PILLAR_MAP.keys())

for pillar, members in PILLAR_MAP.items():
    num_members = [base_to_num[m] for m in members if m in base_to_num]
    if num_members:
        df_year[f"{pillar}_raw_num"] = df_year[num_members].mean(axis=1)
    else:
        df_year[f"{pillar}_raw_num"] = np.nan

    df_year[f"{pillar}_score"] = normalize_0_100(df_year[f"{pillar}_raw_num"])

# ---------------------------
# RANK DETECTION
# ---------------------------
rank_candidates = ["Rank","Overall Rank","Overall_Rank","Predicted_Rank"]
rank_col = next((c for c in rank_candidates if c in df_year.columns), None)
if not rank_col:
    for c in df_year.columns:
        if "rank" in c.lower():
            rank_col = c
            break

if rank_col:
    df_year["_rank_num"] = pd.to_numeric(
        df_year[rank_col].astype(str).str.replace(r"[^0-9]","",regex=True),
        errors="coerce"
    )
else:
    df_year["_rank_num"] = np.nan

# ---------------------------
# SELECTED ROW
# ---------------------------
sel_row = df_year[df_year[inst_col] == sel_inst]
if sel_row.empty:
    st.error("Institute not found after parsing")
    st.stop()

sel_row = sel_row.iloc[0]

# ---------------------------
# RAW vs CLEANED (FRU demo)
# ---------------------------
st.subheader("Raw vs cleaned preview (example: FRU)")
raw_col = f"raw::FRU"
num_col = base_to_num.get("FRU")

preview = {}
if raw_col in df_year.columns:
    preview["FRU raw"] = sel_row.get(raw_col)
if num_col:
    preview["FRU numeric"] = sel_row.get(num_col)

if preview:
    st.table(pd.DataFrame([preview]))
else:
    st.info("FRU not found in this year's data.")

# ---------------------------
# METRIC SUMMARY
# ---------------------------
st.subheader("Five pillar summary")
c_cols = st.columns(5)

for i, p in enumerate(pillars):
    val = sel_row.get(f"{p}_score", np.nan)
    median = df_year[f"{p}_score"].median() if f"{p}_score" in df_year else np.nan

    if pd.isna(val):
        c_cols[i].metric(p, "N/A")
    else:
        c_cols[i].metric(p, f"{val:.1f}/100", f"{val - median:.1f}" if not pd.isna(median) else "N/A")

st.markdown("---")

# ---------------------------
# TOP-K COMPARISON
# ---------------------------
if df_year["_rank_num"].notna().any():
    df_sorted = df_year.sort_values("_rank_num", na_position="last")
else:
    df_sorted = df_year.sort_values(f"{pillars[0]}_score", ascending=False)

top_k = 3
topk_df = df_sorted.head(top_k)
selected_df = df_year[df_year[inst_col] == sel_inst]
compare_df = pd.concat([topk_df, selected_df]).drop_duplicates(subset=[inst_col])

plot_cols = [f"{p}_score" for p in pillars]
for c in plot_cols:
    if c not in compare_df.columns:
        compare_df[c] = np.nan

# ---------------------------
# CHARTS
# ---------------------------
chart_type = st.selectbox("Choose chart type", ["Grouped Bar","Radar","Parallel Coordinates","Line","Polar"])
st.subheader("Comparison Chart")

fig = None

# --- Grouped Bar ---
if chart_type == "Grouped Bar":
    bar_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col:"Institute"})
    rename = {f"{p}_score": p for p in pillars}
    bar_df = bar_df.rename(columns=rename)
    melt_df = bar_df.melt(id_vars="Institute", var_name="Pillar", value_name="Score")
    melt_df["Score"] = pd.to_numeric(melt_df["Score"], errors="coerce")

    fig = px.bar(melt_df, x="Institute", y="Score", color="Pillar", barmode="group", height=520)
    fig.update_layout(xaxis_tickangle=-25, yaxis=dict(title="Score", rangemode="tozero"))

# --- Radar ---
elif chart_type == "Radar":
    cats = pillars + [pillars[0]]
    sel_vals = [sel_row.get(f"{p}_score",0) for p in pillars]
    top_avg = compare_df.head(top_k)[plot_cols].mean().tolist()
    top1    = compare_df.head(1)[plot_cols].iloc[0].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=top_avg+[top_avg[0]], theta=cats, fill='toself', name=f"Top {top_k} Avg"))
    fig.add_trace(go.Scatterpolar(r=sel_vals+[sel_vals[0]], theta=cats, fill='toself', name=sel_inst))
    fig.add_trace(go.Scatterpolar(r=top1+[top1[0]], theta=cats, fill='toself', name="Top 1"))

    fig.update_layout(height=520, polar=dict(radialaxis=dict(range=[0,100])))

# --- Parallel Coordinates ---
elif chart_type == "Parallel Coordinates":
    pc_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col:"Institute"})
    for c in plot_cols:
        pc_df[c] = pd.to_numeric(pc_df[c], errors="coerce")

    try:
        fig = px.parallel_coordinates(
            pc_df,
            dimensions=plot_cols,
            color=plot_cols[0],
            labels={c:c.replace("_score","") for c in plot_cols},
            height=520
        )
    except:
        st.warning("Parallel coords failed due to missing numeric values")

# --- Line ---
elif chart_type == "Line":
    fig = go.Figure()
    for _, row in compare_df.iterrows():
        name = row[inst_col]
        vals = [row[c] for c in plot_cols]
        fig.add_trace(go.Scatter(x=pillars, y=vals, mode="lines+markers", name=name))
    fig.update_layout(height=520, yaxis=dict(range=[0,100], title="Score"))

# --- Polar Chart ---
elif chart_type == "Polar":
    theta = pillars
    fig = go.Figure()
    for _, row in compare_df.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[c] for c in plot_cols],
                theta=theta,
                fill='toself',
                name=row[inst_col]
            )
        )
    fig.update_layout(height=520, polar=dict(radialaxis=dict(range=[0,100])))

# Show chart
if fig:
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# EXPORT SECTION
# ---------------------------
st.markdown("---")
st.subheader("Export Data / Figures")

colA, colB = st.columns(2)

with colA:
    csv_bytes = df_year.to_csv(index=False).encode("utf-8")
    st.download_button("Download full-year CSV", csv_bytes, file_name=f"nirf_{sel_year}.csv")

with colB:
    if fig:
        try:
            png_bytes = fig.to_image(format="png", engine="kaleido")
            st.download_button("Download chart PNG", png_bytes, file_name=f"chart_{chart_type}_{sel_year}.png")
        except:
            st.info("Install kaleido for PNG exports: pip install -U kaleido")