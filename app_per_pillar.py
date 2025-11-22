# app_per_pillar.py
# Streamlit app: per-year NIRF dashboard with per-subparameter small charts
# Requirements: streamlit, pandas, numpy, plotly
# Optional (for PNG export): kaleido
#
# Run:
#   streamlit run app_per_pillar.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="NIRF — Per-pillar subcharts", layout="wide")

# ---------------------------
# CONFIG: fallback per-year files
# ---------------------------
FALLBACK_FILES = [
    "/Users/shivanshpathak/Code/AIML/Mlops/data/ranked_data/VV2018.csv",
    "/Users/shivanshpathak/Code/AIML/Mlops/data/ranked_data/VV2019.csv",
    "/Users/shivanshpathak/Code/AIML/Mlops/data/ranked_data/VV2021.csv",
    "/Users/shivanshpathak/Code/AIML/Mlops/data/ranked_data/VV2022.csv",
    "/mnt/data/VV2023.csv",
]

PILLAR_MAP = {
    "Teaching": ["SS", "FSR", "FQE", "FRU", "OE"],
    "Research": ["PU", "QP", "IPR", "FPPP", "GPHD"],
    "Graduation": ["GUE", "GPHD"],
    "Outreach": ["RD", "WD", "ESCS", "PCS"],
    "Perception": ["PR"],
}

# ---------------------------
# Helpers
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
        s = str(x)
        s = clean_numeric_string(s)
        if s in ("", ".", "-", "-."):
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def preserve_raw_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    orig_cols = list(df.columns)

    for c in orig_cols:
        raw_name = f"raw::{c}"
        if raw_name not in df.columns:
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
        digit_count = sum(1 for s in sample if re.search(r"\d", s))
        if digit_count >= max(1, int(len(sample) * 0.3)):
            cleaned = df[c].astype(str).apply(clean_numeric_string)
            df[num_col] = cleaned.replace("", np.nan).map(parse_float_safe)
        else:
            df[num_col] = np.nan
    return df

def normalize_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    amin, amax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(amin) or pd.isna(amax) or np.isclose(amax, amin):
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - amin) / (amax - amin) * 100.0

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Load data
# ---------------------------
st.sidebar.title("Data source")
uploaded = st.sidebar.file_uploader("Upload per-year CSV files (multiple)", type="csv", accept_multiple_files=True)

year_dfs = {}
errors = {}

if uploaded:
    for f in uploaded:
        name = f.name
        digits = re.findall(r"\d{4}", name)
        year = int(digits[0]) if digits else None
        try:
            f.seek(0)
            df_raw = pd.read_csv(f, engine="python", on_bad_lines="skip")
            df_proc = preserve_raw_and_clean(df_raw)
            if year is None and "Year" in df_proc.columns:
                cand = pd.to_numeric(df_proc["Year"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
                if cand.dropna().size > 0:
                    year = int(cand.dropna().iloc[0])
            if year is None:
                year = st.sidebar.number_input(f"Year for {name}", min_value=2000, max_value=2100, value=2023, key=name)
            df_proc["Year"] = int(year)
            year_dfs[int(year)] = df_proc
        except Exception as e:
            errors[name] = str(e)
else:
    for p in FALLBACK_FILES:
        try:
            path = Path(p)
            if path.exists():
                df_raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
                df_proc = preserve_raw_and_clean(df_raw)
                digits = re.findall(r"\d{4}", path.name)
                year = int(digits[0]) if digits else 2022
                df_proc["Year"] = year
                year_dfs[year] = df_proc
        except Exception as e:
            errors[p] = str(e)

if errors:
    st.sidebar.error("Some files failed to load:")
    for n, e in errors.items():
        st.sidebar.write(f"- {n}: {e}")

if not year_dfs:
    st.error("No data loaded. Upload CSVs or fix fallback paths.")
    st.stop()

# ---------------------------
# UI controls
# ---------------------------
years_available = sorted(year_dfs.keys())
sel_year = st.sidebar.selectbox("Select Year", years_available, index=len(years_available)-1)
df_year = year_dfs[sel_year].copy()

def detect_institute_col(df):
    candidates = ["Institute Name", "Institution Name", "Institute", "Institution", "InstituteName", "Institute_Name"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "instit" in str(c).lower() or "univer" in str(c).lower() or "college" in str(c).lower():
            return c
    return df.columns[0]

inst_col = detect_institute_col(df_year)
df_year[inst_col] = df_year[inst_col].astype(str).str.strip()
institutes = sorted(df_year[inst_col].dropna().unique())
default_inst = "Indian Institute of Technology Madras" if "Indian Institute of Technology Madras" in institutes else institutes[0]
sel_inst = st.sidebar.selectbox("Select Institute", institutes, index=institutes.index(default_inst))
top_k = st.sidebar.slider("Compare with Top N (average)", 2, 10, 3)

# ---------------------------
# Numeric column mapping
# ---------------------------
numeric_cols = [c for c in df_year.columns if c.endswith("_num")]
base_to_num = {c[:-4]: c for c in numeric_cols}

# Pillar scores (raw + normalized)
for pillar, members in PILLAR_MAP.items():
    cols = [base_to_num.get(m) for m in members if base_to_num.get(m)]
    if cols:
        df_year[f"{pillar}_raw"] = df_year[cols].mean(axis=1)
    else:
        df_year[f"{pillar}_raw"] = np.nan
    df_year[f"{pillar}_score"] = normalize_0_100(df_year[f"{pillar}_raw"])

# Rank column detection
rank_col = next((c for c in df_year.columns if "rank" in str(c).lower()), None)
if rank_col:
    df_year["_rank_num"] = pd.to_numeric(df_year[rank_col].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
else:
    df_year["_rank_num"] = np.nan

sel_row = df_year[df_year[inst_col] == sel_inst].iloc[0]

# ---------------------------
# Header
# ---------------------------
st.title(f"NIRF — {sel_year} (per-subparameter charts)")
st.markdown(f"**Selected:** `{sel_inst}` · Institute column: `{inst_col}`")
st.markdown("---")

# High-level pillar metrics
st.subheader("Pillar scores (0–100 normalized)")
cols = st.columns(5)
for i, p in enumerate(PILLAR_MAP.keys()):
    val = sel_row.get(f"{p}_score", np.nan)
    median = df_year[f"{p}_score"].median()
    delta = val - median if not pd.isna(val) and not pd.isna(median) else None
    cols[i].metric(p, f"{val:.1f}" if not pd.isna(val) else "N/A",
                   f"{delta:+.1f}" if delta is not None else None)

st.markdown("---")
st.subheader("Per-subparameter comparison charts")

for pillar, members in PILLAR_MAP.items():
    st.markdown(f"### {pillar} — breakdown")
    present = [m for m in members if m in base_to_num]
    if not present:
        st.info("No sub-parameters found for this pillar.")
        continue

    # Sort data for Top-K extraction
    if df_year["_rank_num"].notna().any():
        df_sorted = df_year.sort_values("_rank_num")
    else:
        df_sorted = df_year.sort_values(f"{pillar}_score", ascending=False)

    top_df = df_sorted.head(top_k)

    col_layout = st.columns(2)
    col_i = 0

    for sub in present:
        num_col = base_to_num[sub]

        # --------------------------------------------------------------
        #  NEW: guarantee values exist & replace NaN with 0
        # --------------------------------------------------------------
        uni_row = sel_row.copy()
        top1_row = top_df.iloc[0].copy() if len(top_df) > 0 else sel_row.copy()

        # Ensure every sub-parameter column exists → fill missing with 0
        for col in [num_col]:
            if col not in uni_row:
                uni_row[col] = 0
            if col not in top1_row:
                top1_row[col] = 0

        # Replace any remaining NaN with 0
        uni_row = uni_row.fillna(0)
        top1_row = top1_row.fillna(0)

        sel_val = float(uni_row[num_col])
        top1_val = float(top1_row[num_col])
        topk_avg = top_df[num_col].fillna(0).mean()

        # Small dataframe for the chart
        small = pd.DataFrame({
            "label": [f"Top {top_k} avg", "Top 1", sel_inst],
            "value": [topk_avg, top1_val, sel_val]
        })

        # Y-axis scaling
        vals = small["value"]
        minv, maxv = vals.min(), vals.max()
        span = maxv - minv if maxv > minv else 1
        miny = max(0, minv - 0.1 * span)
        maxy = maxv + 0.1 * span

        fig = px.bar(small, x="label", y="value", text="value",
                     title=f"{sub} — raw values",
                     labels={"value": "Score", "label": ""},
                     height=360)
        fig.update_traces(texttemplate="%{text:.4g}", textposition="outside")
        fig.update_layout(yaxis=dict(range=[miny, maxy]), margin=dict(t=50, b=30))

        with col_layout[col_i]:
            st.plotly_chart(fig, use_container_width=True)

            raw_col = f"raw::{sub}"
            raw_val = sel_row.get(raw_col, "") if raw_col in df_year.columns else ""
            st.table(pd.DataFrame({
                "Name": [sel_inst, f"Top {top_k} avg", "Top 1"],
                "Parsed (used)": [f"{sel_val:.6g}", f"{topk_avg:.6g}", f"{top1_val:.6g}"],
                "Raw string (selected)": [raw_val, "", ""]
            }).set_index("Name"))

        col_i = (col_i + 1) % 2

    st.markdown("---")

# ---------------------------
# Export
# ---------------------------
st.subheader("Export")
keep_cols = [inst_col]
if rank_col:
    keep_cols += [rank_col]
keep_cols += [c for c in df_year.columns if c.startswith("raw::") or c.endswith(("_num", "_score", "_raw"))]
st.download_button(
    "Download cleaned CSV for this year",
    data=df_to_csv_bytes(df_year[keep_cols]),
    file_name=f"nirf_{sel_year}_cleaned.csv",
    mime="text/csv"
)

st.caption("Note: install `kaleido` if you want PNG export from Plotly charts.")
st.markdown("---")
st.caption("Made for students & parents | NIRF India per-year analysis — built by Shivansh")