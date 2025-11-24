# Main code :

# Streamlit app: per-year NIRF dashboard with file upload support + fallback file
# Requirements: streamlit, pandas, numpy, plotly
# Optional (for PNG export): kaleido
#
# Run:
#   streamlit run app_per_year_upload.py

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
    """Remove common noise from numeric-like strings and return string ready for float()."""
    if not isinstance(s, str):
        return s
    # remove unicode non-breaking spaces, commas, percent signs, backticks, and trailing text
    s2 = s.replace("\xa0", " ").replace(",", "").replace("%", "").replace("`", "").strip()
    # sometimes there are multiple spaces; collapse them
    s2 = re.sub(r"\s+", " ", s2)
    # remove any trailing non-digit/non-dot/minus characters
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
        return float(s) if s not in ("", ".", "-", "-.") else np.nan
    except Exception:
        return np.nan

# ---------------------------
# FILE READ HELPERS
# ---------------------------
def safe_read_csv_filelike(filelike):
    """Read an uploaded file-like object robustly, resetting pointer as needed."""
    try:
        filelike.seek(0)
    except Exception:
        pass
    try:
        return pd.read_csv(filelike)
    except Exception:
        try:
            filelike.seek(0)
        except Exception:
            pass
        # fallback: use python engine and skip bad lines
        return pd.read_csv(filelike, engine="python", on_bad_lines="skip")

def safe_read_csv_path(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        # fallback to python engine
        return pd.read_csv(path, engine="python", on_bad_lines="skip")

# ---------------------------
# PRESERVE RAW & PARSE NUMERIC
# ---------------------------
def preserve_raw_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a raw::COL copy (string) for each original column and add COL_num for parsed numeric value.
    Uses robust heuristics and keeps original columns intact.
    """
    df = df.copy(deep=True)
    # normalize column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    orig_cols = list(df.columns)

    # create raw:: columns from original string representation (only once)
    for c in orig_cols:
        raw_name = f"raw::{c}"
        if raw_name not in df.columns:
            df[raw_name] = df[c].astype(str).replace("nan", "", regex=False)

    # create numeric parsed columns like `<col>_num`
    for c in orig_cols:
        num_col = f"{c}_num"
        # if already numeric dtype, cast to float and set
        if pd.api.types.is_numeric_dtype(df[c]):
            df[num_col] = pd.to_numeric(df[c], errors="coerce").astype(float)
            continue
        # sample to see if it appears numeric-ish
        sample = df[c].dropna().astype(str).head(20).tolist()
        if not sample:
            df[num_col] = np.nan
            continue
        digit_count = sum(1 for s in sample if re.search(r"[0-9]", s))
        # if many samples contain digits, attempt cleaning parse
        if digit_count >= max(1, int(len(sample) * 0.3)):
            cleaned = df[c].astype(str).apply(clean_numeric_string)
            df[num_col] = cleaned.replace("", np.nan).map(parse_float_safe)
        else:
            df[num_col] = np.nan

    return df

# ---------------------------
# NORMALIZE HELPER
# ---------------------------
def normalize_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    amin = s.min(skipna=True)
    amax = s.max(skipna=True)
    if pd.isna(amin) or pd.isna(amax) or np.isclose(amax, amin):
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - amin) / (amax - amin) * 100.0

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def try_fig_png(fig):
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception:
        return None

# -----------------------------
# CONFIG / PILLAR MAP
# -----------------------------
FALLBACK_COMBINED = Path("/mnt/data/nirf_combined_predicted.csv")
PILLAR_MAP = {
    "Teaching": ["SS", "FSR", "FQE", "FRU", "OE"],
    "Research": ["PU", "QP", "IPR", "FPPP", "GPHD"],
    "Graduation": ["GUE", "GPHD"],
    "Outreach": ["RD", "WD", "ESCS", "PCS"],
    "Perception": ["PR"]
}

# -----------------------------
# UI — Upload area
# -----------------------------
st.sidebar.title("Upload per-year CSVs")
st.sidebar.markdown("Upload CSVs for years (2018, 2019, 2021, 2022, 2023). If none uploaded, app tries fallback combined file.")
uploaded_files = st.sidebar.file_uploader("CSV files (multiple)", type="csv", accept_multiple_files=True)

year_dfs = {}
upload_errors = {}

# process uploaded files
if uploaded_files:
    for f in uploaded_files:
        name = f.name
        # infer year from filename digits (first 4-digit number)
        digits = re.findall(r"\d{4}", name)
        year = int(digits[0]) if digits else None
        try:
            df_raw = safe_read_csv_filelike(f)
            df_proc = preserve_raw_and_clean(df_raw)
            # if year not found, try Year column
            if year is None:
                if "Year" in df_proc.columns:
                    yr_vals = pd.to_numeric(df_proc["Year"].astype(str).str.extract(r"(\d{4})", expand=False).dropna(), errors="coerce")
                    if not yr_vals.dropna().empty:
                        year = int(yr_vals.dropna().iloc[0])
            if year is None:
                # prompt small input in sidebar per file (safe default)
                year = st.sidebar.number_input(f"Year for file {name}", min_value=2000, max_value=2100, value=2022, step=1, key=name)
            df_proc["Year"] = int(year)
            year_dfs[int(year)] = df_proc
        except Exception as e:
            upload_errors[name] = str(e)

# fallback: combined file (split by Year column if present)
if not year_dfs and FALLBACK_COMBINED.exists():
    try:
        df_combined = safe_read_csv_path(FALLBACK_COMBINED)
        df_combined = preserve_raw_and_clean(df_combined)
        if "Year" in df_combined.columns:
            # convert Year_num if there
            if "Year_num" not in df_combined.columns:
                # try to parse Year column into Year_num
                if pd.api.types.is_numeric_dtype(df_combined["Year"]):
                    df_combined["Year_num"] = pd.to_numeric(df_combined["Year"], errors="coerce").astype(pd.Int64Dtype())
                else:
                    df_combined["Year_num"] = df_combined["Year"].astype(str).str.extract(r"(\d{4})", expand=False)
                    df_combined["Year_num"] = pd.to_numeric(df_combined["Year_num"], errors="coerce").astype(pd.Int64Dtype())
            for y in sorted(df_combined["Year_num"].dropna().unique().astype(int)):
                df_y = df_combined[df_combined["Year_num"] == y].copy()
                if not df_y.empty:
                    year_dfs[int(y)] = df_y
        else:
            # single-year fallback
            year_dfs[2022] = df_combined
    except Exception as e:
        st.sidebar.error(f"Fallback combined file read error: {e}")

# show upload errors
if upload_errors:
    st.sidebar.error("Some uploaded files failed to read:")
    for n, err in upload_errors.items():
        st.sidebar.write(f"- {n}: {err}")

if not year_dfs:
    st.error("No per-year data available. Upload CSV files or place a fallback at /mnt/data/nirf_combined_predicted.csv")
    st.stop()

# -----------------------------
# Choose year & institute
# -----------------------------
years_available = sorted(year_dfs.keys())
sel_year = st.sidebar.selectbox("Select year", years_available, index=len(years_available)-1)
df_year = year_dfs[sel_year].copy()

# detect institute column (robust)
def detect_institute_col(df):
    candidates = ["Institute Name", "Institution Name", "Institute", "Institution", "InstituteName", "Institute_Name"]
    cols = [c for c in df.columns]
    # exact names first
    for cand in candidates:
        if cand in cols:
            return cand
    # try raw:: columns
    for c in cols:
        if c.startswith("raw::"):
            orig = c.replace("raw::", "")
            if any(k.lower() in orig.lower() for k in ["instit", "college", "university", "iit", "iisc", "jawaharlal", "banaras", "amity"]):
                return orig  # return original column name (not the raw one)
    # fallback: any column with 'instit' substring
    for c in cols:
        if "instit" in c.lower() or "univer" in c.lower() or "college" in c.lower():
            return c
    return None

inst_col = detect_institute_col(df_year)
if inst_col is None:
    st.error("Could not detect institute name column. Columns available:")
    st.write(list(df_year.columns))
    st.stop()

# ensure institute strings
df_year[inst_col] = df_year[inst_col].astype(str).str.strip()

institutes = sorted(df_year[inst_col].dropna().unique().tolist())
if not institutes:
    st.error("No institutes found after parsing.")
    st.stop()

default_choice = "Indian Institute of Technology Madras" if "Indian Institute of Technology Madras" in institutes else institutes[0]
sel_inst = st.sidebar.selectbox("Select institute", institutes, index=institutes.index(default_choice))

st.title(f"NIRF Visuals — {sel_year}")
st.markdown(f"**Institute column detected:** `{inst_col}` · Selected: **{sel_inst}**")
st.markdown("---")

# -----------------------------
# Build mapping of base -> numeric column
# -----------------------------
num_cols = [c for c in df_year.columns if c.endswith("_num")]
base_to_num = {c[:-4]: c for c in num_cols}  # remove trailing '_num' to get base name

# debug: show which numeric columns we have
st.sidebar.markdown("**Parsed numeric columns (sample)**")
for k, v in list(base_to_num.items())[:20]:
    st.sidebar.write(f"- {k} -> {v}")

# -----------------------------
# Create pillar raw & score (numerics)
# -----------------------------
pillars = list(PILLAR_MAP.keys())
for pillar, members in PILLAR_MAP.items():
    numeric_members = [base_to_num[m] for m in members if m in base_to_num]
    if numeric_members:
        df_year[f"{pillar}_raw_num"] = df_year[numeric_members].mean(axis=1)
    else:
        df_year[f"{pillar}_raw_num"] = np.nan
    df_year[f"{pillar}_score"] = normalize_0_100(df_year[f"{pillar}_raw_num"])

# -----------------------------
# rank detection (heuristic)
# -----------------------------
rank_candidates = ["Rank", "Overall_Rank", "Overall Rank", "Official_Overall_Rank", "Predicted_Rank"]
rank_col = None
for c in rank_candidates:
    if c in df_year.columns:
        rank_col = c
        break
if rank_col is None:
    for c in df_year.columns:
        if "rank" in c.lower():
            rank_col = c
            break

# try to create numeric rank
if rank_col:
    df_year["_rank_num"] = pd.to_numeric(df_year[rank_col].astype(str).str.replace(r"[^0-9]", "", regex=True), errors="coerce")
else:
    df_year["_rank_num"] = np.nan

# -----------------------------
# Selected institute row
# -----------------------------
sel_rows = df_year[df_year[inst_col].astype(str) == sel_inst]
if sel_rows.empty:
    st.error("Selected institute not found (possible minor string mismatch).")
    st.stop()
sel_row = sel_rows.iloc[0]

# show debug raw vs cleaned for FRU if present
st.subheader("Raw vs cleaned preview (example: FRU)")
raw_col = f"raw::FRU"
num_col = base_to_num.get("FRU")
debug = {}
if raw_col in df_year.columns:
    debug["FRU raw string"] = sel_row.get(raw_col, "")
if num_col:
    debug["FRU parsed numeric"] = sel_row.get(num_col, np.nan)
if debug:
    st.table(pd.DataFrame([debug]))
else:
    st.info("FRU column not found in this year's data (check CSV header names).")

# -----------------------------
# Big pillar metrics display
# -----------------------------
st.subheader("Five pillar summary (selected institute)")
cols_disp = st.columns(5)
for i, p in enumerate(pillars):
    val = sel_row.get(f"{p}_score", np.nan)
    median = df_year[f"{p}_score"].median() if f"{p}_score" in df_year.columns else np.nan
    if pd.isna(val):
        disp = "N/A"
        delta = "N/A"
    else:
        disp = f"{float(val):.1f}/100"
        delta = f"{(float(val)-float(median)):.1f}" if not pd.isna(median) else "N/A"
    cols_disp[i].metric(label=p, value=disp, delta=delta)

st.markdown("---")

# -----------------------------
# Build compare_df (Top3 by rank if present)
# -----------------------------
if not df_year["_rank_num"].isna().all():
    df_sorted = df_year.sort_values(by="_rank_num", na_position="last")
else:
    # fallback: sort by first pillar score descending
    df_sorted = df_year.sort_values(by=f"{pillars[0]}_score", ascending=False, na_position="last")

top_k = 3
topk = df_sorted.head(top_k).copy()
selected_df = df_year[df_year[inst_col].astype(str) == sel_inst].copy()
compare_df = pd.concat([topk, selected_df], ignore_index=True).drop_duplicates(subset=[inst_col])

# ensure plot_cols exist
plot_cols = [f"{p}_score" for p in pillars]
for c in plot_cols:
    if c not in compare_df.columns:
        compare_df[c] = np.nan

# -----------------------------
# Chart selector & plotting
# -----------------------------
chart_type = st.selectbox("Choose chart type", ["Grouped Bar", "Radar", "Parallel Coordinates", "Line", "Polar"])

st.subheader("Comparison chart")
fig = None

if chart_type == "Grouped Bar":
    bar_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col: "Institute"})
    rename_map = {f"{p}_score": p for p in pillars}
    bar_df = bar_df.rename(columns=rename_map)
    df_melt = bar_df.melt(id_vars="Institute", value_vars=list(rename_map.values()), var_name="Pillar", value_name="Score")
    # ensure numeric
    df_melt["Score"] = pd.to_numeric(df_melt["Score"], errors="coerce")
    fig = px.bar(df_melt, x="Institute", y="Score", color="Pillar", barmode="group", height=520)
    fig.update_layout(xaxis_tickangle=-25, yaxis=dict(title="Score", rangemode="tozero"))

elif chart_type == "Radar":
    cats = pillars + [pillars[0]]
    sel_vals = [float(sel_row.get(f"{p}_score", 0) or 0) for p in pillars]
    top_avg_vals = compare_df.head(top_k)[plot_cols].mean().values.tolist() if not compare_df.empty else [0]*len(pillars)
    top1_vals = compare_df.head(1)[plot_cols].iloc[0].values.tolist() if not compare_df.empty else [0]*len(pillars)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(top_avg_vals)+[top_avg_vals[0]], theta=cats, fill='toself', name=f"Top {top_k} avg"))
    fig.add_trace(go.Scatterpolar(r=sel_vals+[sel_vals[0]], theta=cats, fill='toself', name=sel_inst))
    fig.add_trace(go.Scatterpolar(r=list(top1_vals)+[top1_vals[0]], theta=cats, fill='toself', name="Top1"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])), height=520)

elif chart_type == "Parallel Coordinates":
    pc_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col: "Institute"})
    # ensure numeric for plotly
    for pc in plot_cols:
        pc_df[pc] = pd.to_numeric(pc_df[pc], errors="coerce")
    try:
        fig = px.parallel_coordinates(pc_df, color=plot_cols[0], dimensions=plot_cols,
                                      labels={pc: pc.replace("_score", "") for pc in plot_cols}, height=520)
    except Exception:
        fig = None
        st.warning("Parallel coordinates couldn't render due to missing numeric data.")

elif chart_type == "Line":
    ln_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col: "Institute"})
    fig = go.Figure()
    for _, row in ln_df.iterrows():
        name = row["Institute"]
        vals = [row[c] if not pd.isna(row[c]) else None for c in plot_cols]
        fig.add_trace(go.Scatter(x=pillars, y=vals, mode="lines+markers", name=name))
    fig.update_layout(height=520, yaxis=dict(range=[0,100], title="Score"))

elif chart_type == "Polar":
    pol_df = compare_df[[inst_col] + plot_cols].rename(columns={inst_col: "Institute"})
    fig = go.Figure()
    theta = pillars
    for _, r in pol_df.iterrows():
        vals = [float(r[c] if not pd.isna(r[c]) else 0) for c in plot_cols]
        fig.add_trace(go.Barpolar(r=vals, theta=theta, name=r["Institute"], opacity=0.7))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])), height=520)

if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Chart not available (maybe missing numeric data).")

# -----------------------------
# Export CSV/PNG
# -----------------------------
st.markdown("---")
st.subheader("Export")
export_cols = [inst_col]
if rank_col and rank_col in df_year.columns:
    export_cols.append(rank_col)
export_cols += plot_cols
export_small = compare_df[[c for c in export_cols if c in compare_df.columns]].copy()
st.download_button("Download comparison CSV", data=df_to_csv_bytes(export_small),
                   file_name=f"comparison_{sel_year}_{sel_inst.replace(' ', '_')}.csv", mime="text/csv")
png = try_fig_png(fig) if fig is not None else None
if png:
    st.download_button("Download chart PNG", data=png, file_name=f"chart_{sel_year}_{sel_inst.replace(' ', '_')}.png", mime="image/png")
else:
    st.info("To enable PNG export, install `kaleido` in your venv: pip install kaleido")

# -----------------------------
# Drilldown: show sub-parameters of a pillar
# -----------------------------
st.markdown("---")
st.subheader("Drill-down: show sub-parameters of a pillar")

available_pillars = [p for p in PILLAR_MAP.keys() if any(m in base_to_num for m in PILLAR_MAP[p])]
if not available_pillars:
    st.info("No numeric sub-parameter data available for drill-down in this year.")
else:
    drill_p = st.selectbox("Choose pillar", available_pillars)
    if st.button("Show sub-parameters"):
        subs = [m for m in PILLAR_MAP[drill_p] if m in base_to_num]
        if not subs:
            st.warning("No numeric sub-parameters for this pillar.")
        else:
            comp_sub = compare_df[[inst_col] + [base_to_num[s] for s in subs]].copy()
            # normalize each subparam using df_year values for parity
            norm_cols = []
            for s in subs:
                col_num = base_to_num[s]
                norm_col = f"{s}_norm"
                comp_sub[norm_col] = normalize_0_100(pd.to_numeric(df_year[col_num], errors="coerce"))
                norm_cols.append(norm_col)
            melt = comp_sub.melt(id_vars=inst_col, value_vars=norm_cols, var_name="Subparam", value_name="Score")
            # tidy subparam names
            melt["Subparam"] = melt["Subparam"].str.replace("_norm", "", regex=False)
            fig_sub = px.bar(melt, x=inst_col, y="Score", color="Subparam", barmode="group", height=520)
            st.plotly_chart(fig_sub, use_container_width=True)
            show_cols = [inst_col] + [f"raw::{s}" for s in subs if f"raw::{s}" in df_year.columns] + [base_to_num[s] for s in subs]
            st.dataframe(compare_df[show_cols].fillna(""), height=300)

# -----------------------------
# Strengths & Gaps
# -----------------------------
st.markdown("---")
st.subheader("Quick take: Strengths and Biggest gaps vs Top1")
top1 = compare_df.head(1).iloc[0] if not compare_df.empty else None
strengths, gaps = [], []
if top1 is not None:
    for p in pillars:
        s_val = sel_row.get(f"{p}_score", np.nan)
        t1_val = top1.get(f"{p}_score", np.nan)
        if pd.isna(s_val) or pd.isna(t1_val):
            continue
        diff = s_val - t1_val
        if diff >= 0:
            strengths.append((p, diff))
        else:
            gaps.append((p, -diff))

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Strengths")
    if strengths:
        for p, d in sorted(strengths, key=lambda x: -x[1])[:6]:
            st.success(f"{p}: +{d:.1f}")
    else:
        st.info("No strengths (or missing numeric data).")
with c2:
    st.markdown("### Biggest gaps")
    if gaps:
        for p, d in sorted(gaps, key=lambda x: -x[1])[:6]:
            st.error(f"{p}: {d:.1f} behind Top1")
    else:
        st.info("No gaps found (or missing numeric data).")

# -----------------------------
# Topk table (raw + parsed)
# -----------------------------
st.markdown("---")
st.subheader(f"Top {top_k} this year — raw & parsed numeric")
table_cols = [inst_col]
if rank_col and rank_col in df_year.columns:
    table_cols.append(rank_col)
for p in sorted({m for v in PILLAR_MAP.values() for m in v}):
    if f"raw::{p}" in df_year.columns:
        table_cols.append(f"raw::{p}")
    if p in base_to_num:
        table_cols.append(base_to_num[p])
table_df = df_sorted.head(top_k)[table_cols].copy()
if sel_inst not in table_df[inst_col].values:
    sel_small = df_year[df_year[inst_col].astype(str) == sel_inst][table_cols]
    if not sel_small.empty:
        table_df = pd.concat([table_df, sel_small], ignore_index=True)
st.dataframe(table_df.fillna(""), height=360)

st.markdown("---")
st.caption("Notes: `raw::X` = original CSV string; `X_num` = cleaned numeric value extracted by parser.")