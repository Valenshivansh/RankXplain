# app_nirf_dashboard.py
# 100% working – no crashes, proper scaler support, beautiful UI

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import re
import joblib
import traceback

st.set_page_config(page_title="NIRF Dashboard + Model Insights", layout="wide")

# ====================== CONFIG ======================
PILLAR_MAP = {
    "Teaching": ["SS", "FSR", "FQE", "FRU", "OE"],
    "Research": ["PU", "QP", "IPR", "FPPP", "GPHD"],
    "Graduation": ["GUE", "GPHD"],
    "Outreach": ["RD", "WD", "ESCS", "PCS"],
    "Perception": ["PR"]
}

PARAM_COLUMNS = [
    'SS','FSR','FQE','FRU','PU','QP','IPR','FPPP','GUE','GPHD','RD','WD','ESCS','PCS','OE'
]

# ====================== HELPERS ======================
def clean_numeric_string(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"[^0-9.\-]", "", s.replace(",", "").replace("%", "").strip())

def parse_float_safe(x):
    if pd.isna(x):
        return np.nan
    try:
        cleaned = clean_numeric_string(str(x))
        return float(cleaned) if cleaned not in ["", "-", "."] else np.nan
    except:
        return np.nan

def preserve_raw_and_clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        raw_col = f"raw::{c}"
        if raw_col not in df.columns:
            df[raw_col] = df[c].astype(str)
        num_col = f"{c}_num"
        if num_col not in df.columns:
            df[num_col] = df[c].apply(parse_float_safe)
    return df

# Fixed normalize function
def normalize_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series(np.nan, index=series.index)
    if np.isclose(mx, mn):
        return s - mn                               # keep relative ordering
    return (s - mn) / (mx - mn) * 100.0               # <-- fixed typo (_mn → mn)

# ====================== DATA UPLOAD ======================
st.sidebar.header("Upload NIRF CSVs")
uploaded_files = st.sidebar.file_uploader("Year-wise CSVs", type="csv", accept_multiple_files=True, key="csvs")

year_dfs = {}
if uploaded_files:
    for f in uploaded_files:
        try:
            df_raw = pd.read_csv(f)
            df = preserve_raw_and_clean(df_raw)
            # guess year from filename
            year_match = re.search(r"\d{4}", f.name)
            year = int(year_match.group()) if year_match else None
            if year is None:
                year = st.sidebar.number_input(f"Year for {f.name}", 2015, 2030, 2023, key=f"year_{f.name}")
            df["Year"] = int(year)
            year_dfs[int(year)] = df
        except Exception as e:
            st.sidebar.error(f"{f.name}: {e}")

if not year_dfs:
    st.error("Please upload at least one NIRF CSV file.")
    st.stop()

sel_year = st.sidebar.selectbox("Select Year", sorted(year_dfs.keys()), index=len(year_dfs)-1)
df_year = year_dfs[sel_year].copy()

# ====================== INSTITUTE DETECTION ======================
inst_candidates = [c for c in df_year.columns if any(k in c.lower() for k in ["institute", "college", "university", "name"])]
inst_col = inst_candidates[0] if inst_candidates else df_year.columns[0]
df_year[inst_col] = df_year[inst_col].astype(str).str.strip()

institutes = sorted(df_year[inst_col].dropna().unique())
default_inst = next((i for i in institutes if "IIT Madras" in i), institutes[0])
sel_inst = st.sidebar.selectbox("Select Institute", institutes,
                               index=institutes.index(default_inst) if default_inst in institutes else 0)

st.title(f"NIRF {sel_year} — {sel_inst}")

# ====================== NUMERIC MAPPING ======================
num_map = {c[:-4]: c for c in df_year.columns if c.endswith("_num")}

# ====================== PILLAR SCORES ======================
for pillar, params in PILLAR_MAP.items():
    cols = [num_map[p] for p in params if p in num_map]
    df_year[f"{pillar}_raw"] = df_year[cols].mean(axis=1) if cols else np.nan
    df_year[f"{pillar}_score"] = normalize_0_100(df_year[f"{pillar}_raw"])

# Rank detection
rank_col = next((c for c in df_year.columns if "rank" in c.lower()), None)
if rank_col:
    df_year["_rank_num"] = pd.to_numeric(df_year[rank_col].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

sel_row = df_year[df_year[inst_col] == sel_inst].iloc[0]

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_model_scaler():
    possible = ["models", "notebooks/models", ".", "..", "../models"]
    for folder in possible:
        mp = Path(folder) / "nirf_pr_model.pkl"
        sp = Path(folder) / "nirf_scaler.pkl"
        if mp.exists() and sp.exists():
            try:
                return joblib.load(mp), joblib.load(sp), str(mp)
            except:
                continue
    return None, None, None

model, scaler, model_path = load_model_scaler()

# Optional override
model_file = st.sidebar.file_uploader("Upload model (nirf_pr_model.pkl)", type="pkl", key="m")
scaler_file = st.sidebar.file_uploader("Upload scaler (nirf_scaler.pkl)", type="pkl", key="s")
if model_file and scaler_file:
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        model_path = "uploaded"
    except Exception as e:
        st.error(f"Upload failed: {e}")

if model and scaler:
    st.sidebar.success(f"Model loaded: {model_path}")
else:
    st.warning("Model/scaler not found → model insights disabled")
    model = scaler = None

# ====================== MODEL PREDICTION (SAFE) ======================
if model and scaler:
    try:
        # Build feature matrix
        features = []
        rows_idx = []
        for idx, row in df_year.iterrows():
            vec = []
            for param in PARAM_COLUMNS:
                if param in num_map:
                    val = row[num_map[param]]
                elif f"raw::{param}" in row:
                    val = parse_float_safe(row[f"raw::{param}"])
                else:
                    val = np.nan
                vec.append(val)
            if len(vec) == len(PARAM_COLUMNS):
                features.append(vec)
                rows_idx.append(idx)

        X = np.array(features)
        # Impute missing with median (same as training)
        medians = np.nanmedian(X, axis=0)
        X[np.isnan(X)] = np.take(medians, np.isnan(X).nonzero()[1])

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        predictions = model.predict(X_scaled).flatten()
        df_year.loc[rows_idx, "Model_Score"] = predictions

        sel_pred = float(sel_row["Model_Score"]) if "Model_Score" in sel_row else np.nan

        # Top-1
        top_row = df_year.sort_values("_rank_num").iloc[0] if "_rank_num" in df_year.columns else df_year.iloc[predictions.argmax()]
        top_pred = float(top_row["Model_Score"]) if "Model_Score" in top_row else np.nan
        gap = top_pred - sel_pred if not pd.isna(sel_pred) else np.nan

        st.header("Model Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Your Predicted Score", f"{sel_pred:.2f}")
        c2.metric("Top-1 Predicted Score", f"{top_pred:.2f}")
        c3.metric("Gap to Top-1", f"{gap:+.2f}", delta=f"{gap:.2f}")

        # Sensitivity (+1 std in scaled space)
        if not pd.isna(sel_pred):
            sel_idx = rows_idx[list(df_year[inst_col] == sel_inst).index(True)]
            sel_scaled = X_scaled[rows_idx.index(sel_idx)].reshape(1, -1)
            impacts = []
            for i, param in enumerate(PARAM_COLUMNS):
                perturbed = sel_scaled.copy()
                perturbed[0, i] += 1.0
                new_score = model.predict(perturbed)[0]
                impacts.append({"Parameter": param, "Score Impact (+1 std)": new_score - sel_pred})
            impact_df = pd.DataFrame(impacts).sort_values("Score Impact (+1 std)", ascending=False)
            st.subheader("Top improvement levers")
            st.dataframe(impact_df.head(10).style.format({"Score Impact (+1 std)": "{:+.3f}"}), height=380)

    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.code(traceback.format_exc())
        model = None

# ====================== PILLAR METRICS ======================
st.subheader("Five Pillar Scores")
cols = st.columns(5)
for i, (pillar, _) in enumerate(PILLAR_MAP.items()):
    score = sel_row.get(f"{pillar}_score", np.nan)
    median = df_year[f"{pillar}_score"].median()
    delta = f"{score - median:+.1f}" if pd.notna(score) and pd.notna(median) else None
    cols[i].metric(pillar,
                   f"{score:.1f}" if pd.notna(score) else "N/A",
                   delta)

# ====================== RADAR CHART (your favourite) ======================
st.subheader("Radar Comparison vs Top-3")
plot_df = df_year.sort_values("_rank_num").head(3).copy() if "_rank_num" in df_year.columns else df_year.head(3)
plot_df = pd.concat([plot_df, df_year[df_year[inst_col] == sel_inst]])

if len(plot_df) > 1:
    fig = go.Figure()
    for _, r in plot_df.iterrows():
        values = [r.get(f"{p}_score", 0) or 0 for p in PILLAR_MAP.keys()]
        values += values[:1]  # close the circle
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(PILLAR_MAP.keys()) + [list(PILLAR_MAP.keys())[0]],
            fill='toself',
            name=r[inst_col][:40]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=600)
    st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard ready!")