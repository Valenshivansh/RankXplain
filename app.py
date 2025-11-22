# app.py
# Streamlit NIRF Rank Explainer — Final with drill-down + 5 chart types + exports
# Requirements: streamlit, pandas, numpy, plotly
# Optional (for PNG export): kaleido, pillow
#
# Put this file in your project root and run:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io, traceback

st.set_page_config(page_title="NIRF Rank Explainer", layout="wide", initial_sidebar_state="expanded")

# ---------- File paths ----------
PRIMARY_PATH = Path("data/processed/nirf_combined_predicted.csv")   # preferred project path
FALLBACK_PATH = Path("/mnt/data/nirf_combined_predicted.csv")      # uploaded fallback (your file)

# ---------- Helpers ----------
@st.cache_data(ttl=600)
def load_csv(primary: Path, fallback: Path):
    p = primary if primary.exists() else (fallback if fallback.exists() else None)
    if p is None:
        return None, None
    try:
        df = pd.read_csv(p)
        return df, str(p)
    except Exception:
        try:
            df = pd.read_csv(p, on_bad_lines="skip")
            return df, str(p)
        except Exception as e:
            return None, str(p)

def coalesce_columns(df, primary, fallback):
    if primary in df.columns and fallback in df.columns:
        df[primary] = df[primary].fillna(df[fallback])
        df = df.drop(columns=[fallback])
    elif fallback in df.columns and primary not in df.columns:
        df = df.rename(columns={fallback: primary})
    return df

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalize_series(s):
    s = pd.Series(s, dtype=float)
    amin = s.min(skipna=True)
    amax = s.max(skipna=True)
    if pd.isna(amin) or pd.isna(amax) or np.isclose(amax, amin):
        return pd.Series([50.0]*len(s), index=s.index)
    return (s - amin) / (amax - amin) * 100

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def try_fig_to_png(fig):
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception:
        return None

# ---------- Load dataset ----------
df, loaded_from = load_csv(PRIMARY_PATH, FALLBACK_PATH)
if df is None:
    st.error(f"Could not find dataset in `{PRIMARY_PATH}` or fallback `{FALLBACK_PATH}`. Put the file there and rerun.")
    st.stop()

# ---------- Basic clean & column coalesce ----------
df = coalesce_columns(df, "Institute Name", "Institution Name")
df = coalesce_columns(df, "Institute Code", "Institution Code")
df.columns = [c.strip() for c in df.columns]  # trim spaces

if "Year" not in df.columns:
    st.error("Dataset missing 'Year' column. Add it and try again.")
    st.stop()

# ---------- Pillar mapping (you said TLR divided in parts etc.) ----------
PILLAR_MAP = {
    "Teaching": ["SS", "FSR", "FQE", "FRU", "OE"],   # Teaching learning & resources
    "Research": ["PU", "QP", "IPR", "FPPP", "GPHD"], # Research & professional practice
    "Graduation": ["GUE", "GPHD"],                  # Graduation outcomes
    "Outreach": ["RD", "WD", "ESCS", "PCS"],         # Outreach and Inclusivity
    "Perception": ["PR"]                             # Perception
}

# ---------- Convert numeric columns (subparams and rank fields) ----------
all_subs = sorted({x for v in PILLAR_MAP.values() for x in v})
df = safe_numeric(df, all_subs + ["Rank", "Predicted_Rank", "Official_Overall_Rank"])

# fill numeric missing with median for stability (so charts render)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---------- UI: Sidebar controls ----------
st.sidebar.title("Filters")
years = sorted(df["Year"].dropna().unique().astype(int).tolist())
sel_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
df_year = df[df["Year"] == sel_year].copy()
if df_year.empty:
    st.error(f"No data for year {sel_year}.")
    st.stop()

institutes = sorted(df_year["Institute Name"].dropna().unique().tolist())
default_inst = "Indian Institute of Technology Madras" if "Indian Institute of Technology Madras" in institutes else institutes[0]
sel_inst = st.sidebar.selectbox("Select Institute", institutes, index=institutes.index(default_inst))
top_k = 3  # fixed Top 3 as requested
st.sidebar.markdown("---")
st.sidebar.markdown("Made for students & parents | NIRF India 2018–2024")

# ---------- Header & subtitle ----------
st.title("NIRF Rank Explainer")
st.markdown("<h3 style='text-align:center;'>Select a year and an institute — the dashboard explains the why in plain language.</h3>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Compute pillar raw and normalized scores for the chosen year ----------
for pillar, members in PILLAR_MAP.items():
    present = [c for c in members if c in df_year.columns]
    if present:
        df_year[pillar + "_raw"] = df_year[present].mean(axis=1)
    else:
        df_year[pillar + "_raw"] = np.nan

for pillar in PILLAR_MAP.keys():
    df_year[pillar + "_score"] = normalize_series(df_year[pillar + "_raw"])

# ---------- Pick the ranking column to use ----------
rank_candidates = []
if "Official_Overall_Rank" in df_year.columns and df_year["Official_Overall_Rank"].notna().any():
    rank_candidates.append("Official_Overall_Rank")
if "Rank" in df_year.columns and df_year["Rank"].notna().any():
    rank_candidates.append("Rank")
if "Predicted_Rank" in df_year.columns and df_year["Predicted_Rank"].notna().any():
    rank_candidates.append("Predicted_Rank")
rank_col = rank_candidates[0] if rank_candidates else None
if rank_col is None:
    st.error("No ranking column found. Make sure 'Rank' or 'Predicted_Rank' or 'Official_Overall_Rank' exists.")
    st.stop()

# ---------- Selected institute row & ranking display ----------
sel_row = df_year[df_year["Institute Name"] == sel_inst].iloc[0]

try:
    display_rank = int(float(sel_row.get(rank_col))) if not pd.isna(sel_row.get(rank_col)) else None
except:
    display_rank = sel_row.get(rank_col, None)

c1, c2 = st.columns([3,1])
with c1:
    st.markdown(f"**{sel_row['Institute Name']} — Rank #{display_rank if display_rank else '—'} ({sel_year})**")
    if display_rank == 1:
        st.success("🎉 Rank #1 — outstanding!")
    elif display_rank and display_rank <= 5:
        st.success(f"🏆 Top {display_rank} institute")
    else:
        st.info("✅ See strengths and gaps below.")
with c2:
    if display_rank == 1:
        st.balloons()

st.markdown("")

# ---------- Big pillar metrics (large) ----------
emoji = {"Teaching":"📚","Research":"🔬","Graduation":"🎓","Outreach":"🤝","Perception":"🌟"}
cols = st.columns(5)
pillars = list(PILLAR_MAP.keys())
for i,p in enumerate(pillars):
    val = float(sel_row[p + "_score"]) if not pd.isna(sel_row[p + "_score"]) else 0.0
    median = float(df_year[p + "_score"].median())
    delta = val - median
    cols[i].metric(label=f"{emoji.get(p)} {p}", value=f"{val:.1f}/100", delta=f"{delta:.1f}")

st.markdown("---")

# ---------- Chart selector (main screen) ----------
chart_type = st.selectbox("Choose comparison chart type", [
    "Grouped Bar (Top 3 + Selected)",
    "Radar (Selected vs Top3 avg vs #1)",
    "Parallel Coordinates (Top 3 + Selected)",
    "Line (Pillar trends for Top 3 + Selected)",
    "Polar / Star Chart (Aesthetic)"
])

# ---------- Build compare_df: Top 3 + selected institute (unique) ----------
df_sorted = df_year.sort_values(by=rank_col)
top3_df = df_sorted.head(top_k).copy()
selected_df = df_year[df_year["Institute Name"] == sel_inst].copy()
compare_df = pd.concat([top3_df, selected_df], ignore_index=True).drop_duplicates(subset=["Institute Name"])
# reorder with top3 first then selected at end if not in top3
compare_df_names = top3_df["Institute Name"].tolist()
if sel_inst not in compare_df_names:
    # keep top3 then selected
    compare_df = pd.concat([top3_df, selected_df], ignore_index=True).drop_duplicates(subset=["Institute Name"])

# values for plotting
sel_vals = [float(sel_row[p + "_score"]) if not pd.isna(sel_row[p + "_score"]) else 0.0 for p in pillars]
top3_avg_vals = compare_df.head(top_k)[[p + "_score" for p in pillars]].mean().values.tolist()
top1_vals = compare_df.head(1)[[p + "_score" for p in pillars]].iloc[0].values.tolist()

# ---------- Plot based on selection ----------
st.subheader("Comparison: selected institute vs top performers")
fig = None

if chart_type.startswith("Grouped Bar"):
    bar_df = compare_df[["Institute Name"] + [p + "_score" for p in pillars]].copy()
    bar_df = bar_df.rename(columns={p + "_score": p for p in pillars})
    melt = bar_df.reset_index().melt(id_vars="Institute Name", var_name="Pillar", value_name="Score")
    fig = px.bar(melt, x="Institute Name", y="Score", color="Pillar", barmode="group",
                 title=f"Top {top_k} + {sel_inst} by pillar", height=520)
    fig.update_layout(xaxis_tickangle=-30)

elif chart_type.startswith("Radar"):
    cats = pillars + [pillars[0]]
    sel_plot = sel_vals + [sel_vals[0]]
    top_avg_plot = top3_avg_vals + [top3_avg_vals[0]]
    top1_plot = top1_vals + [top1_vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=top_avg_plot, theta=cats, fill='toself', name=f"Top {top_k} avg", marker=dict(color='rgba(31,119,180,0.5)')))
    fig.add_trace(go.Scatterpolar(r=sel_plot, theta=cats, fill='toself', name=sel_inst, marker=dict(color='rgba(44,160,44,0.9)')))
    fig.add_trace(go.Scatterpolar(r=top1_plot, theta=cats, fill='toself', name=f"{compare_df.iloc[0]['Institute Name']} (#1)", marker=dict(color='gold')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), height=520, legend=dict(orientation="h"))

elif chart_type.startswith("Parallel"):
    pc_df = compare_df[["Institute Name"] + [p + "_score" for p in pillars]].copy()
    pc_df = pc_df.rename(columns={p + "_score": p for p in pillars})
    # add numeric label for color
    pc_df["rank_order"] = range(1, len(pc_df)+1)
    fig = px.parallel_coordinates(pc_df, color="rank_order", dimensions=pillars,
                                  color_continuous_scale=px.colors.diverging.Tealrose, title="Parallel Coordinates (Top 3 + Selected)", height=520)

elif chart_type.startswith("Line"):
    # each institute is a line across pillars
    ln_df = compare_df[["Institute Name"] + [p + "_score" for p in pillars]].copy()
    ln_df = ln_df.rename(columns={p + "_score": p for p in pillars})
    fig = go.Figure()
    for _, r in ln_df.iterrows():
        name = r["Institute Name"]
        vals = [r[p] for p in pillars]
        fig.add_trace(go.Scatter(x=pillars, y=vals, mode='lines+markers', name=name))
    fig.update_layout(title=f"Pillar lines — Top {top_k} + {sel_inst}", height=520, yaxis=dict(range=[0,100]))

elif chart_type.startswith("Polar"):
    # Polar / star plot: a different aesthetic, use barpolar for each institute stacked
    pol_df = compare_df[["Institute Name"] + [p + "_score" for p in pillars]].copy()
    pol_df = pol_df.rename(columns={p + "_score": p for p in pillars})
    fig = go.Figure()
    theta = pillars
    for _, r in pol_df.iterrows():
        vals = [r[p] for p in pillars]
        fig.add_trace(go.Barpolar(r=vals, theta=theta, name=r["Institute Name"], opacity=0.7))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])), height=520)

# show chart
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

# ---------- Export area ----------
st.markdown("---")
st.subheader("Download / Export")
# CSV export for the compare table
export_small_cols = ["Institute Name", rank_col] + [p + "_raw" for p in pillars] + [p + "_score" for p in pillars]
export_small = compare_df[[c for c in export_small_cols if c in compare_df.columns]].copy()
st.download_button("Download comparison CSV", data=df_to_csv_bytes(export_small),
                   file_name=f"comparison_{sel_year}_{sel_inst.replace(' ','_')}.csv", mime="text/csv")

# PNG export (attempt)
png_bytes = try_fig_to_png(fig) if fig is not None else None
if png_bytes:
    st.download_button("Download chart PNG", data=png_bytes, file_name=f"chart_{sel_year}_{sel_inst.replace(' ','_')}.png", mime="image/png")
else:
    st.info("To enable PNG export of charts, install `kaleido` in your venv: `pip install kaleido pillow`")

# ---------- Drilldown: show sub-parameters of a pillar when user requests ----------
st.markdown("---")
st.subheader("Drill-down: expand a pillar to see its sub-parameters")
pillars_available = [p for p in PILLAR_MAP.keys() if any([c in df_year.columns for c in PILLAR_MAP[p]])]
drill_pillar = st.selectbox("Choose a pillar to expand", pillars_available)
if st.button("Show sub-parameters for selected pillar"):
    subs = [c for c in PILLAR_MAP[drill_pillar] if c in df_year.columns]
    if not subs:
        st.warning("No sub-parameters available for this pillar in this year.")
    else:
        # Prepare comparison df for subparams: Top3 + selected
        comp_sub = compare_df[["Institute Name"] + subs].copy()
        # Normalize each sub to 0-100 within this year for visual parity
        for s in subs:
            comp_sub[s] = pd.to_numeric(comp_sub[s], errors="coerce")
            comp_sub[s] = normalize_series(comp_sub[s])
        # Melt and plot grouped bars
        melt_sub = comp_sub.melt(id_vars="Institute Name", var_name="Subparam", value_name="Score")
        fig_sub = px.bar(melt_sub, x="Institute Name", y="Score", color="Subparam", barmode="group",
                         title=f"{drill_pillar} — sub-parameters (Top {top_k} + selected)", height=520)
        fig_sub.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig_sub, use_container_width=True)

        # Also show numeric table
        st.dataframe(comp_sub.style.format({c:"{:.2f}" for c in comp_sub.columns if c!="Institute Name"}), height=300)

# ---------- Strengths & Gaps vs #1 ----------
st.markdown("---")
st.subheader("Quick take: Strengths (green) and largest gaps vs #1 (red)")
strengths=[]; gaps=[]
top1_row = compare_df.head(1).iloc[0]
for p in pillars:
    s_val = float(sel_row[p + "_score"]) if not pd.isna(sel_row[p + "_score"]) else 0.0
    t1_val = float(top1_row[p + "_score"]) if not pd.isna(top1_row[p + "_score"]) else 0.0
    diff = s_val - t1_val
    if diff >= 0:
        strengths.append((p, diff))
    else:
        gaps.append((p, -diff))

sc, gc = st.columns(2)
with sc:
    st.markdown("### Strengths")
    if strengths:
        for p,d in sorted(strengths, key=lambda x:-x[1])[:6]:
            st.success(f"{p}: +{d:.1f} vs #1")
    else:
        st.info("No strengths vs #1")

with gc:
    st.markdown("### Biggest gaps")
    if gaps:
        for p,d in sorted(gaps, key=lambda x:-x[1])[:6]:
            st.error(f"{p}: {d:.1f} behind #1")
    else:
        st.info("No meaningful gaps vs #1")

# ---------- Compact Top 3 table (plus selected if not in top3) ----------
st.markdown("---")
st.subheader(f"Top {top_k} institutes — Year {sel_year}")
table_cols = ["Institute Name", rank_col] + [p + "_raw" for p in pillars] + [p + "_score" for p in pillars]
table_cols = [c for c in table_cols if c in df_year.columns]
table_df = df_year.sort_values(by=rank_col).head(top_k)[table_cols].copy()
if sel_inst not in table_df["Institute Name"].values:
    sel_small = df_year[df_year["Institute Name"]==sel_inst][table_cols]
    if not sel_small.empty:
        table_df = pd.concat([table_df, sel_small], ignore_index=True)
st.dataframe(table_df.style.format({c:"{:.2f}" for c in table_df.columns if c not in ["Institute Name", rank_col]}), height=320)

# ---------- Tutorial for parents ----------
st.markdown("---")
with st.expander("How to read these charts (quick parent-friendly guide)"):
    st.markdown("**Radar chart**: each axis is a pillar (0–100). Larger shaded area = more balanced strength.")
    st.markdown("**Grouped bar**: compare institutes pillar-by-pillar (best for quick visual reading).")
    st.markdown("**Parallel coordinates**: each line = an institute across pillars; consistent high lines mean consistently strong.")
    st.markdown("**Line chart**: shows pillar values as lines (easy to compare shapes).")
    st.markdown("**Polar/Star**: visual style similar to radar but different aesthetics.")
    st.markdown("**Drill-down**: pick a pillar and click 'Show sub-parameters' to see its internal components (teaching broken into FSR, FQE, etc.).")

st.markdown("---")
st.markdown("Made for students & parents | NIRF India 2018–2024 · Built with ❤️ by Shivansh")