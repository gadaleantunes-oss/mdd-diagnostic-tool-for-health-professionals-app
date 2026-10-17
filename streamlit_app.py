import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind

# =====================================================================
# PALETTE (taken from the "Random Forest Observatory" visualizer)
# =====================================================================
BG = "#030305"
PANEL = "#08080a"
BORDER = "#2a1b3d"
TEXT = "#e0aaff"
TEXT_DIM = "#7b2cbf"
NEON = "#c77dff"
PINK = "#f72585"
DEPTH = ["#3c096c", "#5a189a", "#7b2cbf", "#9d4edd", "#c77dff"]
FONT = "Courier New, Courier, monospace"

# Volcano plot colors: down (blue) / normal (gray) / up (red)
COLOR_DOWN = "#2979FF"
COLOR_NORMAL = "#8A8F98"
COLOR_UP = "#FF1744"

PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "purple_obs", ["#0f081c"] + DEPTH + ["#f3d9ff"]
)

# =====================================================================
# PAGE CONFIGURATION AND THEME
# =====================================================================
st.set_page_config(page_title="MDD BioMarker Tracker", page_icon="🧬", layout="wide")

st.markdown(f"""
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {{
        font-family: {FONT} !important;
    }}
    .stApp {{
        background: radial-gradient(circle at 50% 0%, #0f081c 0%, {BG} 60%);
        color: {TEXT};
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    h1 {{
        color: #ffffff !important;
        font-weight: normal !important;
        letter-spacing: 2px;
        text-shadow: 0 0 10px {NEON};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 12px;
    }}
    h2, h3, h4 {{ color: {NEON} !important; letter-spacing: 1px; }}
    p, li, span {{ color: {TEXT}; }}
    hr {{ border-color: {BORDER} !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: transparent; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PANEL}; border-radius: 6px 6px 0 0; color: {TEXT_DIM};
        padding: 10px 20px; border: 1px solid {BORDER}; border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #3c096c; color: #ffffff !important;
        border-color: #9d4edd; text-shadow: 0 0 8px {NEON};
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {NEON}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] h2 {{ font-size: 1.05rem; text-transform: uppercase; letter-spacing: 2px; }}
    [data-testid="stFileUploaderDropzone"] {{ background-color: #0f081c; border: 1px dashed #5a189a; }}

    /* Buttons */
    .stButton>button, [data-testid="stFileUploaderDropzone"] button {{
        background-color: #5a189a; color: white; border: 1px solid #9d4edd; border-radius: 6px;
    }}
    .stButton>button:hover, [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: #9d4edd; color: white; border-color: {NEON}; box-shadow: 0 0 10px {NEON}55;
    }}

    /* Sliders / inputs */
    [data-testid="stSlider"] [role="slider"] {{ background-color: {NEON}; box-shadow: 0 0 8px {NEON}; }}
    [data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: {NEON}; }}
    [data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInput"] input {{
        background-color: #0f081c !important; border-color: {BORDER} !important; color: {TEXT} !important;
    }}

    /* Metrics and expanders as "panels" */
    [data-testid="stMetric"] {{
        background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 14px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_DIM}; text-transform: uppercase; letter-spacing: 1px; }}
    [data-testid="stMetricValue"] {{ color: #ffffff; text-shadow: 0 0 10px {NEON}; }}
    [data-testid="stExpander"] {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}

    /* Progress bar */
    .stProgress > div > div > div > div {{ background-image: linear-gradient(90deg, #3c096c, {NEON}); }}

    /* Alerts */
    [data-testid="stAlert"] {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
    </style>
""", unsafe_allow_html=True)

st.title("🧬 MDD BioMarker Tracker")
st.markdown("Interactive platform for screening biological markers of **Major Depressive Disorder (MDD)**.")


def style_plotly(fig, **kwargs):
    """Applies the dark purple look to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT),
        legend=dict(bgcolor="rgba(8,8,10,0.6)", bordercolor=BORDER, borderwidth=1),
        **kwargs,
    )
    if fig.layout.title.text:  # avoids a stray "undefined" title on untitled charts
        fig.update_layout(title_font=dict(color=NEON))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=DEPTH[2], linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=DEPTH[2], linecolor=BORDER)
    return fig


# =====================================================================
# LITERATURE SIGNATURE
# =====================================================================
LITERATURE_SIGNATURE = {
    'KRT23': 'UP', 'PLSCR1': 'UP', 'CLEC4A': 'UP',
    'CAPRIN1': 'UP', 'PROK2': 'UP', 'ZBTB16': 'DOWN', 'MLC1': 'DOWN'
}

# =====================================================================
# PROCESSING ENGINE
# =====================================================================
def process_data(file):
    raw_text = file.getvalue().decode("utf-8", errors='ignore')
    lines = raw_text.splitlines()

    rows_to_skip = 0
    for i, line in enumerate(lines):
        if line.startswith('FEATURES'):
            rows_to_skip = i
            break
        elif '!series_matrix_table_begin' in line:
            rows_to_skip = i + 1
            break

    separator = '\t' if file.name.endswith('.txt') else ','
    df = pd.read_csv(io.StringIO(raw_text), sep=separator, skiprows=rows_to_skip, low_memory=False)

    is_agilent = 'LogRatio' in df.columns and 'PValueLogRatio' in df.columns

    # 1. LOGIC FOR AGILENT FILES
    if is_agilent:
        if 'ControlType' in df.columns:
            df = df[df['ControlType'] == 0].copy()

        df['LogRatio'] = pd.to_numeric(df['LogRatio'], errors='coerce')
        df['PValueLogRatio'] = pd.to_numeric(df['PValueLogRatio'], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['LogRatio', 'PValueLogRatio'])

        p_vals = np.clip(df['PValueLogRatio'], 1e-300, 1.0)
        df['Log10_P'] = -np.log10(p_vals)

        df['GeneTemp'] = df.get('GeneName', np.nan)
        for col in ['SystematicName', 'ProbeUID', 'ProbeName', 'ID_REF']:
            if col in df.columns:
                df['GeneTemp'] = df['GeneTemp'].fillna(df[col])

        df['GeneTemp'] = df['GeneTemp'].astype(str).str.upper()

        df_exp = pd.DataFrame()
        col_g = next((c for c in ['gProcessedSignal', 'gDyeNormSignal', 'gMeanSignal'] if c in df.columns), None)
        col_r = next((c for c in ['rProcessedSignal', 'rDyeNormSignal', 'rMeanSignal'] if c in df.columns), None)

        if col_g and col_r:
            g_signal = pd.to_numeric(df[col_g], errors='coerce')
            r_signal = pd.to_numeric(df[col_r], errors='coerce')

            df_signal = pd.DataFrame({
                'Gene': df['GeneTemp'],
                'Control (Cy3)': np.log2(np.maximum(g_signal, 1)),
                'MDD (Cy5)': np.log2(np.maximum(r_signal, 1))
            }).dropna()
            df_exp = df_signal.groupby('Gene').mean()
            cols_control = ['Control (Cy3)']
            cols_mdd = ['MDD (Cy5)']
        else:
            cols_control = []
            cols_mdd = []

        df_stats = pd.DataFrame({
            'Gene': df['GeneTemp'],
            'Log2FC': df['LogRatio'],
            'Log10_P': df['Log10_P'],
            'P_Value': df['PValueLogRatio']
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=['Log2FC', 'Log10_P'])

        return df_exp, df_stats, cols_control, cols_mdd, True

    # 2. LOGIC FOR GEO MATRICES
    else:
        if 'GeneName' in df.columns: df = df.set_index('GeneName')
        elif 'Symbol' in df.columns: df = df.set_index('Symbol')
        elif 'ID_REF' in df.columns: df = df.set_index('ID_REF')
        else: df = df.set_index(df.columns[0])

        df = df[df.index.notnull()]
        df = df[~df.index.astype(str).str.contains('Unknown|Unidentified', case=False, na=False)]
        df.index = df.index.astype(str).str.upper()

        df_num = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        df_num = df_num.groupby(level=0).mean(numeric_only=True)
        if not df_num.empty and df_num.max().max() > 100:
            df_num = np.log2(df_num + 1)

        cols = df_num.columns
        half = len(cols) // 2
        cols_control, cols_mdd = cols[:half], cols[half:]

        log2fc = df_num[cols_mdd].mean(axis=1) - df_num[cols_control].mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_val = ttest_ind(df_num[cols_mdd], df_num[cols_control], axis=1)

        p_val = np.nan_to_num(p_val, nan=1.0)
        log10_p = -np.log10(np.clip(p_val, 1e-300, 1.0))

        df_stats = pd.DataFrame({
            'Gene': df_num.index,
            'Log2FC': log2fc,
            'P_Value': p_val,
            'Log10_P': log10_p
        }, index=df_num.index).replace([np.inf, -np.inf], np.nan).dropna(subset=['Log2FC', 'Log10_P']).reset_index(drop=True)

        return df_num, df_stats, cols_control, cols_mdd, False

# =====================================================================
# MAIN INTERFACE
# =====================================================================
st.sidebar.header("📥 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Matrix", type=['csv', 'txt'])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Statistical Filters")
p_val_thresh = st.sidebar.slider("P-value Cutoff (-Log10)", 1.0, 5.0, 1.3)
fc_thresh = st.sidebar.slider("Log2 Fold Change Cutoff", 0.1, 3.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Control / MDD Split")
n_control_input = st.sidebar.number_input(
    "Number of CONTROL columns (the first N columns)",
    min_value=1, value=33, step=1,
    help="Define how many of the first columns in the matrix are Control samples. The remaining columns will be MDD."
)

if uploaded_file:
    with st.spinner('Processing genomic matrix...'):
        df_exp, df_stats, cols_control, cols_mdd, is_agilent = process_data(uploaded_file)

    if not is_agilent:
        n_total = len(df_exp.columns)
        n_ctrl = min(int(n_control_input), n_total - 1)
        cols_control = df_exp.columns[:n_ctrl]
        cols_mdd = df_exp.columns[n_ctrl:]

        log2fc = df_exp[cols_mdd].mean(axis=1) - df_exp[cols_control].mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_val = ttest_ind(df_exp[cols_mdd], df_exp[cols_control], axis=1)
        p_val = np.nan_to_num(p_val, nan=1.0)
        log10_p = -np.log10(np.clip(p_val, 1e-300, 1.0))
        df_stats = pd.DataFrame({
            'Gene': df_exp.index,
            'Log2FC': log2fc,
            'P_Value': p_val,
            'Log10_P': log10_p
        }, index=df_exp.index).replace([np.inf, -np.inf], np.nan).dropna(subset=['Log2FC', 'Log10_P']).reset_index(drop=True)

    st.success(f"File processed! Probes: {len(df_stats)} | Control: {len(cols_control)} samples | MDD: {len(cols_mdd)} samples")

    tab1, tab2, tab3 = st.tabs(["🌋 Volcano Plot", "🎯 Clinical Diagnosis", "🧬 Expression Heatmap"])

    # ---------------------------------------------------------------------
    # TAB 1: VOLCANO PLOT
    # ---------------------------------------------------------------------
    with tab1:
        st.subheader("🌋 Interactive Volcano Plot")

        df_clean = df_stats.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log2FC', 'Log10_P']).copy()

        def categorize(row):
            if row['Gene'] in LITERATURE_SIGNATURE.keys():
                return 'MDD Signature'
            elif row['Log10_P'] > p_val_thresh:
                if row['Log2FC'] > fc_thresh:
                    return 'Upregulated'
                elif row['Log2FC'] < -fc_thresh:
                    return 'Downregulated'
            return 'Not Significant'

        df_clean['Status'] = df_clean.apply(categorize, axis=1)

        fig_volcano = px.scatter(
            df_clean, x='Log2FC', y='Log10_P', color='Status', hover_name='Gene',
            hover_data={'Log2FC': ':.3f', 'Log10_P': ':.2f', 'P_Value': ':.2e', 'Status': True},
            color_discrete_map={
                'Downregulated': COLOR_DOWN,
                'Not Significant': COLOR_NORMAL,
                'Upregulated': COLOR_UP,
                'MDD Signature': '#FFFFFF',
            },
            category_orders={'Status': ['Downregulated', 'Not Significant', 'Upregulated', 'MDD Signature']},
            opacity=1, render_mode='webgl'
        )
        # Flat, solid markers: no transparency or outlines, so no shading where points overlap
        fig_volcano.update_traces(marker=dict(size=5, line=dict(width=0)))

        df_sig = df_clean[df_clean['Status'] == 'MDD Signature']
        if not df_sig.empty:
            fig_volcano.add_scatter(
                x=df_sig['Log2FC'], y=df_sig['Log10_P'], text=df_sig['Gene'], mode='markers+text',
                marker=dict(color='#FFFFFF', size=12, line=dict(color=NEON, width=2)),
                textposition='top center', textfont=dict(color='#FFFFFF', size=13), showlegend=False
            )

        fig_volcano.add_hline(y=p_val_thresh, line_dash="dash", line_color=COLOR_NORMAL)
        fig_volcano.add_vline(x=fc_thresh, line_dash="dash", line_color=COLOR_UP)
        fig_volcano.add_vline(x=-fc_thresh, line_dash="dash", line_color=COLOR_DOWN)

        style_plotly(fig_volcano)
        st.plotly_chart(fig_volcano, use_container_width=True)

    # ---------------------------------------------------------------------
    # TAB 2: CLINICAL DIAGNOSIS
    # ---------------------------------------------------------------------
    with tab2:
        st.subheader("🎯 Molecular Diagnosis Center")

        st.markdown("### 1. Overview: MDD Cohort vs Control")
        genes_found = df_stats[df_stats['Gene'].isin(LITERATURE_SIGNATURE.keys())]

        if not genes_found.empty:
            genes_found = genes_found.groupby('Gene').mean(numeric_only=True)

        points = 0
        details = []
        for gene, expected_dir in LITERATURE_SIGNATURE.items():
            if gene in genes_found.index:
                actual_fc = genes_found.loc[gene, 'Log2FC']
                actual_dir = 'UP' if actual_fc > 0 else 'DOWN'
                if actual_dir == expected_dir:
                    points += 1
                    status = "✅ Confirms"
                else:
                    status = "❌ Reversed"
                details.append(f"**{gene}**: Expected {expected_dir} | Observed {actual_dir} (Log2FC: {actual_fc:.2f}) -> {status}")
            else:
                details.append(f"**{gene}**: Not found in matrix")

        prob = (points / len(LITERATURE_SIGNATURE)) * 100

        col_grp1, col_grp2 = st.columns([1, 2])
        with col_grp1:
            st.metric("Average MDD Cohort Alignment", f"{prob:.1f}%")
            st.progress(int(prob))
        with col_grp2:
            with st.expander("View gene breakdown for the experimental group"):
                for d in details: st.markdown(f"- {d}")

        st.divider()

        st.markdown("### 👤 2. Individual Screening (Precision Medicine)")
        st.markdown("Assess the expression deviation of **a single patient** relative to the healthy control group mean to confirm the clinical signature.")

        if df_exp.empty:
            st.warning("⚠️ Intensity/sample data not found in the matrix, so an individual profile cannot be generated.")
        else:
            if is_agilent:
                st.info("Agilent Dual-Color matrix detected. Analyzing the Cy5 sample (MDD) against the Cy3 reference (Control).")
                selected_sample = "Patient Cy5"

                df_ind = df_stats[df_stats['Gene'].isin(LITERATURE_SIGNATURE.keys())].copy()
                df_ind['Log2FC_Individual'] = df_ind['Log2FC']
            else:
                col_sel1, col_sel2 = st.columns([1, 2])
                with col_sel1:
                    selected_sample = st.selectbox("Select the Patient (MDD Sample):", cols_mdd)

                control_mean = df_exp[cols_control].mean(axis=1)
                patient_exp = df_exp[selected_sample]
                log2fc_ind = patient_exp - control_mean

                df_ind = pd.DataFrame({
                    'Gene': df_exp.index,
                    'Log2FC_Individual': log2fc_ind
                })
                df_ind = df_ind[df_ind['Gene'].isin(LITERATURE_SIGNATURE.keys())].copy()

            if df_ind.empty:
                st.warning("No MDD signature genes found in this sample.")
            else:
                COLOR_MATCH, COLOR_REVERSED = NEON, PINK

                df_ind['Expected'] = df_ind['Gene'].map(LITERATURE_SIGNATURE)
                df_ind['Color'] = df_ind.apply(
                    lambda x: COLOR_MATCH if (x['Log2FC_Individual'] > 0 and x['Expected'] == 'UP') or
                                             (x['Log2FC_Individual'] < 0 and x['Expected'] == 'DOWN') else COLOR_REVERSED,
                    axis=1
                )
                df_ind['Match'] = df_ind.apply(
                    lambda x: '✅ Match' if x['Color'] == COLOR_MATCH else '❌ Reversed', axis=1
                )

                col_chart, col_metrics = st.columns([2, 1])

                with col_chart:
                    fig_ind = px.bar(
                        df_ind, x='Log2FC_Individual', y='Gene', orientation='h',
                        color='Color', color_discrete_map="identity",
                        title=f"Signature Expression Deviation: {selected_sample}",
                        text='Log2FC_Individual', hover_data={'Expected': True, 'Match': True, 'Color': False}
                    )

                    fig_ind.update_traces(texttemplate='%{text:.2f}', textposition='outside',
                                          marker_line_color='#ffffff', marker_line_width=0.5)
                    style_plotly(
                        fig_ind,
                        xaxis_title="Patient Deviation (Log2 Fold Change vs Controls)",
                        yaxis_title="MDD Markers",
                        height=400,
                    )
                    fig_ind.add_vline(x=0, line_width=2, line_color="white")
                    st.plotly_chart(fig_ind, use_container_width=True)

                with col_metrics:
                    ind_points = sum(df_ind['Color'] == COLOR_MATCH)
                    ind_total = len(df_ind)
                    ind_score = (ind_points / ind_total) * 100 if ind_total > 0 else 0

                    st.metric(f"Clinical Score - {selected_sample}", f"{ind_score:.1f}%", f"{ind_points} of {ind_total} genes confirmed")

                    if ind_score >= 70:
                        st.error("🚨 **HIGH RISK:** Patient shows a strong MDD transcriptomic signature.")
                    elif ind_score >= 40:
                        st.warning("⚠️ **MODERATE RISK:** Patient shows an intermediate/mixed clinical profile.")
                    else:
                        st.success("✅ **LOW RISK:** Patient shows low molecular correlation with MDD.")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Chart Legend:**")
                    st.markdown(f"<span style='color:{COLOR_MATCH}'>■ **Lilac:**</span> direction matches the clinical literature.", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:{COLOR_REVERSED}'>■ **Pink:**</span> gene direction opposes the literature.", unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # TAB 3: EXPRESSION HEATMAP
    # ---------------------------------------------------------------------
    with tab3:
        st.subheader("🧬 Expression Heatmap (Top Genes + Signature)")
        st.markdown("Global view of expression patterns across Control and MDD samples.")

        if df_exp.empty:
            st.warning("⚠️ **Heatmap Unavailable:** Requires matrices with intensity or sample data.")
        else:
            top_genes = df_stats.nlargest(30, 'Log10_P')['Gene'].tolist()
            heatmap_targets = list(set(top_genes + list(LITERATURE_SIGNATURE.keys())))

            targets_present = df_exp.index.intersection(heatmap_targets)
            if not targets_present.empty:
                df_heat = df_exp.loc[targets_present].copy()

                if not is_agilent:
                    new_names = {}
                    for i, c in enumerate(df_heat.columns):
                        if c in cols_control:
                            new_names[c] = f"C{list(cols_control).index(c)+1}"
                        else:
                            new_names[c] = f"M{list(cols_mdd).index(c)+1}"
                    df_heat = df_heat.rename(columns=new_names)

                # Row-wise z-score (vectorized; always returns a DataFrame)
                row_mean = df_heat.mean(axis=1)
                row_std = df_heat.std(axis=1, ddof=0).replace(0, np.nan)
                df_heat = df_heat.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)

                plt.style.use('dark_background')
                plt.rcParams['font.family'] = 'monospace'
                n_cols = len(df_heat.columns)
                fig_width = max(14, n_cols * 0.4)
                fig, ax = plt.subplots(figsize=(fig_width, 10))
                fig.patch.set_facecolor(BG)
                ax.set_facecolor(BG)

                sns.heatmap(df_heat, cmap=PURPLE_CMAP, center=0, ax=ax, yticklabels=True,
                            xticklabels=True, cbar_kws={'label': 'Z-Score'},
                            linecolor=PANEL, linewidths=0.5)

                for label in ax.get_xticklabels():
                    txt = label.get_text()
                    if txt.startswith('C'): label.set_color(NEON)
                    elif txt.startswith('M'): label.set_color(PINK)
                    label.set_fontsize(8)
                ax.set_xlabel('C = Control | M = MDD', color=TEXT, fontsize=12)

                for label in ax.get_yticklabels():
                    if label.get_text() in LITERATURE_SIGNATURE.keys():
                        label.set_color(PINK)
                        label.set_fontweight('bold')
                    else:
                        label.set_color(TEXT)

                cbar = ax.collections[0].colorbar
                cbar.ax.yaxis.label.set_color(TEXT)
                cbar.ax.tick_params(colors=TEXT)

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No genes found to display in the Heatmap.")
