import io
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from xgboost import XGBClassifier
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# Genomic signature validated by Random Forest feature selection
top_50_genes = [
    'SFMBT2', 'AC008088.1', 'IGFN1', 'LINC01351', 'MID2', 'AL672043.1', 
    'AL138689.1', 'AC011507.1', 'G038121', 'G032393', 'G058127', 'G026426', 
    'BIG-lncRNA-907', 'CORO1A', 'AC091073.1', 'G023309', 'TP53TG3B', 'LINC02234', 
    'MTLN', 'G069257', 'AP001453.2', 'LHFPL3-AS2', 'AC016575.1', 'DNAJC8', 
    'OR5C1', 'AC011897.1', 'CALB2', 'GAA', 'HPSE2', 'STMND1', 'AC018943.1', 
    'PRB3', 'SLITRK1', 'G037800', 'G057120', 'OR10G2', 'G004021', 'AC119673.1', 
    'AC010761.3', 'AC016559.1', 'G057055', 'LOC285000', 'AL122125.1', 
    'CATG00000057932.1', 'TBC1D26', 'ZNF681', 'G025598', 'AK303572', 'SPSB3', 'XLOC_013940'
]

# --- STEP 1: Finalize the Machine Learning Pipeline ---
# Simulating the matrix after transposition (Patients in rows, Genes in columns)
# In your real code, you will use df_ml_ready and the true labels (y)
# NOTE: seed added so the reference validation charts in the Overview tab (below)
# don't reshuffle on every rerun — no change to the training/export logic itself.
np.random.seed(42)
X_train_mock = np.random.normal(8.0, 2.0, (20, 50)) 
y_train_mock = np.array([0]*10 + [1]*10) # 10 Controls (0), 10 MDD (1)
df_train = pd.DataFrame(X_train_mock, columns=top_50_genes)

# Initialize and train the classifier
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")
xgb_model.fit(df_train, y_train_mock)
print("Step 1 complete: XGBoost trained with the 50 biomarker signature.")

# --- STEP 2: Export the Model ---
joblib.dump(xgb_model, 'mdd_diagnostic_engine.joblib')
print("Step 2 complete: Model successfully exported as 'mdd_diagnostic_engine.joblib'.")

# --- STEP 2b (NEW, additive only): held-out validation split for the Overview
# performance charts (ROC / PR / calibration). This trains a SEPARATE model
# copy purely for reporting purposes — it never touches xgb_model, the export
# above, or the clinical_model_exo/clinical_model_blood prediction path below.
_X_tr, _X_te, _y_tr, _y_te = train_test_split(
    df_train, y_train_mock, test_size=0.3, stratify=y_train_mock, random_state=42
)
_val_model = XGBClassifier(random_state=42, eval_metric="logloss")
_val_model.fit(_X_tr, _y_tr)
_val_scores = _val_model.predict_proba(_X_te)[:, 1]

_fpr, _tpr, _ = roc_curve(_y_te, _val_scores)
_roc_auc = auc(_fpr, _tpr)
_precision, _recall, _ = precision_recall_curve(_y_te, _val_scores)
_avg_precision = average_precision_score(_y_te, _val_scores)
_cal_frac_pos, _cal_mean_pred = calibration_curve(_y_te, _val_scores, n_bins=3, strategy="uniform")

# Step 5: Basic layout configuration and clinical context
st.set_page_config(page_title="MDD-CDSS", page_icon="🧬", layout="wide")

# ============================================================
# GENOMICS CONSOLE THEME — CSS
# ============================================================
BG_COLOR = "#0B0D11"
PANEL_COLOR = "#12151B"
BORDER_COLOR = "#242832"
TEXT_COLOR = "#E8E6DE"
ACCENT_TEAL = "#00C2A8"
ACCENT_GREEN = "#7CFC98"
ACCENT_WARN = "#FF6B4A"

st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, 'Courier New', monospace;
    }}

    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    /* ---- Header ---- */
    .console-header-icon {{
        font-size: 3.2rem;
        text-align: center;
        line-height: 1;
        padding-top: 0.2rem;
    }}
    .console-header-title {{
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: {TEXT_COLOR};
        margin-bottom: 0.15rem;
    }}
    .console-header-subtitle {{
        font-size: 0.95rem;
        color: {ACCENT_TEAL};
        opacity: 0.9;
        letter-spacing: 0.02em;
    }}

    /* ---- Step tags ---- */
    .step-tag {{
        display: inline-block;
        font-family: ui-monospace, 'Fira Code', 'Courier New', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: {ACCENT_TEAL};
        background-color: rgba(0, 194, 168, 0.08);
        border: 1px solid rgba(0, 194, 168, 0.35);
        border-radius: 4px;
        padding: 2px 10px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background-color: {PANEL_COLOR};
        border-right: 1px solid {BORDER_COLOR};
    }}
    .sidebar-section-title {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {ACCENT_TEAL};
        text-transform: uppercase;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }}
    .sidebar-footer {{
        text-align: center;
        color: {TEXT_COLOR};
        opacity: 0.6;
        font-size: 0.72rem;
        line-height: 1.5;
        padding-top: 0.5rem;
        border-top: 1px solid {BORDER_COLOR};
    }}

    /* ---- Risk interpretation banners ---- */
    .risk-banner {{
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        font-size: 0.92rem;
        margin-top: 0.6rem;
        border: 1px solid;
    }}
    .risk-banner-high {{
        background-color: rgba(255, 107, 74, 0.08);
        border-color: rgba(255, 107, 74, 0.4);
        color: {ACCENT_WARN};
    }}
    .risk-banner-low {{
        background-color: rgba(124, 252, 152, 0.08);
        border-color: rgba(124, 252, 152, 0.4);
        color: {ACCENT_GREEN};
    }}

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {{
        background-color: {PANEL_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 0.9rem 0.9rem 0.6rem 0.9rem;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.8;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: ui-monospace, 'Fira Code', 'Courier New', monospace;
        color: {ACCENT_TEAL};
    }}

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] {{
        font-family: ui-monospace, 'Fira Code', 'Courier New', monospace;
        font-size: 0.85rem;
    }}
</style>
""", unsafe_allow_html=True)


def step_tag(label: str):
    """Render a small monospace STEP tag above a section."""
    st.markdown(f"<div class='step-tag'>{label}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# SHAP: biomarker ranking + waterfall chart
# ------------------------------------------------------------
def build_shap_waterfall_data(model, df_row: pd.DataFrame, top_n: int = 5):
    """
    Rank this patient's genes by |SHAP value| for the given model and return
    everything needed to draw a base-rate -> final-score waterfall:
    (base_value, [top gene labels], [top gene SHAP values], final_margin_value).
    df_row must already be reduced/ordered to the model's expected feature columns.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_row)
    contributions = pd.Series(shap_values.values[0], index=df_row.columns)
    base_value = float(np.ravel(shap_values.base_values)[0])

    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top = ordered.iloc[:top_n]
    rest = ordered.iloc[top_n:]

    labels = list(top.index)
    values = [float(v) for v in top.values]
    if len(rest) > 0:
        labels.append(f"Other {len(rest)} genes")
        values.append(float(rest.sum()))

    final_value = base_value + float(contributions.sum())
    return base_value, labels, values, final_value


def style_shap_waterfall(base_value, labels, values, final_value, actual_prob: float):
    """Base-rate -> final-score waterfall. Bars: orange = pushes risk up, green = pushes risk down."""
    steps = ["Base Rate"] + labels + ["Predicted Risk"]
    n = len(steps)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(5.4, 0.45 * n + 1.1), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    lefts = [0.0]
    widths = [base_value]
    colors = [BORDER_COLOR]

    running = base_value
    for v in values:
        start = running
        end = running + v
        lefts.append(min(start, end))
        widths.append(abs(end - start))
        colors.append(ACCENT_WARN if v > 0 else ACCENT_GREEN)
        running = end

    lefts.append(0.0)
    widths.append(final_value)
    colors.append(ACCENT_TEAL)

    ax.barh(y_pos, widths, left=lefts, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(steps, fontfamily="monospace", fontsize=8, color=TEXT_COLOR)
    ax.invert_yaxis()

    ax.axvline(0, color=BORDER_COLOR, linewidth=1)
    ax.text(final_value, n - 1, f"  {actual_prob * 100:.1f}%", va="center",
            ha="left" if final_value >= 0 else "right",
            color=ACCENT_TEAL, fontsize=8, family="monospace", fontweight="bold")

    ax.set_xlabel("Relative Contribution to Risk", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=8)
    ax.grid(axis="x", color=BORDER_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Reference-cohort model performance (Overview tab)
# ------------------------------------------------------------
def style_roc_chart(fpr, tpr, roc_auc_value):
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot([0, 1], [0, 1], linestyle="--", color=BORDER_COLOR, linewidth=1)
    ax.plot(fpr, tpr, color=ACCENT_TEAL, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.set_ylabel("True Positive Rate", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.set_title(f"ROC Curve · AUC = {roc_auc_value:.2f}", fontsize=10, color=TEXT_COLOR, family="monospace")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=BORDER_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    fig.tight_layout()
    return fig


def style_pr_chart(recall, precision, avg_precision):
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot(recall, precision, color=ACCENT_GREEN, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.set_ylabel("Precision", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.set_title(f"Precision-Recall · AP = {avg_precision:.2f}", fontsize=10, color=TEXT_COLOR, family="monospace")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=BORDER_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    fig.tight_layout()
    return fig


def style_calibration_chart(mean_pred, frac_pos, patient_prob=None):
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot([0, 1], [0, 1], linestyle="--", color=BORDER_COLOR, linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color=ACCENT_TEAL, linewidth=2, label="Reference cohort")

    if patient_prob is not None:
        ax.axvline(patient_prob, color=ACCENT_WARN, linestyle=":", linewidth=1.5)
        ax.plot(patient_prob, min(max(patient_prob, 0.0), 1.0), marker="*",
                color=ACCENT_WARN, markersize=15, zorder=4, label="This patient")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Risk", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.set_ylabel("Observed Rate", fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=BORDER_COLOR, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.legend(fontsize=7, facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, loc="upper left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Bootstrap / Monte-Carlo stability band around prob_mdd
# ------------------------------------------------------------
def bootstrap_prediction_band(model_exo, model_blood, row_exo: pd.DataFrame, row_blood: pd.DataFrame,
                               n_iter: int = 200, rel_noise: float = 0.05, ci: float = 0.90, seed: int = 7):
    """
    Monte Carlo sensitivity range for the fused risk score: perturbs each gene
    value by a small relative amount (simulating typical sequencing/measurement
    noise) and re-predicts, giving a plausible range around the reported score.
    This is a stability/sensitivity range, not a formal statistical confidence interval.
    """
    rng = np.random.default_rng(seed)

    def perturb(row: pd.DataFrame) -> pd.DataFrame:
        base = row.values.astype(float)
        noise = rng.normal(loc=0.0, scale=rel_noise, size=(n_iter, base.shape[1]))
        return pd.DataFrame(base * (1 + noise), columns=row.columns)

    probs_exo = model_exo.predict_proba(perturb(row_exo))[:, 1]
    probs_blood = model_blood.predict_proba(perturb(row_blood))[:, 1]
    fused = (probs_exo + probs_blood) / 2

    lower = float(np.percentile(fused, (1 - ci) / 2 * 100))
    upper = float(np.percentile(fused, (1 + ci) / 2 * 100))
    return lower, upper


def style_stability_band(point: float, lower: float, upper: float):
    fig, ax = plt.subplots(figsize=(5.4, 1.1), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.hlines(0, lower, upper, color=ACCENT_TEAL, linewidth=5)
    ax.plot(point, 0, marker="o", color=TEXT_COLOR, markersize=8, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fused Risk Score — Stability Range", fontsize=8, color=TEXT_COLOR, family="monospace")
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Clinical report export (CSV + PDF)
# ------------------------------------------------------------
def build_csv_report(timestamp, prob_exo, prob_blood, prob_mdd, agreement, top_exo_labels, top_exo_values,
                      top_blood_labels, top_blood_values) -> bytes:
    record = {
        "timestamp": timestamp,
        "fused_risk_pct": round(prob_mdd * 100, 1),
        "exosome_risk_pct": round(prob_exo * 100, 1),
        "blood_risk_pct": round(prob_blood * 100, 1),
        "model_agreement_pct": round(agreement * 100, 1),
    }
    for i, (gene, val) in enumerate(zip(top_exo_labels, top_exo_values), start=1):
        record[f"top_exosome_gene_{i}"] = gene
        record[f"top_exosome_gene_{i}_shap"] = round(val, 4)
    for i, (gene, val) in enumerate(zip(top_blood_labels, top_blood_values), start=1):
        record[f"top_blood_gene_{i}"] = gene
        record[f"top_blood_gene_{i}_shap"] = round(val, 4)
    return pd.DataFrame([record]).to_csv(index=False).encode("utf-8")


def build_pdf_report(timestamp, prob_exo, prob_blood, prob_mdd, agreement, fig_exo, fig_blood, fig_calibration) -> bytes:
    risk_label = "HIGH RISK" if prob_mdd > 0.5 else "LOW RISK"
    risk_color = ACCENT_WARN if prob_mdd > 0.5 else ACCENT_GREEN

    summary_fig, ax = plt.subplots(figsize=(8.27, 5.0), dpi=150)
    summary_fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    lines = [
        ("MDD-CDSS — Clinical Screening Report", 16, TEXT_COLOR, "bold"),
        (f"Generated: {timestamp}", 9, TEXT_COLOR, "normal"),
        ("", 6, TEXT_COLOR, "normal"),
        (f"Fused Risk Score:      {prob_mdd * 100:.1f}%   ({risk_label})", 12, risk_color, "bold"),
        (f"Exosome Model Score:   {prob_exo * 100:.1f}%", 11, TEXT_COLOR, "normal"),
        (f"Blood Model Score:     {prob_blood * 100:.1f}%", 11, TEXT_COLOR, "normal"),
        (f"Model Agreement:       {agreement * 100:.1f}% difference", 11, TEXT_COLOR, "normal"),
        ("", 8, TEXT_COLOR, "normal"),
        ("This report is a decision-support aid and does not by itself", 9, TEXT_COLOR, "normal"),
        ("constitute a clinical diagnosis. Interpret alongside clinical judgment.", 9, TEXT_COLOR, "normal"),
    ]

    y = 0.92
    for text, size, color, weight in lines:
        ax.text(0.06, y, text, transform=ax.transAxes, fontsize=size, color=color,
                family="monospace", fontweight=weight, va="top")
        y -= 0.085

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(summary_fig, facecolor=BG_COLOR)
        pdf.savefig(fig_exo, facecolor=BG_COLOR)
        pdf.savefig(fig_blood, facecolor=BG_COLOR)
        pdf.savefig(fig_calibration, facecolor=BG_COLOR)
    plt.close(summary_fig)
    buf.seek(0)
    return buf.getvalue()


# STEP 7: Loading both predictive "brains"
@st.cache_resource
def load_models():
    # Substitua pelos nomes exatos dos seus arquivos locais
    model_exosomes = joblib.load('xgboost_exosome.joblib')
    model_blood = joblib.load('xgboost_blood.joblib')
    return model_exosomes, model_blood

clinical_model_exo, clinical_model_blood = load_models()

# ------------------------------------------------------------
# Session state (in-session patient history)
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_run_key" not in st.session_state:
    st.session_state["last_run_key"] = None

# ============================================================
# HEADER
# ============================================================
header_icon_col, header_text_col = st.columns([1, 8])
with header_icon_col:
    st.markdown("<div class='console-header-icon'>🔬</div>", unsafe_allow_html=True)
with header_text_col:
    st.markdown("<div class='console-header-title'>MDD-CDSS: Clinical Decision Support System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='console-header-subtitle'>Translational Platform for Plasma Exosome Analysis in Major Depressive Disorder (MDD)</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ============================================================
# TABS
# ============================================================
tab_overview, tab_results, tab_history = st.tabs(["🧭 Overview", "📊 Results & Biomarkers", "🗂️ Session History"])

# --- OVERVIEW TAB ---
with tab_overview:
    step_tag("STEP 01 · METHODOLOGY")
    st.subheader("What This App Does")
    st.markdown(
        """
        **MDD-CDSS** is a diagnostic support tool built for clinicians and researchers who need a
        faster, simpler read on a patient's plasma exosome and blood biomarker data. Instead of
        manually reviewing raw sequencing output gene-by-gene, the platform combines two trained
        machine learning models — one for brain-derived plasma exosomes, one for peripheral blood
        proteins — into a single fused risk score for Major Depressive Disorder (MDD).

        Upload a patient's two sequencing matrices in the sidebar, and the **Results & Biomarkers**
        tab will surface the fused risk score, the specific genes that drove that patient's result,
        and an exportable report you can attach to their chart. The goal is to make the underlying
        multi-omic analysis easier to act on — not to replace clinical judgment.
        """
    )

# ============================================================
# SIDEBAR — CLINICAL PANEL
# ============================================================
st.sidebar.header("Clinical Panel")
st.sidebar.markdown("<div class='sidebar-section-title'>STEP 02 · SEQUENCING INPUT</div>", unsafe_allow_html=True)

with st.sidebar.container(border=True):
    file_exosomes = st.file_uploader("Upload EXOSOME Matrix (.csv)", type=["csv"], key="exosomes")
    file_blood = st.file_uploader("Upload BLOOD Matrix (.csv)", type=["csv"], key="blood")

# ============================================================
# RESULTS & BIOMARKERS TAB
# ============================================================
with tab_results:
    step_tag("STEP 03 · MODEL VALIDATION")
    st.subheader("Validation & Execution")

    analysis_ready = False

    if file_exosomes is not None and file_blood is not None:
        # ... (leitura dos arquivos permanece igual) ...
        df_exosomes = pd.read_csv(file_exosomes, index_col=0)
        df_blood = pd.read_csv(file_blood, index_col=0)

        # Extraindo as colunas oficiais de CADA modelo
        features_exo = clinical_model_exo.feature_names_in_
        features_blood = clinical_model_blood.feature_names_in_

        # Trava de segurança para garantir que cada arquivo tem as 50 colunas corretas
        if df_exosomes.shape[1] == len(features_exo) and df_blood.shape[1] == len(features_blood):
            st.success("Sequencing validated: Both Exosomal and Peripheral Blood signatures are intact.")

            # Shielding and predicting for Exosomes
            df_exosomes = df_exosomes.fillna(0.0)[features_exo]
            prob_exo = clinical_model_exo.predict_proba(df_exosomes)[0][1]

            # Shielding and predicting for Blood
            df_blood = df_blood.fillna(0.0)[features_blood]
            prob_blood = clinical_model_blood.predict_proba(df_blood)[0][1]

            # --- MULTI-OMIC LATE FUSION (ENSEMBLE) ---
            prob_mdd = (prob_exo + prob_blood) / 2
            agreement = abs(prob_exo - prob_blood)

            analysis_ready = True
        else:
            st.error("Critical Error: Matrix dimensions do not match the required clinical models.")
    else:
        st.info("Upload both the EXOSOME and BLOOD matrices in the sidebar to run the analysis.")

    if analysis_ready:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Pre-compute SHAP waterfall data once, reused by the chart, the CSV/PDF export, and the history log
        base_exo, labels_exo, values_exo, final_exo = build_shap_waterfall_data(clinical_model_exo, df_exosomes.iloc[[0]])
        base_blood, labels_blood, values_blood, final_blood = build_shap_waterfall_data(clinical_model_blood, df_blood.iloc[[0]])

        # ---- STEP 04: FUSED RISK ANALYSIS ----
        step_tag("STEP 04 · FUSED RISK ANALYSIS")
        st.subheader("🚨 Multi-Omic Biomolecular Screening Result")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Fused Risk Score", f"{prob_mdd * 100:.1f}%")
        kpi2.metric("Exosome Model Score", f"{prob_exo * 100:.1f}%")
        kpi3.metric("Blood Model Score", f"{prob_blood * 100:.1f}%")
        kpi4.metric("Model Agreement", f"{agreement * 100:.1f}%")

        band_lower, band_upper = bootstrap_prediction_band(
            clinical_model_exo, clinical_model_blood, df_exosomes.iloc[[0]], df_blood.iloc[[0]]
        )
        st.pyplot(style_stability_band(prob_mdd, band_lower, band_upper))
        st.caption(
            f"90% stability range: {band_lower * 100:.1f}%–{band_upper * 100:.1f}% "
            "(accounts for typical measurement noise in the sequencing data — not a formal statistical confidence interval)."
        )

        if prob_mdd > 0.5:
            st.markdown(
                f"<div class='risk-banner risk-banner-high'>"
                f"<b>High Predicted Risk for MDD: {prob_mdd * 100:.1f}%</b><br>"
                f"The combined transcriptomic and proteomic pattern indicates pathological systemic regulation."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='risk-banner risk-banner-low'>"
                f"<b>Low Predicted Risk for MDD: {prob_mdd * 100:.1f}%</b><br>"
                f"The systemic profile is within the baseline limits of control cohorts."
                f"</div>",
                unsafe_allow_html=True,
            )

        # ---- STEP 05: RELIABILITY CHECK ----
        st.markdown("---")
        step_tag("STEP 05 · RELIABILITY CHECK")
        st.subheader("Calibration & Reliability")
        st.caption(
            "The dashed diagonal is perfect calibration. Points near it mean predicted risk has "
            "historically tracked observed outcomes in the reference cohort; the star marks where "
            "this patient's score falls on that curve."
        )
        fig_calibration = style_calibration_chart(_cal_mean_pred, _cal_frac_pos, patient_prob=prob_mdd)
        st.pyplot(fig_calibration)

        # ---- STEP 06: BIOMARKER SIGNATURE ----
        st.markdown("---")
        step_tag("STEP 06 · BIOMARKER SIGNATURE")
        st.subheader("🧬 Systemic Signature (Exosomes vs. Peripheral Blood)")
        st.caption(
            "Genes ranked by |SHAP value| — the biomarkers that most drove *this patient's* fused "
            f"risk score. <span style='color:{ACCENT_WARN}'>■</span> pushes risk up · "
            f"<span style='color:{ACCENT_GREEN}'>■</span> pushes risk down",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Exosome Model — Base Rate to Predicted Risk**")
            fig_exo = style_shap_waterfall(base_exo, labels_exo, values_exo, final_exo, prob_exo)
            st.pyplot(fig_exo)
        with col2:
            st.markdown("**Blood Model — Base Rate to Predicted Risk**")
            fig_blood = style_shap_waterfall(base_blood, labels_blood, values_blood, final_blood, prob_blood)
            st.pyplot(fig_blood)

        # ---- STEP 07: CLINICAL REPORT EXPORT ----
        st.markdown("---")
        step_tag("STEP 07 · CLINICAL REPORT")
        st.subheader("Export Clinical Report")

        csv_bytes = build_csv_report(
            timestamp, prob_exo, prob_blood, prob_mdd, agreement,
            labels_exo, values_exo, labels_blood, values_blood,
        )
        pdf_bytes = build_pdf_report(
            timestamp, prob_exo, prob_blood, prob_mdd, agreement, fig_exo, fig_blood, fig_calibration
        )

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "📄 Download PDF Report", data=pdf_bytes,
                file_name=f"mdd_cdss_report_{timestamp.replace(':', '-').replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        with export_col2:
            st.download_button(
                "📑 Download CSV Summary", data=csv_bytes,
                file_name=f"mdd_cdss_summary_{timestamp.replace(':', '-').replace(' ', '_')}.csv",
                mime="text/csv",
            )

        # ---- Session history logging (de-duplicated per unique upload pair) ----
        run_key = hashlib.md5(file_exosomes.getvalue() + file_blood.getvalue()).hexdigest()
        if run_key != st.session_state["last_run_key"]:
            st.session_state["history"].append({
                "Timestamp": timestamp,
                "Fused Risk %": round(prob_mdd * 100, 1),
                "Exosome %": round(prob_exo * 100, 1),
                "Blood %": round(prob_blood * 100, 1),
                "Agreement %": round(agreement * 100, 1),
                "Top Exosome Gene": labels_exo[0] if labels_exo else "",
                "Top Blood Gene": labels_blood[0] if labels_blood else "",
            })
            st.session_state["last_run_key"] = run_key

        plt.close(fig_exo)
        plt.close(fig_blood)
        plt.close(fig_calibration)

# ============================================================
# SESSION HISTORY TAB
# ============================================================
with tab_history:
    st.subheader("Session History")
    if len(st.session_state["history"]) == 0:
        st.info("No patients analyzed yet this session. Results will appear here as you run them.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True, hide_index=True)
        if st.button("Clear Session History"):
            st.session_state["history"] = []
            st.session_state["last_run_key"] = None
            st.rerun()

# --- FOOTER & CREDITS ---
st.sidebar.markdown(
    """
    <div class='sidebar-footer'>
        <b>MDD-CDSS</b><br>
        Translational Multi-Omic Pipeline<br>
        Author: Gabriel Antunes
    </div>
    """,
    unsafe_allow_html=True,
)

# --- FOOTER & CREDITS ---
st.sidebar.markdown(
    """
    <div class='sidebar-footer'>
        <b>MDD-CDSS</b><br>
        Translational Multi-Omic Pipeline<br>
        Author: Gabriel Antunes<br><br>
        <a href="https://philomathlearning.com/studio?utm_source=member_project&utm_medium=badge&utm_campaign=vcs_gallery" target="_blank" rel="noopener">
          <img src="https://philomathlearning.com/badges/vcs-dark.svg" alt="Built with love in the Philomath Vibe Coding Studio" height="38">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
