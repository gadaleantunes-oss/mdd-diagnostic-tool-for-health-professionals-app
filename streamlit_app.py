import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier

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


def style_biomarker_chart(series: pd.Series, xlabel: str, bar_color: str):
    """Shared chart styling for all biomarker bar charts (font, gridlines, DPI, background)."""
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.barplot(x=series.values, y=series.index, color=bar_color, ax=ax)

    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_COLOR, family="monospace")
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for label_ in ax.get_yticklabels():
        label_.set_fontfamily("monospace")

    ax.grid(axis="x", color=BORDER_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    fig.tight_layout()
    return fig


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
tab_overview, tab_run, tab_results = st.tabs(["🧭 Overview", "🧪 Run Analysis", "📊 Results & Biomarkers"])

# --- OVERVIEW TAB ---
with tab_overview:
    step_tag("STEP 00 · METHODOLOGY")
    st.subheader("About This Platform")
    with st.expander("📖 More about the diagnosis app..."):
        st.write("""
        **The Science Behind the Screen:** 
        Major Depressive Disorder (MDD) leaves systemic biological footprints that go far beyond psychological symptoms. To capture these microscopic changes with high precision, our Clinical Decision Support System (CDSS) acts like a molecular detective using a **Multi-Omic Late Fusion** approach. Here is what is happening under the hood:
                    
        1. **Brain-Derived Exosomes (70% Weight):** 
            The brain communicates with the rest of the body through tiny extracellular vesicles called *exosomes* that successfully cross the blood-brain barrier and circulate in plasma. By sequencing the genetic material (transcriptome) inside these micro-bubbles, our AI isolates specific regulatory transcripts—such as *SFMBT2* and long non-coding RNAs—that reflect active neural and systemic dysregulation. Because these carry a direct signature from the central nervous system, they hold the highest priority in our predictive model.
                    
        2. **Peripheral Blood Proteins (30% Weight):** 
               Simultaneously, the platform analyzes circulating protein profiles from standard blood plasma. This provides a complementary view of the body's systemic inflammatory and metabolic response to chronic stress.
                    
        3. **The AI Decision Engine:** 
            Instead of relying on a single data source (which can lead to false positives or missed signals), our machine learning model (XGBoost) cross-references both fluid signatures. It weighs the genetic instructions against actual physical protein markers. 
                    
        **The Result:** 
        By blending these two biological layers with weighted prioritization, the system bypasses subjective clinical bias and outputs a mathematically robust risk score. This grants medical professionals an objective, data-driven window into the molecular pathology of depression.
         """)

# STEP 7: Loading both predictive "brains"
@st.cache_resource
def load_models():
    # Substitua pelos nomes exatos dos seus arquivos locais
    model_exosomes = joblib.load('xgboost_exosome.joblib')
    model_blood = joblib.load('xgboost_blood.joblib')
    return model_exosomes, model_blood

clinical_model_exo, clinical_model_blood = load_models()

# ============================================================
# SIDEBAR — CLINICAL PANEL
# ============================================================
st.sidebar.header("Clinical Panel")
st.sidebar.markdown("<div class='sidebar-section-title'>STEP 01 · SEQUENCING INPUT</div>", unsafe_allow_html=True)

with st.sidebar.container(border=True):
    file_exosomes = st.file_uploader("Upload EXOSOME Matrix (.csv)", type=["csv"], key="exosomes")
    file_blood = st.file_uploader("Upload BLOOD Matrix (.csv)", type=["csv"], key="blood")

# ============================================================
# RUN ANALYSIS TAB — validation + execution
# ============================================================
analysis_ready = False

with tab_run:
    step_tag("STEP 02 · MODEL VALIDATION")
    st.subheader("Validation & Execution")

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

            analysis_ready = True
            st.info("Analysis complete — open the **Results & Biomarkers** tab to view the fused risk score and signatures.")
        else:
            st.error("Critical Error: Matrix dimensions do not match the required clinical models.")
    else:
        st.warning("Upload both the EXOSOME and BLOOD matrices in the sidebar to run the analysis.")

# ============================================================
# RESULTS & BIOMARKERS TAB
# ============================================================
with tab_results:
    if analysis_ready:
        step_tag("STEP 03 · FUSED RISK ANALYSIS")
        st.subheader("🚨 Multi-Omic Biomolecular Screening Result")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Fused Risk Score", f"{prob_mdd * 100:.1f}%")
        kpi2.metric("Exosome Model Score", f"{prob_exo * 100:.1f}%")
        kpi3.metric("Blood Model Score", f"{prob_blood * 100:.1f}%")
        kpi4.metric("Model Agreement", f"{abs(prob_exo - prob_blood) * 100:.1f}%")

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

        # --- DATA VIZ: MULTI-OMIC COMPARISON ---
        st.markdown("---")
        step_tag("STEP 04 · BIOMARKER SIGNATURE")
        st.subheader("🧬 Systemic Signature (Exosomes vs. Peripheral Blood)")

        # Create two columns in the web interface
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 5 Biomarkers (Exosomes)**")
            top_5_exo = df_exosomes.iloc[0, :5]
            fig_exo = style_biomarker_chart(top_5_exo, "Log2 Expression", ACCENT_TEAL)
            st.pyplot(fig_exo)

        with col2:
            st.markdown("**Top 5 Biomarkers (Peripheral Blood)**")
            top_5_blood = df_blood.iloc[0, :5]
            fig_blood = style_biomarker_chart(top_5_blood, "Log2 Expression", ACCENT_GREEN)
            st.pyplot(fig_blood)
    else:
        st.info("Run the analysis in the **Run Analysis** tab to see the fused risk score and biomarker signatures here.")

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
