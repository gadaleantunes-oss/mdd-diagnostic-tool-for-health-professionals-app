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

st.title("🔬 MDD-CDSS: Clinical Decision Support System")
st.markdown("### Translational Platform for Plasma Exosome Analysis in Major Depressive Disorder (MDD)")
st.divider()

# Step 6: Implement data input (Exam Upload)
st.sidebar.header("Clinical Panel")

# Instantiate the upload component, strictly accepting clinical spreadsheets
exam_file = st.sidebar.file_uploader("Upload Expression Matrix (.csv)", type=["csv"])

if exam_file is not None:
    # Reading the dynamically loaded exam into a Pandas DataFrame
    # index_col=0 ensures the first column (patient name) is the index
    df_patient = pd.read_csv(exam_file, index_col=0)
    
    st.success("Sequencing received. Inspecting data integrity...")
    st.write("**Expression Matrix Preview (Log2 Values):**")
    st.dataframe(df_patient.head())


# STEP 7: Loading the predictive "brain". 
# Done outside the upload loop so memory doesn't reload the model on every click.
@st.cache_resource
def load_model():
    return joblib.load('mdd_diagnostic_engine.joblib')

clinical_model = load_model()

if exam_file is not None:
    exam_file.seek(0)
    
    if exam_file.size > 0:
        df_patient = pd.read_csv(exam_file, index_col=0)
        
        if df_patient.shape[1] == 50:
            st.success("Sequencing validated: 50 exosomal biomarker signature is intact.")
            
            df_patient = df_patient.fillna(0.0)
            
            # --- COLUMN SHIELDING LOCK ---
            # Ensures columns strictly follow the same order as training
            official_columns = clinical_model.feature_names_in_
            df_patient = df_patient[official_columns]
            
            probabilities = clinical_model.predict_proba(df_patient)
            prob_mdd = probabilities[0][1]
            
            st.subheader("🚨 Biomolecular Screening Result")
            if prob_mdd > 0.5:
                st.error(f"High Predicted Risk for MDD: {prob_mdd * 100:.1f}%")
                st.markdown("*The transcriptional pattern indicates pathological systemic regulation compatible with major depression.*")
            else:
                st.success(f"Low Predicted Risk for MDD: {prob_mdd * 100:.1f}%")
                st.markdown("*The transcriptional pattern is within the baseline limits of control cohorts.*")
            
            # --- PHASE 4: VISUAL EXPLAINABILITY (DATA VIZ) ---
            st.markdown("---")
            st.subheader("🧬 Patient's Molecular Signature (Top 5 Biomarkers)")
            st.markdown("*Expression levels in Log2 scale of the transcripts with the highest systemic weight:*")
            
            # Extract the first 5 genes of the patient's row
            top_5_genes = df_patient.iloc[0, :5]
            
            # Build the chart with Seaborn and Matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                x=top_5_genes.values, 
                y=top_5_genes.index, 
                palette="viridis", 
                ax=ax
            )
            ax.set_xlabel("Expression Level (Log2)")
            ax.set_ylabel("Exosomal Biomarker")
            ax.set_title("Individual Transcriptional Deviation")
            
            # Render in Streamlit
            st.pyplot(fig)
            
        else:
            st.error(f"Critical Error: Invalid biomolecular signature. The exam contains {df_patient.shape[1]} genes, but the predictive engine requires exactly 50 biomarkers.")
    else:
        st.error("Critical Error: The uploaded file is empty (0 bytes). Please select a valid exam.")
else:
    st.warning("Waiting for omics data insertion by the clinical staff...")
