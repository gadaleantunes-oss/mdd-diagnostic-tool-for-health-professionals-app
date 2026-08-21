import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier

# A assinatura genômica validada pela seleção de características do Random Forest
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

# --- PASSO 1: Finalizar o Pipeline de Machine Learning ---
# Simulando a matriz após a transposição (Pacientes nas linhas, Genes nas colunas)
# No seu código real, você utilizará o df_ml_ready e as labels (y) verdadeiras
X_train_mock = np.random.normal(8.0, 2.0, (20, 50)) 
y_train_mock = np.array([0]*10 + [1]*10) # 10 Controles (0), 10 MDD (1)
df_train = pd.DataFrame(X_train_mock, columns=top_50_genes)

# Inicializa e treina o classificador
modelo_xgb = XGBClassifier(random_state=42, eval_metric="logloss")
modelo_xgb.fit(df_train, y_train_mock)
print("Passo 1 concluído: XGBoost treinado com a assinatura de 50 biomarcadores.")

# --- PASSO 2: Exportar o Modelo ---
joblib.dump(modelo_xgb, 'motor_diagnostico_mdd.joblib')
print("Passo 2 concluído: Modelo exportado com sucesso como 'motor_diagnostico_mdd.joblib'.")

# Passo 5: Configuração do layout básico e contexto clínico
st.set_page_config(page_title="MDD-CDSS", page_icon="🧬", layout="wide")

st.title("🔬 MDD-CDSS: Clinical Decision Support System")
st.markdown("### Plataforma Translacional de Análise de Exossomos Plasmáticos para Transtorno Depressivo Maior (TDM)")
st.divider()

# Passo 6: Implementar a entrada de dados (Upload do Exame)
st.sidebar.header("Painel Clínico")
st.sidebar.info("Faça o upload do exame transcriptômico do paciente para iniciar a triagem biomolecular.")

# Instancia o componente de upload aceitando estritamente planilhas clínicas
arquivo_exame = st.sidebar.file_uploader("Upload da Matriz de Expressão (.csv)", type=["csv"])

if arquivo_exame is not None:
    # Lendo o exame carregado dinamicamente para um DataFrame do Pandas
    # O index_col=0 garante que a primeira coluna (nome do paciente) seja o índice
    df_paciente = pd.read_csv(arquivo_exame, index_col=0)
    
    st.success("Sequenciamento recebido. Inspecionando a integridade dos dados...")
    st.write("**Pré-visualização da Matriz de Expressão (Valores em Log2):**")
    st.dataframe(df_paciente.head())
    
    # As Fases 3 e 4 (Carregamento do modelo e Predição) entrarão exatamente aqui!
    
else:
    st.warning("Aguardando inserção de dados ômicos pelo corpo clínico...")

# PASSO 7: Carregando o "cérebro" preditivo. 
# Fazemos isso fora do loop de upload para que a memória não recarregue o modelo a cada clique.

@st.cache_resource
def carregar_modelo():
    return joblib.load('motor_diagnostico_mdd.joblib')

modelo_clinico = carregar_modelo()

if arquivo_exame is not None:
    arquivo_exame.seek(0)
    
    if arquivo_exame.size > 0:
        df_paciente = pd.read_csv(arquivo_exame, index_col=0)
        
        if df_paciente.shape[1] == 50:
            st.success("Sequenciamento validado: Assinatura de 50 biomarcadores exossomais íntegra.")
            
            df_paciente = df_paciente.fillna(0.0)
            
            # --- TRAVA DE BLINAGEM DE COLUNAS ---
            # Garante que as colunas sigam rigorosamente a mesma ordem do treinamento
            colunas_oficiais = modelo_clinico.feature_names_in_
            df_paciente = df_paciente[colunas_oficiais]
            # ------------------------------------
            
            probabilidades = modelo_clinico.predict_proba(df_paciente)
            prob_tdm = probabilidades[0][1]
            
            st.subheader("🚨 Resultado da Triagem Biomolecular")
            if prob_tdm > 0.5:
                st.error(f"Alto Risco Predito de TDM: {prob_tdm * 100:.1f}%")
                st.markdown("*O padrão transcricional indica regulação sistêmica patológica compatível com depressão maior.*")
            else:
                st.success(f"Baixo Risco Predito de TDM: {prob_tdm * 100:.1f}%")
                st.markdown("*O padrão transcricional encontra-se dentro dos limites da linha de base de coortes controle.*")
            
            # --- FASE 4: EXPLICABILIDADE VISUAL (DATA VIZ) ---
            st.markdown("---")
            st.subheader("🧬 Assinatura Molecular do Paciente (Top 5 Biomarcadores)")
            st.markdown("*Níveis de expressão em escala Log2 dos transcritos de maior peso sistêmico:*")
            
            # Extrai os primeiros 5 genes da linha do paciente
            top_5_genes = df_paciente.iloc[0, :5]
            
            # Constrói o gráfico com Seaborn e Matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(
                x=top_5_genes.values, 
                y=top_5_genes.index, 
                palette="viridis", 
                ax=ax
            )
            ax.set_xlabel("Nível de Expressão (Log2)")
            ax.set_ylabel("Biomarcador Exossomal")
            ax.set_title("Desvio Transcricional Individual")
            
            # Renderiza no Streamlit
            st.pyplot(fig)
            
        else:
            st.error(f"Erro Crítico: Assinatura biomolecular inválida. O exame contém {df_paciente.shape[1]} genes, mas o motor preditivo exige exatos 50 biomarcadores.")
    else:
        st.error("Erro Crítico: O arquivo enviado está vazio (0 bytes). Por favor, selecione um exame válido.")
else:
    st.warning("Aguardando inserção de dados ômicos pelo corpo clínico...")
