import streamlit as st
from src.predict import predict

# Configuração da página
st.set_page_config(
    page_title="Previsão de Alzheimer",
    page_icon="🧠",
    layout="centered"
)

# Sidebar
st.sidebar.title("Sobre")
st.sidebar.info(
    "Modelo de Machine Learning para apoio à detecção precoce de Alzheimer."
)

# Título
st.title("🧠 Previsão de Alzheimer")
st.markdown("### Sistema de apoio à decisão clínica")

st.info("Preencha os dados e clique em prever para avaliar o risco.")

st.divider()

# Formulário
with st.form("form_paciente"):

    st.subheader("Dados do paciente")

    idade = st.slider("Idade", 30, 100, 65)
    st.caption("Risco: <60 baixo | 60–75 moderado | >75 elevado")

    educacao = st.slider("Anos de educação", 0, 20, 10)
    st.caption("Maior escolaridade → maior reserva cognitiva")

    mmse = st.slider("Pontuação MMSE", 0, 30, 20)
    st.caption("MMSE: 24–30 normal | 18–23 leve | <18 severo")

    volume_cerebral = st.number_input(
        "Volume cerebral normalizado", 0.5, 1.0, 0.7
    )
    st.caption("Valores típicos: 0.70–0.85")

    volume_craniano = st.number_input(
        "Volume craniano total", 1000, 2500, 1500
    )

    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    sexo = 0 if sexo == "Masculino" else 1

    submit = st.form_submit_button("🔍 Prever")

# Resultado
if submit:

    data = {
        "Idade": idade,
        "Anos_Educacao": educacao,
        "Pontuacao_MMSE": mmse,
        "Volume_Cerebral_Normalizado": volume_cerebral,
        "Volume_Craniano_Total": volume_craniano,
        "Sexo": sexo,
        "Fator_Escala_Atlas": 0.0,
        "Status_Socioeconomico": 2,
        "Visita": 1,
        "Atraso_RM": 0
    }

    resultado, prob = predict(data)

    st.divider()
    st.subheader("Resultado")

    # Probabilidade
    st.write(f"### Probabilidade de Alzheimer: {prob*100:.1f}%")
    st.progress(float(prob))

    # Classificação de risco
    if prob < 0.3:
        st.success("🟢 Baixo risco")
        interpretacao = (
            "O modelo indica baixa probabilidade de Alzheimer. "
            "Os indicadores cognitivos e clínicos estão dentro de padrões esperados. "
            "Recomenda-se manter acompanhamento preventivo."
        )

    elif prob < 0.7:
        st.warning("🟡 Risco moderado")
        interpretacao = (
            "O modelo indica risco moderado. Alguns sinais podem sugerir "
            "comprometimento cognitivo leve. Recomenda-se avaliação clínica "
            "mais detalhada e monitoramento contínuo."
        )

    else:
        st.error("🔴 Alto risco")
        interpretacao = (
            "O modelo indica alta probabilidade de Alzheimer. "
            "Os padrões observados são compatíveis com comprometimento significativo. "
            "Recomenda-se avaliação médica especializada com urgência."
        )

    # Resultado binário
    if resultado == 1:
        st.error("⚠️ Classificação do modelo: provável Alzheimer")
    else:
        st.success("✅ Classificação do modelo: sem Alzheimer")

    # Interpretação inteligente
    st.markdown("### Interpretação")
    st.write(interpretacao)

    # Observação final
    st.info("ℹ️ Este sistema não substitui avaliação médica profissional.")