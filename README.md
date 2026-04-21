<div align="center">
  <h1>Projeto de Disciplina: Operacionalização de Modelos com MLOps</h1>
</div>

<div align="center">

  **Pós-Graduação em Machine Learning, Deep Learning e Inteligência Artificial**<br>
  **Disciplina:** Operacionalização de Modelos com MLOps<br>
  **Professor:** Ícaro Augusto Maccari Zelioli<br>
  **Aluna:** Maria Glatthardt Grisolia <a href="https://github.com/MariaGGrisolia"><img src="https://img.shields.io/badge/GitHub-perfil-black?logo=github" alt="GitHub"></a>

  <p>
    <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-1.6%2B-orange?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?style=flat-square" alt="MLflow">
    <img src="https://img.shields.io/badge/Streamlit-inferência-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  </p>
</div>

---

## Visão Geral do Projeto

Este repositório representa o **PD2** da disciplina, estruturando um sistema completo de Machine Learning para **previsão de risco de Alzheimer**, com rastreamento de experimentos via `MLflow` e interface de inferência via `Streamlit`.

O dataset utilizado é o **OASIS Longitudinal** (Open Access Series of Imaging Studies), contendo dados clínicos e cognitivos de pacientes.

O foco desta etapa está em demonstrar:
- estrutura de projeto adequada à prática de engenharia de ML
- pipeline de dados e features com decisões explícitas
- experimentos reprodutíveis e rastreáveis com MLflow
- operacionalização com inferência via Streamlit

---

## Estrutura do Repositório

```text
alzheimer-mlops/
├── app/
│   └── app.py                      # Interface Streamlit de inferência
├── data/
│   └── oasis_longitudinal.csv      # Dataset OASIS Longitudinal
├── models/
│   ├── model.pkl                   # Modelo campeão serializado
│   └── champion_run_id.txt         # run_id do modelo campeão no MLflow
├── src/
│   └── predict.py                  # Módulo de inferência
├── mlflow_simulation.py            # Orquestração dos 3 experimentos MLflow
├── requirements.txt
└── README.md
```

---

## Resultados Principais

| Experimento | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Perceptron (baseline) | 0.8382 | 0.8254 | 0.7879 | 0.8667 | 0.9061 |
| Árvore de Decisão (GridSearchCV) | 0.7941 | 0.7742 | 0.7500 | 0.8000 | 0.8241 |
| **Random Forest (RandomizedSearchCV)** ⭐ | **0.8824** | **0.8621** | **0.8929** | **0.8333** | **0.9456** |

**Modelo campeão:** Random Forest com F1=0.8621 e ROC-AUC=0.9456.

---

## Como Executar o Projeto Localmente

### 1. Pré-requisitos

- Python 3.9+
- Dataset `oasis_longitudinal.csv` na raiz do projeto

### 2. Instalar dependências

```bash
pip3 install -r requirements.txt
```

### 3. Rodar os experimentos com MLflow

```bash
python3 mlflow_simulation.py
```

### 4. Abrir a interface do MLflow

```bash
python3 -m mlflow ui
```

Acesse `http://127.0.0.1:5000`.

### 5. Abrir a interface de inferência

```bash
python3 -m streamlit run app/app.py
```

---

## O Que Foi Entregue

- Repositório modularizado com scripts reutilizáveis
- Três experimentos comparativos rastreados no MLflow
- Modelo campeão persistido e utilizado na inferência
- Interface Streamlit para simulação de operação
- Relatório técnico completo

---

<div align="center">
  <small>Desenvolvido para fins acadêmicos — Abril / 2026</small>
</div>
