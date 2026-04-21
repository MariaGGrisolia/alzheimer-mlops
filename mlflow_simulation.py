"""
mlflow_simulation.py
──────────────────────────────────────────────────────────────────────────────
Rastreamento de experimentos com MLflow — Previsão de Alzheimer
Dataset: OASIS Longitudinal (oasis_longitudinal.csv)

O que este script faz:
  1. Carrega e prepara os dados clínicos do OASIS
  2. Treina 3 modelos com ajuste de hiperparâmetros
  3. Registra TUDO no MLflow: parâmetros, métricas e o modelo salvo
  4. Identifica e salva o modelo campeão

Como executar:
  python mlflow_simulation.py

Para abrir a interface visual do MLflow depois:
  mlflow ui
  → Acesse http://127.0.0.1:5000 no navegador
──────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "oasis_longitudinal.csv"   # dataset na raiz do projeto
MODEL_DIR   = "models"                   # pasta onde o campeão será salvo
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# Nome do experimento que aparecerá na interface do MLflow
MLFLOW_EXPERIMENT = "Alzheimer_Previsao_OASIS"


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ──────────────────────────────────────────────────────────────────────────────

def carregar_dados(caminho: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Lê o CSV, faz limpeza básica e retorna X (features) e y (alvo binário).

    Decisões de engenharia:
    - 'Converted' é tratado como Demented (0.5 CDR → já há comprometimento)
    - Colunas de ID e Hand são descartadas (sem valor preditivo)
    - M/F é convertida para 0/1
    - Valores ausentes ficam para o SimpleImputer no pipeline
    """
    df = pd.read_csv(caminho)

    # ── Target: Group → binário (1 = Demented / Converted, 0 = Nondemented)
    df = df[df["Group"] != "Converted"].copy()   # exclui 'Converted' para binário limpo
    df["alvo"] = (df["Group"] == "Demented").astype(int)

    # ── Sexo: M → 0, F → 1
    df["Sexo"] = (df["M/F"] == "F").astype(int)

    # ── Colunas que entram como features
    features = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF",
                "Visit", "MR Delay", "Sexo"]

    X = df[features].copy()
    y = df["alvo"].copy()

    print(f"  Dataset: {X.shape[0]} amostras | {X.shape[1]} features")
    print(f"  Distribuição do alvo: {y.value_counts().to_dict()}")
    print(f"  Valores ausentes: {X.isnull().sum().sum()} células\n")

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUÇÃO DO PIPELINE BASE
# ──────────────────────────────────────────────────────────────────────────────

def criar_pipeline(classificador) -> Pipeline:
    """
    Retorna um pipeline scikit-learn com:
      - Imputação pela mediana (robusta a outliers)
      - Normalização com StandardScaler
      - Classificador passado como parâmetro
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     classificador),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 3. FUNÇÃO DE AVALIAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def avaliar_modelo(pipeline, X_test, y_test) -> dict:
    """
    Calcula métricas de classificação e retorna como dicionário.
    ROC-AUC usa probabilidades quando disponível.
    """
    y_pred = pipeline.predict(X_test)

    # ROC-AUC com probabilidade (melhor estimativa) ou decision_function
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_score), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENTOS
# ──────────────────────────────────────────────────────────────────────────────

def experimento_perceptron(X_train, X_test, y_train, y_test):
    """
    Experimento 1 — Perceptron (baseline linear)
    Sem ajuste de hiperparâmetros; serve como referência mínima.
    """
    print("── Experimento 1: Perceptron (baseline) ──")

    params = {
        "max_iter":   1000,
        "random_state": RANDOM_SEED,
        "tol":        1e-3,
    }

    with mlflow.start_run(run_name="Perceptron_baseline"):

        # Treino
        clf = Perceptron(**params)
        pipeline = criar_pipeline(clf)
        pipeline.fit(X_train, y_train)

        # Cross-validation no treino
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")

        # Avaliação no teste
        metricas = avaliar_modelo(pipeline, X_test, y_test)

        # ── Registrar no MLflow
        mlflow.log_params(params)
        mlflow.log_params({"modelo": "Perceptron", "cv_folds": 5})
        mlflow.log_metrics(metricas)
        mlflow.log_metric("cv_f1_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_f1_std",  round(cv_scores.std(),  4))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    print(f"  Métricas: {metricas}")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    return run_id, metricas, pipeline


def experimento_arvore_decisao(X_train, X_test, y_train, y_test):
    """
    Experimento 2 — Árvore de Decisão com GridSearchCV
    Ajuste controlado para evitar overfitting (max_depth limitado).
    """
    print("── Experimento 2: Árvore de Decisão (GridSearchCV) ──")

    param_grid = {
        "clf__max_depth":        [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion":         ["gini", "entropy"],
    }

    pipeline_base = criar_pipeline(
        DecisionTreeClassifier(random_state=RANDOM_SEED)
    )

    grid_search = GridSearchCV(
        pipeline_base,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params   = grid_search.best_params_
    metricas      = avaliar_modelo(best_pipeline, X_test, y_test)

    with mlflow.start_run(run_name="DecisionTree_GridSearch"):
        mlflow.log_params({"modelo": "DecisionTree", "cv_folds": 5, "busca": "GridSearchCV"})
        mlflow.log_params({k.replace("clf__", ""): v for k, v in best_params.items()})
        mlflow.log_metrics(metricas)
        mlflow.log_metric("cv_best_f1", round(grid_search.best_score_, 4))
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    print(f"  Melhores params: {best_params}")
    print(f"  Métricas:        {metricas}\n")
    return run_id, metricas, best_pipeline


def experimento_random_forest(X_train, X_test, y_train, y_test):
    """
    Experimento 3 — Random Forest com RandomizedSearchCV
    Modelo ensemble; mais robusto e com menor risco de overfitting.
    """
    print("── Experimento 3: Random Forest (RandomizedSearchCV) ──")

    param_dist = {
        "clf__n_estimators":      [50, 100, 200, 300],
        "clf__max_depth":         [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features":      ["sqrt", "log2"],
        "clf__bootstrap":         [True, False],
    }

    pipeline_base = criar_pipeline(
        RandomForestClassifier(random_state=RANDOM_SEED)
    )

    rand_search = RandomizedSearchCV(
        pipeline_base,
        param_dist,
        n_iter=30,
        cv=5,
        scoring="f1",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    rand_search.fit(X_train, y_train)

    best_pipeline = rand_search.best_estimator_
    best_params   = rand_search.best_params_
    metricas      = avaliar_modelo(best_pipeline, X_test, y_test)

    with mlflow.start_run(run_name="RandomForest_RandomizedSearch"):
        mlflow.log_params({"modelo": "RandomForest", "cv_folds": 5, "busca": "RandomizedSearchCV", "n_iter": 30})
        mlflow.log_params({k.replace("clf__", ""): v for k, v in best_params.items()})
        mlflow.log_metrics(metricas)
        mlflow.log_metric("cv_best_f1", round(rand_search.best_score_, 4))
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    print(f"  Melhores params: {best_params}")
    print(f"  Métricas:        {metricas}\n")
    return run_id, metricas, best_pipeline


# ──────────────────────────────────────────────────────────────────────────────
# 5. SELEÇÃO DO CAMPEÃO E PERSISTÊNCIA
# ──────────────────────────────────────────────────────────────────────────────

def selecionar_campeao(resultados: list[dict]) -> dict:
    """
    Escolhe o modelo com maior F1-score no conjunto de teste.
    F1 é a métrica primária por equilibrar precisão e recall —
    ambos importam no contexto clínico.
    """
    return max(resultados, key=lambda r: r["metricas"]["f1"])


def salvar_campeao(pipeline, run_id: str, modelo_nome: str):
    """
    Persiste o pipeline campeão como arquivo .pkl e salva o run_id.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    caminho_model  = os.path.join(MODEL_DIR, "model.pkl")
    caminho_run_id = os.path.join(MODEL_DIR, "champion_run_id.txt")

    joblib.dump(pipeline, caminho_model)

    with open(caminho_run_id, "w") as f:
        f.write(run_id)

    print(f"  Modelo salvo em:    {caminho_model}")
    print(f"  Run ID salvo em:    {caminho_run_id}")
    print(f"  Run ID:             {run_id}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. ORQUESTRAÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MLflow — Rastreamento de Experimentos: Previsão de Alzheimer")
    print("=" * 65)

    # ── Dados
    print("\n[1/5] Carregando e preparando dados...")
    X, y = carregar_dados(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    # ── Configurar experimento no MLflow
    print("\n[2/5] Configurando MLflow...")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"  Experimento: '{MLFLOW_EXPERIMENT}'")

    # ── Rodar os 3 experimentos
    print("\n[3/5] Rodando experimentos...\n")
    resultados = []

    run_id_1, met_1, pip_1 = experimento_perceptron(X_train, X_test, y_train, y_test)
    resultados.append({"nome": "Perceptron",    "run_id": run_id_1, "metricas": met_1, "pipeline": pip_1})

    run_id_2, met_2, pip_2 = experimento_arvore_decisao(X_train, X_test, y_train, y_test)
    resultados.append({"nome": "DecisionTree",  "run_id": run_id_2, "metricas": met_2, "pipeline": pip_2})

    run_id_3, met_3, pip_3 = experimento_random_forest(X_train, X_test, y_train, y_test)
    resultados.append({"nome": "RandomForest",  "run_id": run_id_3, "metricas": met_3, "pipeline": pip_3})

    # ── Comparativo
    print("\n[4/5] Comparativo de experimentos:")
    print(f"  {'Modelo':<20} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8} {'ROC-AUC':>9}")
    print("  " + "-" * 68)
    for r in resultados:
        m = r["metricas"]
        print(f"  {r['nome']:<20} {m['accuracy']:>9.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['roc_auc']:>9.4f}")

    # ── Salvar campeão
    print("\n[5/5] Selecionando e salvando modelo campeão...")
    campeao = selecionar_campeao(resultados)
    print(f"\n  ★ Modelo campeão: {campeao['nome']} (F1 = {campeao['metricas']['f1']:.4f})")
    salvar_campeao(campeao["pipeline"], campeao["run_id"], campeao["nome"])

    print("\n" + "=" * 65)
    print("  Concluído! Para visualizar os experimentos:")
    print("  → Execute: mlflow ui")
    print("  → Acesse:  http://127.0.0.1:5000")
    print("=" * 65)


if __name__ == "__main__":
    main()
