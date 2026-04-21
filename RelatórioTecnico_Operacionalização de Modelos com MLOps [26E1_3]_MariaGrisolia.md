# Relatório Técnico
**por Maria Glatthardt Grisolia**
Classe PGLLIA01C0-24-L1 - disciplina Machine Learning com Scikit-Learn e MLOps [26E1-26E1]

---

# Sistema de Apoio à Decisão Clínica para Previsão de Alzheimer com Machine Learning e MLOps
## Da experimentação à operacionalização de modelos preditivos

**Link Github:** https://github.com/MariaGGrisolia/alzheimer-mlops
---

## 1. Contexto do Problema e Objetivo do Projeto

O presente projeto tem como objetivo desenvolver um sistema de apoio à decisão clínica para a previsão de risco de Alzheimer, utilizando técnicas de Machine Learning aplicadas a dados clínicos e cognitivos.

A doença de Alzheimer representa um dos principais desafios na área da saúde, especialmente devido à dificuldade de detecção precoce e à progressão gradual dos sintomas. Nesse contexto, modelos preditivos podem auxiliar profissionais de saúde na identificação de padrões associados ao comprometimento cognitivo, contribuindo para uma avaliação mais informada.

O problema é estruturado como uma tarefa de classificação binária, na qual o modelo deve prever a presença ou ausência de Alzheimer com base em variáveis como idade, nível educacional, pontuação no teste cognitivo (MMSE) e medidas estruturais do cérebro.

Do ponto de vista clínico, diferentes tipos de erro possuem impactos distintos:

- **Falsos negativos** (não identificar pacientes com Alzheimer) podem atrasar o diagnóstico e comprometer intervenções precoces;
- **Falsos positivos** podem gerar preocupação desnecessária e custos adicionais com exames.

Dessa forma, a escolha do modelo não deve considerar apenas métricas tradicionais como acurácia, mas também o equilíbrio entre precisão e recall, com atenção especial à capacidade de identificar corretamente casos positivos.

### 1.1 Transição de Abordagem: Experimentação para Entrega

Inicialmente, o desenvolvimento foi conduzido em ambiente exploratório (notebook), com foco na análise de dados e comparação de modelos. No entanto, esse formato apresenta limitações importantes em termos de reprodutibilidade, organização e integração.

Neste projeto, foi realizada a transição para uma abordagem orientada à entrega, estruturando o sistema em componentes separados:

- treinamento do modelo;
- rastreamento de experimentos com MLflow;
- pipeline de inferência;
- interface interativa.

Essa mudança reflete a evolução do papel do profissional, passando de uma abordagem exploratória para uma perspectiva de engenharia de Machine Learning, na qual o foco está na construção de soluções utilizáveis, reprodutíveis e integráveis.

### 1.2 Objetivo Técnico

O objetivo técnico do projeto é construir um pipeline de Machine Learning capaz de:

- processar dados clínicos de forma consistente;
- conduzir experimentos comparativos rastreáveis com MLflow;
- gerar previsões confiáveis sobre o risco de Alzheimer;
- disponibilizar essas previsões por meio de uma interface interativa.

Além disso, busca-se garantir a separação entre as etapas de treinamento e inferência, permitindo maior controle, organização e reprodutibilidade do sistema.

---

## 2. Dados, Features e Pipeline de Processamento

### 2.1 Descrição do Dataset

O dataset utilizado é o **OASIS Longitudinal** (Open Access Series of Imaging Studies), contendo 373 registros de pacientes com múltiplas visitas clínicas. Para a modelagem, foram excluídos os casos classificados como "Converted" (pacientes em transição), resultando em 336 amostras com classificação binária definida.

Entre as principais features utilizadas, destacam-se:

- Idade
- Anos de educação (EDUC)
- Pontuação no MMSE (Mini Exame do Estado Mental)
- Volume cerebral normalizado (nWBV)
- Volume craniano total (eTIV)
- Fator de escala Atlas (ASF)
- Status socioeconômico (SES)
- Sexo
- Número da visita e atraso do exame de RM

A variável alvo foi definida a partir da coluna "Group", sendo transformada em variável binária:

- 0 → Sem Alzheimer (Nondemented): 190 amostras
- 1 → Com Alzheimer (Demented): 146 amostras

### 2.2 Tratamento e Preparação dos Dados

Durante a etapa de pré-processamento, foram realizadas as seguintes transformações:

- Exclusão dos casos "Converted" para garantir classificação binária limpa;
- Conversão da variável sexo (M/F) para formato numérico (0/1);
- Remoção de colunas irrelevantes (identificadores e variáveis redundantes).

O dataset apresentou 21 células com valores ausentes, tratados com imputação pela mediana, estratégia robusta a outliers e adequada para dados numéricos.

### 2.3 Qualidade dos Dados e Limitações

A análise exploratória indicou algumas características importantes do dataset:

- presença de variáveis com diferentes escalas, tratada com StandardScaler;
- possíveis correlações entre features;
- tamanho relativamente reduzido da base (336 amostras após filtragem).

Limitações estruturais identificadas:

- ausência de validação externa;
- possível viés na população amostrada;
- simplificação de variáveis clínicas complexas;
- dados longitudinais com múltiplas visitas por paciente, o que pode introduzir dependência entre amostras.

### 2.4 Pipeline de Pré-processamento

Foi implementado um pipeline utilizando a biblioteca scikit-learn, integrando as etapas de pré-processamento e modelagem. O pipeline inclui:

1. imputação de valores ausentes (mediana);
2. normalização das variáveis (StandardScaler);
3. modelo de classificação.

Essa abordagem garante que todas as transformações aplicadas durante o treinamento sejam replicadas na inferência, evitando inconsistências e eliminando o risco de data leakage. O pipeline completo foi registrado como artefato no MLflow, garantindo rastreabilidade total entre experimentos.

### 2.5 Redução de Dimensionalidade

Técnicas de redução de dimensionalidade, como PCA, não foram aplicadas neste projeto. Essa decisão foi baseada nos seguintes fatores:

- número relativamente baixo de features (10 variáveis);
- necessidade de manter interpretabilidade das variáveis para o contexto clínico;
- baixo ganho esperado em desempenho dado o volume de dados.

Em contexto clínico, a interpretabilidade das variáveis originais é um requisito importante, pois permite que profissionais de saúde compreendam quais indicadores influenciam a previsão.

### 2.6 Impacto no Modelo

A utilização de um pipeline estruturado contribuiu para maior consistência entre treinamento e inferência, redução de erros associados a pré-processamento manual e melhor organização do fluxo de dados.

---

## 3. Modelagem e Experimentação

### 3.1 Planejamento Experimental

A etapa de modelagem foi conduzida de forma comparativa e sistemática, com o objetivo de avaliar diferentes abordagens de classificação para o problema de previsão de Alzheimer. Todos os experimentos foram registrados no MLflow sob o experimento denominado **"Alzheimer_Previsao_OASIS"**.

Foram selecionados três modelos:

- **Perceptron** — modelo linear utilizado como baseline;
- **Árvore de Decisão** — modelo interpretável com ajuste via GridSearchCV;
- **Random Forest** — modelo ensemble com ajuste via RandomizedSearchCV.

A escolha desses modelos permitiu comparar diferentes níveis de complexidade, custo computacional e capacidade de generalização.

### 3.2 Metodologia de Avaliação

Os modelos foram avaliados utilizando:

- **Accuracy** — proporção de previsões corretas
- **F1-score** — equilíbrio entre precisão e recall (métrica primária)
- **Precision** — taxa de acerto entre os casos identificados como positivos
- **Recall** — taxa de identificação correta dos casos positivos
- **ROC-AUC** — capacidade discriminativa geral do modelo

A divisão treino/teste foi de 80%/20% com estratificação, garantindo proporção equivalente das classes em ambos os conjuntos. Validação cruzada com 5 folds foi utilizada durante o ajuste de hiperparâmetros.

### 3.3 Ajuste de Hiperparâmetros

Para otimizar o desempenho dos modelos, foram utilizadas técnicas de busca:

- **GridSearchCV** aplicado à Árvore de Decisão, explorando profundidade, critério de divisão e mínimo de amostras por split;
- **RandomizedSearchCV** aplicado ao Random Forest, com 30 iterações, explorando número de estimadores, profundidade, features e uso de bootstrap.

### 3.4 Resultados Experimentais

Os experimentos foram registrados integralmente no MLflow, incluindo parâmetros, métricas e modelos serializados. Os resultados obtidos no conjunto de teste foram:

| Experimento | Accuracy | F1 | Precision | Recall | ROC-AUC | CV F1 |
|---|---|---|---|---|---|---|
| Perceptron (baseline) | 0.8382 | 0.8254 | 0.7879 | 0.8667 | 0.9061 | 0.7933 ± 0.0424 |
| Árvore de Decisão (GridSearchCV) | 0.7941 | 0.7742 | 0.7500 | 0.8000 | 0.8241 | 0.8190 |
| **Random Forest (RandomizedSearchCV)** ⭐ | **0.8824** | **0.8621** | **0.8929** | **0.8333** | **0.9456** | **0.8426** |

**Parâmetros registrados — Perceptron (baseline):**

- max_iter: 1000
- tol: 0.001
- random_state: 42
- cv_folds: 5

**Melhores hiperparâmetros encontrados — Árvore de Decisão (GridSearchCV):**

- max_depth: 7
- criterion: entropy
- min_samples_split: 5
- cv_folds: 5
- busca: GridSearchCV

**Melhores hiperparâmetros encontrados — Random Forest (RandomizedSearchCV):**

- n_estimators: 200
- max_depth: 10
- max_features: log2
- min_samples_split: 2
- bootstrap: False
- cv_folds: 5
- n_iter: 30

**Leitura de engenharia:** o Random Forest apresentou melhor desempenho em todas as métricas principais — maior F1 (0.8621), maior precision (0.8929) e maior ROC-AUC (0.9456). Em contexto clínico, o ROC-AUC de 0.9456 indica excelente capacidade discriminativa. O Perceptron, embora apresente F1 próximo, possui menor precision, gerando mais falsos positivos.

### 3.5 Escolha do Modelo Final

A escolha do modelo final considerou métricas quantitativas e o contexto do problema. O **Random Forest** foi selecionado como modelo campeão por apresentar:

- melhor F1-score (0.8621) — equilíbrio entre precision e recall;
- maior ROC-AUC (0.9456) — melhor capacidade discriminativa geral;
- maior precision (0.8929) — reduz recusas indevidas e falsos alarmes;
- robustez a variações nos dados por ser um modelo ensemble;
- capacidade de gerar probabilidades, possibilitando análise de risco graduada.

### 3.6 Rastreamento de Experimentos com MLflow

Todos os experimentos foram registrados com MLflow, garantindo rastreabilidade completa. Para cada experimento, foram registrados:

- **Parâmetros:** tipo de modelo, estratégia de busca, hiperparâmetros otimizados, número de folds;
- **Métricas:** accuracy, f1, precision, recall, roc_auc, cv_best_f1;
- **Artefatos:** pipeline serializado completo (pré-processamento + modelo).

O run_id do modelo campeão foi persistido em `models/champion_run_id.txt`, permitindo que a interface de inferência carregue automaticamente o melhor modelo registrado.

---

## 4. Operacionalização do Modelo

### 4.1 Persistência do Modelo

Após a seleção, o modelo final foi persistido de duas formas:

1. **Via MLflow** — o pipeline completo foi registrado como artefato no experimento, com run_id rastreável;
2. **Via joblib** — arquivo `models/model.pkl` para uso direto na interface Streamlit.

A persistência foi realizada sobre o pipeline completo, incluindo etapas de pré-processamento e o modelo treinado, garantindo consistência entre treinamento e inferência.

### 4.2 Separação entre Treinamento e Inferência

O projeto foi estruturado de forma a separar claramente o processo de treinamento (realizado em `mlflow_simulation.py`) do processo de inferência (implementado em `predict.py`). A inferência recebe dados de entrada, estrutura no formato adequado, aplica o pipeline do modelo e retorna a predição e a probabilidade.

### 4.3 Interface Interativa com Streamlit

Para disponibilizar o modelo de forma acessível, foi desenvolvida uma aplicação com Streamlit, permitindo:

- inserção de variáveis clínicas relevantes via formulário interativo;
- execução da predição em tempo real;
- visualização do resultado com probabilidade percentual, barra de progresso e classificação de risco (baixo/moderado/alto);
- interpretação textual do resultado.

### 4.4 Apresentação dos Resultados

A aplicação apresenta os resultados em múltiplos níveis:

- Predição binária (presença ou ausência de Alzheimer);
- Probabilidade associada (%);
- Classificação de risco (baixo, moderado ou alto);
- Interpretação textual do resultado.

Essa abordagem reduz o efeito de "caixa preta", tornando o modelo mais compreensível para usuários não técnicos.

### 4.5 Métricas em Produção

As principais métricas consideradas para acompanhamento em produção são accuracy, F1-score e taxa de falsos negativos. Em contexto clínico, a atenção à taxa de falsos negativos é especialmente importante, devido ao impacto potencial de diagnósticos incorretos.

### 4.6 Monitoramento e Drift

Em um cenário real, seria necessário implementar mecanismos de monitoramento do modelo, incluindo:

- acompanhamento da distribuição dos dados de entrada;
- detecção de data drift (mudanças no comportamento dos dados);
- análise de degradação de performance ao longo do tempo.

O MLflow provê infraestrutura para versionamento de modelos e comparação histórica de métricas, sendo a base natural para a implementação desse monitoramento.

### 4.7 Estratégia de Re-treinamento

Com base no monitoramento, o modelo pode ser atualizado por meio de re-treinamento periódico, considerando novos dados disponíveis, mudanças no perfil da população e queda de desempenho do modelo. O pipeline modular implementado facilita esse processo, pois o script `mlflow_simulation.py` pode ser reexecutado com dados atualizados, registrando automaticamente um novo experimento comparável aos anteriores.

---

## 5. Conclusão, Limitações e Próximos Passos

### 5.1 Conclusão

Este projeto demonstrou a construção de um sistema de Machine Learning completo, partindo da experimentação inicial até a operacionalização de um modelo funcional com rastreamento de experimentos via MLflow.

A solução desenvolvida permite a previsão do risco de Alzheimer a partir de variáveis clínicas e cognitivas, sendo disponibilizada por meio de uma interface interativa. Do ponto de vista técnico, o projeto evidenciou a importância de:

- estruturar pipelines de dados consistentes e reprodutíveis;
- rastrear experimentos de forma sistemática com MLflow;
- separar claramente treinamento e inferência;
- selecionar modelos com base em métricas e contexto do problema;
- disponibilizar resultados de forma interpretável.

### 5.2 Limitações

Apesar dos resultados obtidos, o projeto apresenta algumas limitações importantes:

- Dataset limitado, com 336 amostras após filtragem e possível viés amostral;
- Ausência de validação externa, o que pode impactar a generalização;
- Dados longitudinais com múltiplas visitas por paciente, introduzindo possível dependência entre amostras não tratada explicitamente;
- Simplificação de variáveis clínicas, não capturando toda a complexidade do diagnóstico real.

### 5.3 Próximos Passos

Como evolução do projeto, destacam-se:

- coleta de novos dados e expansão da base utilizada;
- validação do modelo em diferentes populações;
- implementação de monitoramento automático de performance e detecção de data drift;
- desenvolvimento de estratégias automatizadas de re-treinamento integradas ao MLflow;
- aplicação de técnicas adicionais de interpretabilidade, como SHAP ou LIME;
- integração de pipeline de CI/CD para automação do deploy.

### 5.4 Considerações Finais

O projeto ilustra a transição de um modelo experimental para um sistema aplicável, evidenciando o papel do engenheiro de Machine Learning na construção de soluções reais. Demonstra não apenas a construção de um modelo preditivo, mas a implementação de um sistema completo alinhado com práticas de MLOps: experimentação rastreável, modelo versionado e interface de inferência operacional.

Mais do que alcançar desempenho preditivo, destaca-se a importância de garantir organização, reprodutibilidade e interpretabilidade — especialmente em contextos com impacto clínico.
