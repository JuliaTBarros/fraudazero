# CONTEXTO — FraudaZero, frente A (Dados e ML) + B1

Diário de trabalho da frente A. Sessão 1: etapas 0 a 8 (A1–A5 + B1, já commitadas e mergeadas no PR #1). Sessão 2: revisão, protocolo de avaliação, A6, A7 e A8 (seções "Sessão 2", no fim do arquivo).
Cada etapa registra: o que foi feito, arquivos, decisões e justificativas, números reais, problemas, pendências e a sugestão de commit.

## Estado atual

- **Sessão 2 concluída (A6, A7 e A8).** Campeão: **Random Forest otimizada**, `champion` = **versão 6**, limiar **0,468518**. Holdout PR-AUC 0,760 [0,626; 0,873]. **Teste final no stream (uma vez): PR-AUC 0,789 [0,696; 0,872], recall 0,707, precisão 0,964, 2 falsos alarmes.** A API serve a v6 com o limiar, sem mudar o contrato; a DAG não sobrescreve o campeão.
- **Achado principal para o relatório (S2.4):** o k-fold estratificado embaralhado superestima os modelos flexíveis em 0,03–0,12 de PR-AUC; o 5-fold temporal concorda com o holdout. A A7 otimizou pelo temporal.
- **Próximo:** A9 (curva de aprendizado) e A10 (`reports/resultados.md`). Pendências das frentes B e C na passagem de bastão (fim do arquivo), incluindo fixar `FRAUD_THRESHOLD=0.468518` no compose.
- **Irvin:** nada foi commitado. Verificações pré-commit na S2.8 (dados do campeão, experimentos do MLflow, ponto de operação com recall ≥ 0,80, reprodutibilidade). Commits sugeridos #7 a #13 na tabela abaixo. `docs/PAUSA.md` e `data/optuna.db` não entram.
- **Atenção nesta máquina:** a porta 8080 continua ocupada por outro projeto seu; o Airflow roda em **8081** por um override local, fora do repositório. A API está rodando com `FRAUD_THRESHOLD=0.468518` passado pelo shell; um `make up` comum volta para 0,5 até a frente B mudar o padrão.

---

## Commits sugeridos

Os commits 1 a 6 (sessão 1) **já estão feitos**: foram juntados no commit `b4d4794` e mergeados em `main` pelo PR #1. Os da sessão 2 começam no #7.

| # | Mensagem | Arquivos |
|---|---|---|
| 1 | `Fixa as versões da pilha do modelo iguais no treino e na API` | `requirements.txt`, `docker/airflow/requirements.txt` |
| 2 | `Roda a DAG uma vez por dia em vez de a cada hora` | `dags/fraud_pipeline.py` |
| 3 | `Adiciona pré-processamento no Pipeline, validação reutilizável e baseline completo no MLflow` | `src/features.py`, `src/validation.py`, `src/train.py` |
| 4 | `Adiciona a EDA do bootstrap dataset com 7 visualizações interpretadas` | `notebooks/01_eda.ipynb`, `reports/figs/eda_01_classes.png` … `reports/figs/eda_07_pca.png` (7) |
| 5 | `Adiciona a validação do baseline: holdout temporal, k-fold, estabilidade e LOO` | `notebooks/02_experimentos.ipynb`, `reports/figs/val_01_baseline_pr_holdout.png`, `reports/figs/val_02_baseline_confusao_holdout.png`, `reports/figs/val_03_baseline_estabilidade_folds.png` |
| 6 | `Documenta decisões, resultados e passagem de bastão da base de ML` | `docs/CONTEXTO.md` |
| 7 | `Formata o notebook de EDA com ruff` | `notebooks/01_eda.ipynb` (só formatação; a do notebook 02 vai junto no #8, mesmo arquivo) |
| 8 | `Separa desenvolvimento e holdout temporal e adiciona validação temporal e IC por bootstrap` | `src/validation.py`, `notebooks/02_experimentos.ipynb`, `reports/figs/val_03_baseline_estabilidade_folds.png` |
| 9 | `Implementa a comparação, a otimização, o campeão e o teste final no stream (compare, diagnose, tune, final, stream_test)` | `src/train.py` (depende do #8), `.gitignore` (`data/optuna.db`) |
| 10 | `Adiciona o notebook 03 de comparação de modelos com análise crítica` | `notebooks/03_comparacao.ipynb`, `scripts/build_nb03.py`, `scripts/nb03_texto.json`, `reports/figs/cmp_01_pr_auc.png` … `reports/figs/cmp_07_estratificado_temporal_holdout.png` (7) |
| 11 | `Registra a otimização de hiperparâmetros da RF e do MLP (A7)` | `reports/figs/opt_random-forest_historico.png`, `opt_random-forest_importancia.png`, `opt_mlp-smote_historico.png`, `opt_mlp-smote_importancia.png` |
| 12 | `Escolhe a Random Forest otimizada como campeã com limiar 0,468518 (A8)` | `src/best_params.json`, `scripts/a8_campeao.py`, `reports/figs/a8_01_limiar.png`, `a8_02_confusao_antes_depois.png`, `a8_stream_confusion_matrix.png`, `a8_stream_pr_curve.png` |
| 13 | `Documenta a sessão 2 da frente A (protocolo, diagnóstico, A6, A7, A8)` | `docs/CONTEXTO.md` (por último) |

Não commitar: `docs/PAUSA.md` (resumo de emergência da queda de energia, já incorporado aqui; pode ser apagado) e `data/optuna.db` (já está no `.gitignore`).

Os notebooks estão salvos **sem saídas** (o filtro `nbstripout` também está ativo no clone). Nenhum dado (`data/`) nem artefato do MLflow (`mlruns/`) entra nos commits; ambos já estão no `.gitignore`.

---

## Decisões gerais (valem para todas as etapas)

### Métricas (definidas antes de qualquer comparação)

| Papel | Métrica | Observação |
|---|---|---|
| **Principal** | **PR-AUC** (`average_precision_score`) | Resume a curva precisão × recall da classe fraude, sem depender de limiar |
| Secundárias | Recall, precisão e F1 **da classe fraude** (classe 1) | Calculadas no limiar em uso (0,5 até a A8 escolher outro) |
| Referência | ROC-AUC | Só para comparação com a literatura; não serve para decidir |

**Por que acurácia não serve:** a base tem 0,1727% de fraudes (492 em 284.807). Um modelo que responde "não é fraude" para tudo acerta 99,83% e não detecta nenhuma fraude. A acurácia é dominada pela classe majoritária e não diz nada sobre o que importa.

**Por que PR-AUC e não ROC-AUC como principal:** a ROC-AUC usa a taxa de falsos positivos, cujo denominador é o número de legítimas (~227 mil no treino). Milhares de falsos alarmes mal mexem nessa taxa, então a ROC-AUC fica otimista em bases muito desbalanceadas. A PR-AUC usa a precisão, que cai visivelmente com cada falso alarme. A referência de uma PR-AUC aleatória é a própria prevalência (~0,0018). No baseline, a diferença é clara: ROC-AUC de 0,974 contra PR-AUC de 0,733, com 2.076 falsos alarmes no holdout.

### Rótulos e a API

O simulador publica `label` no MQTT porque o contrato prevê isso para o dashboard calcular métricas ao vivo. O subscriber envia à API **apenas** `features` ([subscriber/main.py](../subscriber/main.py), função `score`), e a API não aceita `label` (`Transaction` só tem `features`). Portanto os rótulos do `stream.csv` nunca chegam ao modelo. Os rótulos do stream só são lidos offline por `validation.evaluate_on_stream`, para o teste final (A8). **Nenhum contrato foi alterado** (MQTT, API, tabela `transactions`, `FEATURES`).

---

## Etapa 0 — Setup

**Feito**
- Repositório clonado de `https://github.com/JuliaTBarros/fraudazero.git` em `C:\dev\fraudazero` e criada a branch `dados/base-ml` (a partir de `main`, commit `4d0e5cf`). Motivo: a pasta do OneDrive tem espaço e acento no caminho e é sincronizada, o que deixa os volumes do Docker lentos e pode travar arquivos.
- Python 3.11.9 instalado via `winget` (escopo do usuário). Antes só havia o 3.13.
- `.venv` criado com Python 3.11, com `requirements.txt` + ferramentas de desenvolvimento (`ruff`, `nbstripout`, `jupyter`, `nbconvert`, `ipykernel`). Ferramentas de desenvolvimento não entram no `requirements.txt` (não são usadas nos contêineres).
- `nbstripout --install` rodado no clone (precisa ser repetido em cada clone).
- Dataset baixado pelo link público do Kaggle (**sem credencial**) e extraído em `data/raw/creditcard.csv` (150,8 MB).
- Docker Desktop iniciado: engine 28.0.1, 8 GB de RAM e 20 CPUs disponíveis para os contêineres. `make check` → `docker-compose.yml ok`.

**Arquivos alterados**
- `requirements.txt` e `docker/airflow/requirements.txt`: versões da "pilha do modelo" fixadas e idênticas nos dois arquivos.

**Decisão: versões exatas na pilha do modelo**
O modelo é treinado no contêiner do Airflow (`/opt/train-venv`) e desserializado (pickle) na API, que usa outra imagem. Antes, só scikit-learn e imbalanced-learn eram fixados; numpy, scipy, pandas e mlflow aceitavam qualquer versão nova, e cada imagem podia instalar uma diferente em builds feitos em dias diferentes. Agora os dois arquivos têm o mesmo bloco:

| Pacote | Versão |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.13.0 |
| mlflow | 2.22.5 |

São as versões que o pip resolveu no `.venv` (Python 3.11). Validado: o modelo treinado no Airflow carrega e responde na API (etapa 2).

**Problemas**
- O hook de formatação do Claude Code procura o `ruff` no `.venv` do diretório aberto no editor. Como a sessão foi aberta na pasta do OneDrive, o hook não achou o `ruff`; a formatação foi feita manualmente (`.venv\Scripts\ruff format` + `ruff check`, ambos limpos). Abrindo `C:\dev\fraudazero` no VS Code, o hook passa a funcionar.

---

## Etapa 1 — A1. Preparação dos dados

**Feito**
`python -m src.train prepare` (local, no `.venv`; a DAG roda o mesmo comando no Airflow). Ordena por `Time` e separa os primeiros 80% (treino/bootstrap) dos últimos 20% (fluxo do simulador). Nenhum código foi alterado.

**Números reais**

| Arquivo | Linhas | Fraudes | % fraude | `Time` (s) | Nulos | Duplicados (fraudes entre eles) |
|---|---|---|---|---|---|---|
| `data/raw/creditcard.csv` | 284.807 | 492 | 0,1727% | 0 – 172.792 | 0 | 1.081 (19) |
| `data/processed/train.csv` | 227.845 | 417 | 0,1830% | 0 – 145.247 | 0 | 840 (18) |
| `data/stream/stream.csv` | 56.962 | 75 | 0,1317% | 145.248 – 172.792 | 0 | 241 (1) |

**Leituras**
- O corte é limpo no tempo: o treino termina em `Time = 145.247` e o stream começa em `145.248`. Não há sobreposição.
- O stream tem proporção de fraudes menor (0,13% contra 0,18%). Isso importa para a avaliação final: a PR-AUC no stream tende a ser mais baixa, porque a prevalência (o "piso" da PR-AUC) é menor.
- A base cobre só ~48 horas. O stream corresponde ao fim do segundo dia (das ~16h20 às ~24h, contando a partir do início da coleta).

---

## Etapa 2 — B1. Primeiro `make up`

**Feito**
- `docker compose up --build -d` do zero (build de ~10 min na primeira vez, principalmente a imagem do Airflow com o venv de treino).
- Os **8 serviços** ficaram de pé: postgres (healthy), mosquitto, mlflow, airflow, api, subscriber, simulator, dashboard.
- A DAG `fraudazero_pipeline` rodou sozinha na primeira subida e terminou **com sucesso nas 4 tarefas** (`ingest` → `prepare` → `baseline` → `reload_api`) em ~56 s.

**Verificação de ponta a ponta (números reais)**

| Checagem | Resultado |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` |
| `POST /predict` com a 1ª fraude do stream | `score` 0,99999…, `is_fraud: true` |
| `POST /predict` com a 1ª legítima do stream | `score` 0,092, `is_fraud: false` |
| `POST /predict` com features faltando | HTTP 422 |
| Postgres, último minuto | 110 transações recebidas, 110 com score (100%) |
| Latência média da API (subscriber) | 6,7 ms |
| Dashboard `:8501` | HTTP 200 |
| MLflow | experimento `baseline`, run `logreg-baseline` com métricas, parâmetros, 4 artefatos e modelo; `fraudazero@champion` = versão 1 |
| `pytest tests` (smoke) | 2 passed |
| Logs (todos os serviços, depois da DAG) | nenhum erro recorrente |

**Problemas encontrados e como foram resolvidos**

1. **Porta 8080 ocupada nesta máquina** pelo contêiner `avaliacao-pratica-release-app-1`, de outro projeto do Irvin, que sobe sozinho com o Docker Desktop. O Airflow ficou como `Created`, sem iniciar.
   - Não parei o outro contêiner (é de outro projeto; a ação foi bloqueada e deixei a decisão com o Irvin).
   - **Solução local, fora do repositório:** um override do Compose que publica o Airflow em `127.0.0.1:8081`. O `docker-compose.yml` do repositório **não mudou**. Conteúdo do override (salve como `compose.local.yml` **fora** do repo, ou não o commite):
     ```yaml
     services:
       airflow:
         ports: !override
           - "127.0.0.1:8081:8080"
     ```
     Uso: `docker compose -f docker-compose.yml -f <caminho>/compose.local.yml up -d`.
   - **Alternativa:** `docker stop avaliacao-pratica-release-app-1` e usar o `make up` normal, com o Airflow em `:8080`.
   - **Para o README (frente B/C):** acrescentar em "Problemas comuns" que o Airflow usa a 8080, uma porta disputada por muitos projetos.
2. **As imagens dos serviços copiam `src/` no build**; o Airflow monta `src/` ao vivo. Depois de criar `src/features.py`, foi preciso reconstruir as imagens (`up -d --build`), senão a API não conseguiria carregar o modelo novo (ela precisa de `src.features` para o unpickle). O `make up` já faz `--build`, então em clone limpo isso não acontece.
3. **Transações sem score no início:** 197 linhas ficaram com `score` nulo. São de antes de existir modelo registrado (API respondendo 503) e do momento em que a API foi reconstruída. Depois disso, 100% das transações são pontuadas.

**Pendências encontradas para a frente B (não corrigidas, fora do escopo)**
- **B3:** quando o simulador reinicia, os `tx_id` recomeçam de `tx-0-0`. Como o insert é `ON CONFLICT DO NOTHING`, as repetidas são descartadas, e as linhas antigas sem score **nunca são repontuadas**. Opções: incluir no `tx_id` um identificador da execução, ou fazer `ON CONFLICT DO UPDATE` quando o score antigo for nulo.
- **B3:** uma exceção dentro de `on_message` (ex.: Postgres fora do ar) derruba o loop do paho; o contêiner reinicia pelo `restart: unless-stopped`. Sem `client_id` fixo e `clean_start=False`, as mensagens QoS 1 recebidas nesse intervalo se perdem.

**Sugestão de commit:** nenhuma (não houve mudança de código nesta etapa).

---

## Etapa 3 — Correção da DAG e proteção do campeão

**Feito**
- [dags/fraud_pipeline.py](../dags/fraud_pipeline.py): `schedule` de `@hourly` para `@daily`. Com `catchup=False` e `start_date` no passado, a DAG continua rodando sozinha na primeira subida; depois, uma vez por dia (ou manualmente com `make train`).
- [src/train.py](../src/train.py): nova função `_set_champion_if_missing`. O `baseline` continua registrando uma nova versão do modelo `fraudazero` a cada execução, mas só aponta o alias `champion` para ela **se ainda não existir um campeão**.

**Por quê**
Antes, a cada hora o baseline retreinava e **movia o `champion` para a regressão logística**. Depois que a A8 escolher um modelo melhor, ele seria substituído pelo baseline em menos de uma hora, sem ninguém perceber. Retreinar de hora em hora também consome RAM à toa numa stack que já pede 6–8 GB.

**Validação (real)**
- 1ª execução (automática): log `alias 'champion' -> versão 1`.
- 2ª execução (`airflow dags trigger`): registrou a versão 2 e logou `alias 'champion' mantido na versão 1`. O alias continua na versão 1.

**Pendência para a frente B (B6):** quando o comando `final` existir (A8), trocar a tarefa `baseline` por `final` na DAG. É o `final` que deve mover o `champion`.

---

## Etapa 4 — A2. EDA

**Feito**
- [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb): estatísticas descritivas, ausentes, duplicados, **7 visualizações**, cada uma com interpretação logo abaixo, seção de tratamento (etapa 5) e resumo para o relatório. Executado do início ao fim sem erros (`nbconvert --execute`); salvo **sem saídas**.
- Figuras exportadas em `reports/figs/` (PNG, 150 dpi), prontas para o relatório:

| Figura | Arquivo | Mensagem principal |
|---|---|---|
| 1. Classes | `eda_01_classes.png` | 417 fraudes (0,18%) contra 227.428 legítimas: acurácia inútil, PR-AUC |
| 2. Amount por classe (log) | `eda_02_amount.png` | 37,6% das fraudes até 1,50 (105 de exatamente 1,00; "teste de cartão"); 27 de 99,99 |
| 3. Volume e taxa ao longo de Time | `eda_03_tempo.png` | Ciclo de 2 dias; taxa de fraude dispara nos vales de volume (2,05% na hora 26) |
| 4. Taxa por hora do dia | `eda_04_hora_do_dia.png` | 1,71% às 2h e 1,04% às 4h, contra 0,18% de média |
| 5. Correlação com Class | `eda_05_correlacao.png` | V17 (−0,364), V14 (−0,313), V12 (−0,269), V10 (−0,240); Amount e Time ~0 |
| 6. Distribuição das top 4 V | `eda_06_top_v.png` | Fraudes na cauda negativa; parte delas sobreposta às legítimas |
| 7. PCA 2D | `eda_07_pca.png` | Maioria das fraudes longe do núcleo legítimo; parte dentro dele (limite do recall) |

**Números reais (train.csv)**
- 227.845 linhas, 31 colunas numéricas, 0 nulos, 840 cópias exatas (18 fraudes).
- `Amount`: média 90,82, mediana 23,40, máximo 19.656,53, assimetria 14,07 (0,14 depois de `log1p`). Fraudes: mediana 12,31, máximo 2.125,87. 1.567 transações com valor 0 (22 fraudes).
- Desvio-padrão das `V`: de 0,33 a 1,94.

**Como o notebook foi gerado**
O `.ipynb` foi montado por script (células de código + markdown) e executado com `jupyter nbconvert --execute` numa cópia fora do repositório, para conferir que roda e para ler as saídas **antes** de escrever as interpretações. Todo número citado no texto saiu dessa execução. Para reexecutar: abra no VS Code/Jupyter com o kernel do `.venv` e rode tudo (menos de 1 minuto). O notebook descobre a raiz do projeto sozinho e lê `data/processed/train.csv`.

**Decisão: paleta dos gráficos**
Legítima/treino = azul `#2a78d6`, fraude/validação = laranja `#eb6834` (as duas primeiras cores de uma paleta categórica validada para daltonismo). As mesmas cores em todas as figuras.

---

## Etapa 5 — A3. Tratamento dos dados

**Feito**
- [src/features.py](../src/features.py) (novo): `preprocessor()`, um `ColumnTransformer` usado como **primeiro passo de todo Pipeline**. Assim, cada transformação é ajustada só nos dados de treino de cada fold.
- [src/validation.py](../src/validation.py) `load_train()`: remove as cópias exatas do treino.
- Seção 4 do notebook de EDA com a tabela de decisões.

**Decisões e justificativas** (texto pronto para a Metodologia do relatório)

| Tema | Decisão | Justificativa |
|---|---|---|
| Ausentes | Nada a fazer | 0 nulos; checagem registrada no notebook |
| Duplicados | Remover as 840 cópias exatas do **treino**; stream intacto | No k-fold, uma cópia pode cair no treino e a outra na validação, e a métrica fica otimista por memorização (principalmente KNN e árvores). Nenhum par de cópias tem rótulos diferentes. Fica fora do Pipeline porque não se removem linhas na inferência; não há vazamento, pois é uma regra fixa, sem estatística aprendida. Resultado: 227.005 linhas e 399 fraudes |
| `Amount` (escala) | `log1p` + `StandardScaler`, dentro do Pipeline | Assimetria 14,07 → 0,14; KNN, MLP e logística são sensíveis à escala |
| `Amount` (outliers) | **Não remover** | Valores extremos são reais e as fraudes ocupam faixas específicas; remover tiraria sinal. O `log1p` já reduz a influência dos extremos |
| `Time` | Hora do dia (`Time % 86400`) em **seno e cosseno**, dentro do Pipeline | `Time` bruto não se repete: o stream (145.248–172.792 s) está fora do intervalo do treino (0–145.247 s). A hora do dia se repete e carrega sinal (Fig. 4). Seno/cosseno deixam 23h perto de 0h |
| `V1`–`V28` | `StandardScaler`, dentro do Pipeline | São componentes de PCA (média ~0), mas com desvios de 0,33 a 1,94. Nenhuma é removida: correlação linear baixa não significa variável inútil |
| Desbalanceamento | `class_weight="balanced"` no baseline; SMOTE/undersampling **só** em `imblearn.pipeline.Pipeline` | A reamostragem acontece só no treino de cada fold; a validação mantém a proporção real, que é a que vale para a PR-AUC |

**Contrato preservado:** a API continua recebendo `Time` bruto e as 30 `FEATURES`. A conversão para hora do dia acontece **dentro do modelo**, então nenhuma frente precisa mudar nada. Validado na API real (etapa 2).

**Detalhe técnico importante (pickle):** as funções do pré-processamento ficam em `src/features.py`, e não em `lambda`, para o modelo salvo pelo MLflow ser carregado na API (a imagem da API tem `src/` no `PYTHONPATH`). **Se alguém mover ou renomear `hour_of_day_cyclic` ou `_hour_names`, os modelos já registrados param de carregar.**

---

## Etapa 6 — A4. Estratégia de validação

**Feito**
- [src/validation.py](../src/validation.py) (novo), funções reutilizáveis. Todas recebem um Pipeline **não treinado** e fazem `clone`, então escala e reamostragem são ajustadas só no treino de cada divisão:

| Função | O que faz |
|---|---|
| `load_train()` | `train.csv` em ordem de `Time`, sem as 840 cópias (227.005 linhas, 399 fraudes) |
| `metrics(y, scores, threshold)` | PR-AUC, ROC-AUC, recall, precisão, F1 (classe fraude) |
| `holdout_temporal(model, X, y)` | Treina nos primeiros 80% (por `Time`), valida nos 20% finais; devolve métricas de treino e validação, tempo de treino, `y_valid` e `scores_valid` |
| `cv_stratified(model, X, y, k=5)` | Stratified k-fold com `SEED = 42` (**os mesmos folds para todos os modelos**); DataFrame com métricas de treino e validação por fold |
| `summarize_cv(folds)` | Média e desvio de cada coluna, no formato do `mlflow.log_metrics` |
| `evaluate_on_stream(fitted_model)` | Teste final no `stream.csv`. **Usar uma única vez, com o campeão (A8)** |
| `plot_confusion`, `plot_pr_curve` | Figuras padronizadas (também vão para o MLflow) |
| `gap(train, valid)` | Diferença treino − validação por métrica (diagnóstico de overfitting) |

- [notebooks/02_experimentos.ipynb](../notebooks/02_experimentos.ipynb) (novo): seções 1 a 5 prontas (métricas, holdout, k-fold e estabilidade, comparação entre estratégias, LOO); a seção 6 traz o modelo de uso para a A6. Executado do início ao fim sem erros; salvo sem saídas.
- Figuras em `reports/figs/`: `val_01_baseline_pr_holdout.png`, `val_02_baseline_confusao_holdout.png`, `val_03_baseline_estabilidade_folds.png`.

**Decisões e justificativas**
- **Holdout temporal 80/20 dentro do `train.csv`:** os dados são sequenciais e o sistema real aprende com o passado para pontuar o futuro. Nenhuma linha posterior entra no treino. Corte em `Time = 125.329 s`: treino com 181.604 linhas (347 fraudes), validação com 45.401 linhas (52 fraudes).
- **Stratified 5-fold:** mede média e **variabilidade**; a estratificação garante ~80 fraudes em cada fold de validação. Com a semente fixa, todos os modelos usam os mesmos folds, e a comparação pode ser feita fold a fold.
- **Os dois são reportados:** o k-fold ignora o tempo e pode ser otimista para modelos flexíveis; a escolha final deve olhar principalmente o holdout temporal.
- **`stream.csv` reservado** para uma única avaliação final do campeão (A8). Olhar o stream durante a escolha de modelo transformaria o teste em validação.
- **Leave-One-Out não se aplica**, por quatro motivos:
  1. **Custo:** seriam 227.005 treinos. A ~0,3 s por treino, só a regressão logística, o modelo mais barato, levaria cerca de 20 horas num núcleo, contra ~2 s do 5-fold; RF, AdaBoost e MLP, semanas ou meses.
  2. **Sem métricas por fold:** cada validação tem uma única linha, e PR-AUC, recall e precisão não são definidos para uma linha. Não há análise de estabilidade, que é o que a especificação pede.
  3. **Variância alta da estimativa**, porque os conjuntos de treino são quase idênticos.
  4. **Ignora o tempo.**
  - **Demonstração:** rodamos o LOO numa subamostra de 1.000 linhas (50 fraudes + 950 legítimas): 1.000 treinos, PR-AUC agregada de 0,8752. Esse número **não é comparável** com os outros, porque a subamostra tem 5% de fraudes e a PR-AUC sobe com a prevalência. O tempo medido variou entre execuções (de 4,7 s a 56,9 s, com 20 núcleos, por causa do aquecimento dos processos paralelos); a projeção para a base inteira ficou entre 20 e 23 horas.

**Sugestão de commit:** ver a tabela [Commits sugeridos](#commits-sugeridos) (commits 3 e 5).

---

## Etapa 7 — A5. Baseline registrado no MLflow

**Feito**
- [src/train.py](../src/train.py):
  - `baseline_model()`: `Pipeline([preprocessor(), LogisticRegression(class_weight="balanced", max_iter=1000)])`.
  - `evaluate_and_log(model, run_name, params, experiment, register=False)`: a **forma padrão de avaliar e registrar qualquer modelo**. Faz holdout temporal + 5-fold + treino final em todo o `train.csv` e registra no MLflow parâmetros, métricas, `cv_folds.csv`, matriz de confusão, curva PR e o modelo (com `input_example`). Com `register=True`, também registra uma versão do modelo `fraudazero`.
  - `baseline()` agora usa essas duas funções e registra **ROC-AUC** e as métricas de **treino** (antes só registrava 4 métricas de validação).

**Onde está no MLflow** (`http://localhost:5001`)
- Experimento `baseline`, run `logreg-baseline`, id `4df0683d24564d7bba42588cf01b29a8` (gerado pela DAG em 23/11/2026). Modelo `fraudazero` versão 1, alias `champion`.
- A 2ª execução da DAG gerou o run `eb568178ef244d7abddb995130fd00d7` (versão 2, sem alias), com as **mesmas métricas**: o treino é determinístico.
- Em outra máquina ou depois de `make reset`, os ids mudam; as métricas, não.
- **Nomes das métricas no MLflow:** sem prefixo = holdout temporal (validação); `train_*` = treino do holdout; `cv_valid_*_mean/std` e `cv_train_*_mean/std` = 5-fold; `fit_seconds`.

**Números reais (os mesmos no `.venv` local e no Airflow)**

Holdout temporal (limiar 0,5):

| Métrica | Treino | Validação | Diferença |
|---|---|---|---|
| **PR-AUC** | 0,7255 | **0,7330** | −0,0074 |
| ROC-AUC | 0,9872 | 0,9742 | 0,0130 |
| Recall | 0,9193 | 0,9038 | 0,0155 |
| Precisão | 0,0748 | 0,0221 | 0,0527 |
| F1 | 0,1383 | 0,0432 | 0,0951 |

Matriz de confusão do holdout: 47 fraudes detectadas e 5 perdidas, com 2.076 falsos alarmes e 43.273 legítimas corretas.

Stratified 5-fold (validação):

| Fold | PR-AUC | ROC-AUC | Recall | Precisão | F1 |
|---|---|---|---|---|---|
| 1 | 0,7660 | 0,9692 | 0,8734 | 0,0638 | 0,1190 |
| 2 | 0,6387 | 0,9602 | 0,8875 | 0,0598 | 0,1121 |
| 3 | 0,7616 | 0,9773 | 0,9125 | 0,0677 | 0,1260 |
| 4 | 0,6472 | 0,9649 | 0,8875 | 0,0551 | 0,1038 |
| 5 | 0,7608 | 0,9823 | 0,9125 | 0,0575 | 0,1081 |
| **Média ± desvio** | **0,7148 ± 0,0657** | 0,9708 ± 0,0090 | 0,8947 ± 0,0173 | 0,0608 ± 0,0050 | 0,1138 ± 0,0088 |

PR-AUC de treino nos folds: 0,7147 ± 0,0163.

Efeito do limiar no holdout (insumo para a A8):

| Recall mínimo exigido | Limiar | Recall | Precisão |
|---|---|---|---|
| 0,9 | 0,535532 | 0,9038 | 0,0255 |
| 0,8 | 0,951815 | 0,8077 | 0,1927 |
| 0,7 | 0,999968 | 0,7115 | 0,6727 |

**Interpretação (pronta para o relatório)**
- **Sem overfitting:** a PR-AUC de treino e a de validação são praticamente iguais, no holdout (0,726 × 0,733) e na média dos folds (0,7147 × 0,7148).
- **Indício de underfitting:** o modelo linear não passa de ~0,73 de PR-AUC nem no treino; não captura relações não lineares como os picos de `Amount`. O espaço para ganho está nos modelos da A6.
- **Estabilidade:** a PR-AUC varia bastante entre folds (de 0,639 a 0,766; coeficiente de variação de 9,2%), porque cada fold tem só ~80 fraudes. A ROC-AUC varia 10 vezes menos (0,9%). **Diferenças de PR-AUC menores que ~0,07 entre modelos estão dentro do ruído.**
- **Holdout × k-fold:** o holdout (0,733) está dentro do intervalo dos folds e perto da média. Para este modelo, o k-fold embaralhado não foi otimista.
- **O limiar de 0,5 não serve** para este modelo: com `class_weight="balanced"`, os scores sobem, e o limiar padrão gera 2.076 falsos alarmes para 47 acertos. Aceitar 80% de recall multiplica a precisão por ~7,5 (de 2,6% para 19,3%).
- A precisão de treino (7,5%) é maior que a de validação (2,2%) também porque a validação tem menos fraudes (0,11% contra 0,19%): a precisão cai junto com a prevalência.

---

## Etapa 8 — Passagem de bastão

### O que está pronto (A1–A5 + B1)

| Item | Onde |
|---|---|
| Dados preparados e números de referência | etapa 1 |
| Stack de pé, DAG verde, API servindo o baseline | etapa 2 |
| EDA com 7 visualizações interpretadas | `notebooks/01_eda.ipynb`, `reports/figs/eda_*.png` |
| Pré-processamento dentro do Pipeline | `src/features.py` |
| Validação reutilizável (holdout temporal, 5-fold, teste no stream, gráficos) | `src/validation.py` |
| Avaliação + registro padronizado no MLflow | `src/train.py` → `evaluate_and_log` |
| Baseline completo no MLflow, `champion` = versão 1 | experimento `baseline` |
| Justificativas (métricas, tratamento, validação, LOO) | este arquivo e os dois notebooks |

### Como começar a A6 (comparação de modelos)

1. **Ambiente:** `C:\dev\fraudazero`, branch `dados/base-ml` (ou uma nova a partir dela, depois do commit). Ative o `.venv`: `.venv\Scripts\Activate.ps1`. Para registrar no MLflow a partir da máquina local: `$env:MLFLOW_TRACKING_URI = "http://localhost:5001"`, com a stack de pé.
2. **Implemente `compare()` em `src/train.py`:** para cada modelo, monte um Pipeline e chame
   ```python
   evaluate_and_log(model, run_name="random-forest", params={...}, experiment="comparacao")
   ```
   Isso registra holdout, 5-fold (média ± desvio), métricas de treino, matriz de confusão, curva PR, `cv_folds.csv` e o modelo, no mesmo formato do baseline. Não use `register=True` na comparação: o registro do campeão é da A8.
3. **Todo modelo começa com `("prep", preprocessor())`.** Para árvores, RF e AdaBoost a padronização é inócua, mas mantê-la deixa os Pipelines iguais e simplifica a API.
4. **Desbalanceamento:** `class_weight="balanced"` onde existir (árvore, RF, logística). Para SMOTE ou undersampling, use **`imblearn.pipeline.Pipeline`**: o `Pipeline` do sklearn não aceita reamostragem. Nunca reamostre fora do Pipeline.
5. **KNN precisa de subamostra (custo):** a predição do KNN compara cada transação com todo o treino. Faça o **undersampling dentro do Pipeline**, mantendo todas as fraudes: `RandomUnderSampler(sampling_strategy=0.02, random_state=42)` reduz as legítimas a ~50 por fraude **só no treino de cada fold**, e a validação continua com a proporção real. Exemplo completo na seção 6 do `02_experimentos.ipynb`; ele foi testado e roda em segundos. **Não** subamostre a base antes do split: a validação ficaria com mais fraudes que a realidade e a PR-AUC não seria comparável.
6. **AdaBoost e MLP** não têm `class_weight`: use SMOTE ou undersampling via `imblearn`, ou registre que rodaram sem tratamento e compare.
7. **Custo:** `evaluate_and_log` treina 7 vezes (holdout + 5 folds + final). RF e MLP em ~180 mil linhas podem levar minutos por treino. Rode a comparação **localmente ou no notebook**, não no Airflow (o plano já recomenda isso).
8. **Comparação justa:** os folds são os mesmos para todos (semente 42). Compare a PR-AUC **fold a fold** e lembre que diferenças menores que ~0,07 estão dentro do ruído do baseline. Reporte holdout temporal e 5-fold.
9. **Não avalie no `stream.csv`** antes da A8.

### Pendências (A6 em diante)

| Tarefa | O que falta | Observações |
|---|---|---|
| **A6** | `compare()` com KNN, Árvore, RF, AdaBoost e MLP; tabela comparativa e análise crítica na seção 6 do `02_experimentos.ipynb` | Base pronta: `evaluate_and_log` |
| **A7** | `tune()` com Optuna (já instalado) no(s) melhor(es) modelo(s), objetivo PR-AUC, experimento `otimizacao` | Use `val.cv_stratified` como função objetivo |
| **A8** | Escolher o campeão e o **limiar** (curva PR; ver a tabela de limiares da etapa 7); criar `src/best_params.json` e o comando `final`; avaliar **uma vez** no stream com `val.evaluate_on_stream`; mover o `champion` | Informar o `FRAUD_THRESHOLD` às frentes B e C. O limiar 0,5 não serve para o baseline |
| **A9** | Curva de aprendizado e diagnóstico de over/underfitting do campeão | As métricas `train_*` e `cv_train_*` já estão no MLflow para todos os modelos |
| **A10** | `reports/resultados.md` com os números finais e o run de origem | As figuras já estão em `reports/figs/` |
| **B6** | Trocar a tarefa `baseline` por `final` na DAG | Depende da A8 |
| **B3** | Robustez do subscriber (ver etapa 2) | Encontrado aqui, não corrigido |
| **README** | Mencionar o conflito comum da porta 8080 | Ver etapa 2 |

### Decisões que a próxima pessoa NÃO precisa refazer
Métricas (PR-AUC principal), remoção de duplicados, transformação de `Time`, escala de `Amount`, não remover outliers, holdout temporal + 5-fold, LOO justificado, stream reservado para o teste final. Tudo justificado nas etapas 5 e 6 e nos notebooks.

---

## Verificação final pré-commit

Feita a pedido do Irvin antes de commitar. Resultado: **nenhum problema encontrado**; a única alteração foi esta seção (e uma linha no "Estado atual").

### 1. Git
- `git check-ignore -v` confirma que ficam fora do commit: `data/raw/creditcard.csv` (`.gitignore:14`), `data/processed/train.csv` (`:15`), `data/stream/stream.csv` (`:16`), `mlruns/*` (`:20`) e `.venv/` (`:4`). O `.gitignore` não precisou de mudança.
- O override da porta 8081 (`compose.local.yml`) está **fora do repositório** (pasta temporária da sessão); nenhum arquivo de override existe dentro de `C:\dev\fraudazero`.
- Um `git add -A` incluiria **19 arquivos, nenhum acima de 5 MB**. O maior é `reports/figs/eda_07_pca.png`, com 76 KB; os notebooks têm 24 KB e 17 KB; o `docs/CONTEXTO.md` tem ~30 KB.

### 2. LOO (notebook 02, seção 5)
- Subamostra **já era enriquecida e com proporção documentada**: 1.000 linhas = **50 fraudes + 950 legítimas** (5,0%), sorteadas sem reposição com `np.random.RandomState(42)`. **Não foi refeita.**
- Métrica: PR-AUC agregada de **0,8752** (previsões dos 1.000 modelos juntadas no fim).
- Interpretação: o número serve só para mostrar que o LOO roda e quanto custa. Não é comparável com a PR-AUC do baseline (0,733), porque a PR-AUC sobe com a prevalência (5% na subamostra contra 0,18% na base). A justificativa para não usar LOO (custo de ~20 h só para a logística, nenhuma métrica por fold, variância alta, ignora o tempo) já estava escrita e não mudou.

### 3. Estabilidade do k-fold

| Fold | Fraudes na validação | Fraudes no treino | PR-AUC validação | PR-AUC treino |
|---|---|---|---|---|
| 1 | 79 | 320 | 0,7660 | 0,7131 |
| 2 | 80 | 319 | 0,6387 | 0,7347 |
| 3 | 80 | 319 | 0,7616 | 0,6924 |
| 4 | 80 | 319 | 0,6472 | 0,7252 |
| 5 | 80 | 319 | 0,7608 | 0,7081 |

Cada fold de validação tem 45.401 linhas. Média ± desvio da validação: 0,7148 ± 0,0657. O notebook 02 **já discute** esse desvio como limitação ligada ao número pequeno de positivos ("cada fold tem só ~80 fraudes: algumas fraudes com score baixo já mudam bastante a área"), com a consequência prática para a A6 (diferenças menores que ~0,07 estão dentro do ruído; comparar fold a fold). Nada foi acrescentado.

### 4. Tamanho dos notebooks
- `notebooks/01_eda.ipynb`: 24.012 bytes (0,023 MB).
- `notebooks/02_experimentos.ipynb`: 17.357 bytes (0,017 MB).
- **Nenhuma saída e nenhuma imagem embutida** (0 `outputs`, 0 `image/png`). As figuras existem só como PNG em `reports/figs/`.

### 5. Baseline, holdout temporal, limiar 0,5

| | Previsto legítima | Previsto fraude |
|---|---|---|
| **Real legítima** | TN = 43.273 | FP = 2.076 |
| **Real fraude** | FN = 5 | TP = 47 |

Total 45.401 (52 fraudes). Classe fraude: **precision 0,0221**, **recall 0,9038**, **F1 0,0432** (PR-AUC 0,7330, que não depende do limiar). Iguais aos valores do MLflow (run `4df0683d24564d7bba42588cf01b29a8`).

### 6. Reprodutibilidade
- Os dois notebooks foram executados **do zero, cada um num kernel novo** (`jupyter nbconvert --execute`, equivalente a Restart & Run All): **0 erros**, 7 figuras no 01 e 3 no 02.
- Comparando com a execução anterior: **todas as saídas numéricas são idênticas**, e as 10 figuras regeneradas são **idênticas byte a byte** às que estão em `reports/figs/`. As únicas diferenças são tempos medidos (`fit_seconds`, duração do LOO), que variam por natureza, e o endereço de memória de um objeto de tabela.
- **Pontos aleatórios e sementes:**

| Onde | Semente |
|---|---|
| `StratifiedKFold` (`src/validation.py`) | `shuffle=True, random_state=SEED` (42) |
| Amostra de legítimas da PCA (notebook 01) | `random_state=np.random.RandomState(42)` |
| `PCA` (notebook 01) | `random_state=42` |
| Subamostra do LOO (notebook 02) | `np.random.RandomState(42)` |
| `RandomUnderSampler` do exemplo de KNN (notebook 02) | `random_state=42` |
| Holdout temporal | Sem aleatoriedade (corte por posição em `Time`) |
| `LogisticRegression` | Sem `random_state`: o solver padrão `lbfgs` é determinístico (o parâmetro só vale para `sag`, `saga` e `liblinear`). Evidência: métricas idênticas no `.venv` local, na 1ª e na 2ª execução da DAG no Airflow |

**Recomendação para a A6:** RF, Árvore, AdaBoost, MLP e SMOTE **são** aleatórios; passe `random_state=42` em todos.

### Arquivos alterados nesta verificação
- `docs/CONTEXTO.md` (esta seção e uma linha no "Estado atual"). Nenhum outro arquivo mudou.

---

# Sessão 2 (24/11/2026)

## S2.1 — Revisão do estado atual

**Git**
- A sessão começou em `main`, limpa. `dados/base-ml` local = `origin/dados/base-ml` = `b4d4794`, e `main` (`e47b46b`, merge do PR #1) tem o mesmo conteúdo (`git diff` vazio).
- **Commits sugeridos 1 a 6: todos feitos**, juntados num único commit (`b4d4794`, 19 arquivos, exatamente os da tabela). **Nada pendente** no working tree no início da sessão.
- Troquei para `dados/base-ml` (pedido do Irvin). Como ela já foi mergeada, sugiro criar uma branch nova antes de commitar a sessão 2 (ex.: `git switch -c dados/modelos`); o working tree acompanha a troca.

**Stack**
- Docker Desktop estava desligado; iniciado. Os contêineres voltaram sozinhos (`restart: unless-stopped`).
- **Porta 8080 continua ocupada** por `avaliacao-pratica-release-app-1` (outro projeto). Mesma solução da sessão 1: override `compose.local.yml` (Airflow em `127.0.0.1:8081`), recriado **fora do repositório**, na pasta temporária da sessão. Subida com `docker compose -f docker-compose.yml -f <override> up --build -d`: os 8 serviços de pé.
- DAG disparada manualmente: `ingest`, `prepare`, `baseline` e `reload_api` com **sucesso** (~1 min).
- `GET /health` → `{"status":"ok","model_loaded":true}`. MLflow, dashboard e Airflow respondem (HTTP 200). `fraudazero@champion` = **versão 1** (run `4df0683d…`, baseline). As versões 3 e 4 vieram das execuções da DAG (diária automática e manual) e **não moveram o alias**, como esperado.
- Smoke test: **2 passed**.

**Ruff (0.17.6)**
- `ruff format --check`: os dois notebooks da frente A estavam fora do formato (o ruff passou a formatar `.ipynb`). **Corrigido** com `ruff format notebooks` (só código das células; nenhuma lógica mudou). Agora 27/27 arquivos formatados.
- `ruff check src notebooks dags tests`: limpo. O `ruff check .` acusa 11 avisos (`BLE001` exceção genérica, `S110`, `I001`), **todos fora da frente A** (`api/`, `app/`, `embedded/`, `subscriber/`, `.claude/hooks/`). Não mexi; registrados como pendência.

**Problema encontrado (frente B, não corrigido): sem transações novas depois de reiniciar o simulador**
- Depois do rebuild, nenhuma linha nova entrou em `transactions` (última às 15:14:50, antes do rebuild; 45.895 linhas no total).
- Causa confirmada com `mosquitto_sub`: o simulador recomeça em `tx-0-0`, ids que já existem no banco, e o `INSERT ... ON CONFLICT (tx_id) DO NOTHING` do subscriber descarta todos. **Com 0,5 s por transação, o dashboard fica sem dados novos por ~6 horas** depois de qualquer reinício do simulador. É a pendência B3 da etapa 2, com efeito maior do que o registrado antes. **Afeta a demo.**
- Sugestão para a frente B: incluir no `tx_id` um identificador da execução (ex.: `tx-<timestamp de início>-<volta>-<i>`). Paliativo sem mudar código: `make reset` (apaga o banco). Não executei (destrutivo).

**M1:** nada quebrado. A frente A segue.

**Arquivos alterados:** `notebooks/01_eda.ipynb`, `notebooks/02_experimentos.ipynb` (formatação), `docs/CONTEXTO.md`.

---

## S2.2 — Protocolo de avaliação (antes de qualquer modelo novo)

**Problema corrigido:** na sessão 1, o 5-fold do baseline usava o `train.csv` inteiro, **incluindo o trecho do holdout temporal**. Enquanto só havia um modelo, isso não era vazamento. Agora há seleção de modelo, de hiperparâmetros e de limiar: se o holdout entrasse nos folds, a otimização veria o holdout, e a comparação nele deixaria de ser independente.

**Protocolo (três conjuntos, um papel cada)**

| Conjunto | Linhas (fraudes) | `Time` (s) | Papel |
|---|---|---|---|
| **Desenvolvimento** = primeiros 80% do `train.csv` | 181.604 (347) | 0 – 125.328 | Stratified 5-fold (69 ou 70 fraudes por fold) e **toda** otimização de hiperparâmetros |
| **Holdout temporal** = últimos 20% do `train.csv` | 45.401 (52) | 125.329 – 145.247 | Comparar modelos e escolher o campeão e o limiar. Modelos treinados no desenvolvimento |
| **`stream.csv`** | 56.962 (75) | 145.248 – 172.792 | Teste final, **uma vez**, só no campeão, no fim da A8 |

O modelo que vai para produção é retreinado em desenvolvimento + holdout (todo o `train.csv`), com hiperparâmetros e limiar já fixados.

**Concordo com o protocolo proposto**, com uma observação para o relatório: o holdout é usado para várias decisões (modelo, limiar), então o número dele fica levemente otimista para o campeão. É exatamente por isso que o `stream.csv` fica intocado até o fim: ele é a estimativa sem viés de seleção. Outra limitação: o holdout tem só 52 fraudes, então uma fraude a mais ou a menos detectada muda o recall em ~2 pontos.

**Implementação**
- [src/validation.py](../src/validation.py): `DEV_FRACTION = 0.8` e `split_dev_holdout(X, y)`; `holdout_temporal` usa esse corte (o comportamento não mudou) e agora mede também `predict_seconds` (pontuar o holdout) e devolve o modelo treinado. `cv_stratified` avisa na docstring que recebe só o desenvolvimento.
- [src/train.py](../src/train.py) `evaluate_and_log`: o 5-fold roda em `X_dev, y_dev`. Novas métricas `predict_seconds` e `predict_ms_per_1k`; novos parâmetros `validation = dev-stratified-kfold-5 + holdout-temporal-20` e `n_dev_rows`; novo artefato `holdout_scores.csv` (rótulo e score de cada linha do holdout; o notebook da A6 usa para as curvas PR sobrepostas).
- [notebooks/02_experimentos.ipynb](../notebooks/02_experimentos.ipynb): seção 3 usa o desenvolvimento, textos das seções 3, 4 e 5 atualizados com os números novos, seção 6 vira a descrição do protocolo. Executado do zero (`nbconvert --execute`): sem erros. `val_01` e `val_02` saíram idênticas; `val_03` foi regenerada.

**Baseline reexecutado no protocolo novo** (run `3f7332491d774f18b8ca78fd2a63014f`, experimento `baseline`, versão 5 do modelo, sem alias; `champion` continua na versão 1)

| Estimativa | Antes (sessão 1) | Depois (protocolo novo) |
|---|---|---|
| Holdout temporal, PR-AUC | 0,7330 | **0,7330** (igual: o holdout já treinava só nos primeiros 80%) |
| Holdout: recall / precisão / F1 (limiar 0,5) | 0,9038 / 0,0221 / 0,0432 | iguais |
| 5-fold, PR-AUC de validação | 0,7148 ± 0,0657 (`train.csv` inteiro, 79–80 fraudes por fold) | **0,7200 ± 0,0299** (só desenvolvimento, 69–70 fraudes por fold) |
| 5-fold, PR-AUC de treino | 0,7147 ± 0,0163 | 0,7214 ± 0,0080 |
| 5-fold, ROC-AUC | 0,9708 ± 0,0090 | 0,9730 ± 0,0108 |
| 5-fold, recall / precisão / F1 | 0,8947 / 0,0608 / 0,1138 | 0,9021 / 0,0722 / 0,1336 |

Por fold (novo): PR-AUC 0,7224 · 0,6839 · 0,6987 · 0,7353 · 0,7598. Tempo de treino 0,31 s; pontuar as 45.401 linhas do holdout: 0,019 s.

**Leitura:** a média quase não mudou e o desvio caiu à metade. O trecho do holdout é o mais recente e tem menos fraudes (0,11% contra 0,19%); misturado aos folds, ele aumentava a variação entre eles. Com o protocolo novo, **diferenças de PR-AUC da ordem de 0,03 (um desvio) estão dentro do ruído** (antes, ~0,07).

**Outro número que mudou:** a projeção do custo do LOO (notebook 02, seção 5) ficou em 16,2 h nesta execução (0,26 s por treino); o texto agora diz "de 16 a 23 horas, conforme a execução".

**Arquivos alterados:** `src/validation.py`, `src/train.py`, `notebooks/02_experimentos.ipynb`, `reports/figs/val_03_baseline_estabilidade_folds.png`, `docs/CONTEXTO.md`.

---

## S2.3a — A6 parcial, código novo e interrupção (registro feito na retomada)

**Código novo em [src/train.py](../src/train.py)** (todos com `random_state=42` onde há aleatoriedade):

| Função / comando | O que faz | Estado |
|---|---|---|
| `candidate_models()` | Os modelos da A6 com hiperparâmetros padrão razoáveis (ver tabela abaixo) | Usado |
| `compare [nomes]` | `evaluate_and_log` de cada modelo no experimento `comparacao` (uma run por modelo; sem registrar versão nem mexer no `champion`) | Executado para 7 das 9 variantes |
| `tune [nomes]` | Optuna TPE (config da A6 como trial 0, via `enqueue_trial`) maximizando a PR-AUC média do 5-fold no desenvolvimento; no `random-forest`, também Random Search (mesmo espaço, mesmos folds, mesmo orçamento). Experimento `otimizacao` (uma run por estratégia e trials como runs-filhas), figuras `reports/figs/opt_*`, melhor config avaliada com `evaluate_and_log` e gravada em `src/best_params.json` | Testado (3 trials, MLflow temporário); execução real interrompida |
| `final` | Retreina o campeão de `src/best_params.json` em todo o `train.csv`, registra a versão, grava a tag `threshold` e move o `champion` | **Não executado** |
| `stream_test` | Teste único no `stream.csv` com o `champion` e o seu limiar; métricas `stream_*` na run do campeão | **Não executado** |
| `validation.choose_threshold` | Limiar que maximiza o F1 no holdout | Não usado ainda |
| `validation.PLOT_STYLE` | Estilo único das figuras (o mesmo dos notebooks), aplicado na importação | Usado |

**Modelos e tratamento do desbalanceamento**

| Run | Modelo (padrão) | Desbalanceamento |
|---|---|---|
| `logreg` | LogisticRegression(max_iter=1000) | `class_weight="balanced"` |
| `decision-tree` | DecisionTree(max_depth=10, min_samples_leaf=20) | `class_weight="balanced"` |
| `random-forest` | RandomForest(200 árvores, max_features="sqrt") | `class_weight="balanced"` |
| `adaboost` / `adaboost-smote` | AdaBoost(200 stumps, learning_rate=0.5) | nenhum / SMOTE(0,1) |
| `mlp` / `mlp-smote` | MLP(64, 32), alpha=1e-4, max_iter=50 | nenhum / SMOTE(0,1) |
| `knn-under` / `knn-under-smote` | KNN(k=15, weights="distance") | RandomUnderSampler(0,02) / + SMOTE(0,1) |

- **SMOTE(0,1)**: as fraudes sintéticas sobem até 10% das legítimas (1:10), só no treino de cada fold (passo do `imblearn.Pipeline`, depois da padronização, porque o SMOTE usa distâncias). Balancear até 1:1 criaria ~500 pontos sintéticos por fraude real (interpolação demais a partir de ~280 fraudes) e dobraria o custo de treino.
- **KNN, subamostra 0,02**: todas as fraudes do treino do fold + legítimas sorteadas na razão 1:50 (~14 mil linhas em vez de ~145 mil). O KNN compara cada transação com todo o treino na predição; com a subamostra, o custo cai ~10 vezes, e 50 legítimas por fraude ainda representam a variedade das legítimas. A validação mantém a proporção real.

**Probe de custo** (um treino no fold 1 do desenvolvimento; só tempos e a PR-AUC desse fold, sem tocar no holdout):

| Modelo | Treino (s) | Predição (s) | PR-AUC fold 1 |
|---|---|---|---|
| logreg | 0,3 | 0,02 | 0,7224 |
| decision-tree | 4,0 | 0,02 | 0,7581 |
| random-forest | 9,5 | 0,14 | 0,8499 |
| adaboost (sem tratamento) | **1.006,3** | 0,50 | 0,8108 |
| adaboost-smote | 137,2 | 0,50 | 0,8138 |
| mlp | 10,1 | 0,03 | 0,8515 |
| mlp-smote | 10,2 | 0,03 | 0,8501 |
| knn-under | 0,1 | 0,25 | 0,8138 |
| knn-under-smote | 0,1 | 0,26 | 0,7795 |

O AdaBoost sem tratamento foi ~7 vezes mais lento que com SMOTE, mesmo treinando em menos linhas. Hipótese: sem reamostragem, os pesos das ~227 mil legítimas fáceis encolhem a cada rodada até virar números subnormais, e a aritmética com subnormais é muito mais lenta. Como seriam ~2 h para as 7 execuções do protocolo, essa variante será substituída (S2.5).

**A6 parcial** (experimento `comparacao`; 5-fold no desenvolvimento; holdout no limiar 0,5)

| Run | 5-fold PR-AUC | Holdout PR-AUC | ROC-AUC | Recall | Precisão | F1 | Treino (s) | Predição do holdout (s) |
|---|---|---|---|---|---|---|---|---|
| random-forest | 0,8541 ± 0,0122 | **0,7589** | 0,9273 | 0,6731 | 0,9459 | 0,7865 | 15,0 | 0,087 |
| mlp-smote | 0,8496 ± 0,0118 | 0,7069 | 0,9299 | 0,7308 | 0,5846 | 0,6496 | 15,0 | 0,075 |
| mlp | 0,8431 ± 0,0191 | 0,7194 | 0,9120 | 0,7308 | 0,6032 | 0,6609 | 11,8 | 0,030 |
| knn-under | 0,7556 ± 0,0558 | 0,7161 | 0,9114 | 0,7308 | 0,6667 | 0,6972 | 0,2 | 1,209 |
| decision-tree | 0,7549 ± 0,0268 | 0,6915 | 0,8932 | 0,7885 | 0,0940 | 0,1680 | 5,3 | 0,018 |
| knn-under-smote | 0,7323 ± 0,0466 | 0,6047 | 0,9186 | 0,8077 | 0,2320 | 0,3605 | 1,0 | 1,657 |
| logreg (baseline) | 0,7200 ± 0,0299 | 0,7330 | 0,9742 | 0,9038 | 0,0221 | 0,0432 | 0,3 | 0,017 |
| adaboost-smote | interrompido | | | | | | | |
| adaboost | interrompido (será substituído) | | | | | | | |

Tempos medidos com 3 a 4 processos disputando a CPU; servem para ordem de grandeza.

**Interrupção (queda de energia)**
- Antes da queda, os 4 processos em segundo plano foram parados a pedido do Irvin (`Stop-Process`): `tune random-forest` (TPE + Random Search), `tune mlp-smote`, `compare adaboost-smote` e `compare adaboost`.
- **Progresso perdido:** os estudos do Optuna ficavam em memória e só gravavam no MLflow no fim, e o `compare` só cria a run depois de treinar tudo. Por isso nenhum trial nem fold desses 4 processos foi salvo. Lição para a retomada: rodar o `tune` com saída sem buffer e salvar os trials à medida que terminam.
- **MLflow na retomada:** nenhuma run RUNNING ou FAILED. As 7 runs de `comparacao` estão FINISHED, e o experimento `otimizacao` não chegou a ser criado. **Nada precisou ser apagado.**
- O script que monta o notebook 03 foi movido do scratchpad para [scripts/build_nb03.py](../scripts/build_nb03.py). O `docs/PAUSA.md` foi o resumo de emergência; este registro o substitui.

**Arquivos alterados:** `src/train.py`, `src/validation.py`, `scripts/build_nb03.py` (novo), `docs/CONTEXTO.md`, `docs/PAUSA.md` (novo; pode ser descartado).

---

## Nota para a frente B — simulador reiniciado não gera dados novos (para repassar)

**Problema:** o simulador numera as mensagens como `tx-<volta>-<linha>` e recomeça em `tx-0-0` sempre que o contêiner reinicia (rebuild, `make up`, queda). O subscriber grava com `INSERT ... ON CONFLICT (tx_id) DO NOTHING`, então todos os ids já existentes no banco são descartados em silêncio. Com 0,5 s por transação e ~46 mil linhas já gravadas, **o dashboard fica cerca de 6 horas sem nenhuma transação nova** depois de um reinício. Isso afeta diretamente a demo. Confirmado na sessão 2 com `mosquitto_sub` (o simulador publica `tx-0-294`, que já está no banco).

**Correção sugerida (uma linha em `embedded/simulator.py`):** incluir no `tx_id` um identificador da execução, por exemplo `run = int(time.time())` no início e `tx_id = f"tx-{run}-{lap}-{i}"`. O contrato MQTT não muda (`tx_id` continua sendo uma string única).
**Paliativo sem mudar código:** `make reset` antes da demo (apaga o banco).

---

## S2.4 — Por que o 5-fold estratificado e o holdout discordam (Etapa 1)

**Pergunta:** no baseline, o 5-fold e o holdout são parecidos (0,720 e 0,733), mas na RF e no MLP o 5-fold fica perto de 0,85 e o holdout cai para 0,71–0,76. **Hipótese (Irvin):** o k-fold embaralhado mistura transações de momentos diferentes; modelos flexíveis aprendem padrões de fraudes correlacionadas no tempo (rajadas, como na Fig. 3 da EDA), que aparecem dos dois lados do corte. O holdout temporal impede isso.

**O que foi feito**
- [src/validation.py](../src/validation.py): `cv_temporal` (`TimeSeriesSplit`, 5 splits no desenvolvimento, janela crescente: treino de 30 mil a 151 mil linhas, validação de 30.267 linhas no bloco seguinte, com 66, 48, 33, 65 e 38 fraudes) e `bootstrap_ci` (IC 95% da PR-AUC por bootstrap das linhas do holdout, 1.000 reamostragens, `random_state=42`). O código de CV foi fatorado em `_cv`, compartilhado pelas duas validações.
- [src/train.py](../src/train.py): o `evaluate_and_log` agora registra também `cv_temporal_*` (média e desvio por métrica), `cv_temporal_folds.csv`, `pr_auc_ci_low` e `pr_auc_ci_high`. O novo comando `diagnose` acrescentou essas métricas **às 7 runs já existentes** de `comparacao` (mesma configuração; nada foi retreinado nem duplicado no estratificado).

**Resultados** (PR-AUC; holdout com IC 95%; treino = média da PR-AUC no treino de cada fold)

| Modelo | Stratified 5-fold | Temporal 5-fold | Holdout [IC 95%] | Estrat. − temporal | Treino (estrat. / temporal) |
|---|---|---|---|---|---|
| random-forest | 0,8541 ± 0,0122 | 0,7741 ± 0,1225 | **0,7589** [0,619; 0,870] | +0,080 | 1,000 / 1,000 |
| logreg (baseline) | 0,7200 ± 0,0299 | 0,7160 ± 0,1426 | 0,7330 [0,601; 0,839] | **+0,004** | 0,721 / 0,686 |
| mlp | 0,8431 ± 0,0191 | 0,7546 ± 0,1141 | 0,7194 [0,577; 0,833] | +0,089 | 0,993 / 0,999 |
| knn-under | 0,7556 ± 0,0558 | 0,7091 ± 0,1149 | 0,7161 [0,575; 0,836] | +0,047 | 0,923 / 0,962 |
| mlp-smote | 0,8496 ± 0,0118 | 0,7313 ± 0,0723 | 0,7069 [0,565; 0,821] | +0,118 | 1,000 / 1,000 |
| decision-tree | 0,7549 ± 0,0268 | 0,6592 ± 0,1033 | 0,6915 [0,549; 0,825] | +0,096 | 0,878 / 0,867 |
| knn-under-smote | 0,7323 ± 0,0466 | 0,6450 ± 0,1405 | 0,6047 [0,449; 0,746] | +0,087 | 0,890 / 0,929 |

PR-AUC por split temporal: logreg 0,706 · 0,708 · 0,492 · 0,807 · 0,867; RF 0,824 · 0,883 · 0,564 · 0,808 · 0,791; MLP 0,845 · 0,846 · 0,590 · 0,682 · 0,810. O split 3 (`Time` 63.322–76.231 s, 33 fraudes) é o mais difícil para todos.

**Diferenças pareadas no holdout** (as mesmas 1.000 reamostragens para os dois modelos; IC 95% da diferença)

| Comparação | Diferença | IC 95% | P(1º > 2º) |
|---|---|---|---|
| RF − logreg | +0,027 | [−0,028; +0,085] | 0,83 |
| RF − mlp | +0,040 | [−0,022; +0,105] | 0,90 |
| RF − knn-under | +0,043 | [−0,007; +0,097] | 0,95 |
| RF − mlp-smote | +0,053 | [−0,011; +0,126] | 0,94 |
| RF − decision-tree | +0,066 | [+0,007; +0,131] | 0,98 |
| RF − knn-under-smote | +0,153 | [+0,061; +0,245] | 1,00 |
| logreg − knn-under-smote | +0,127 | [+0,040; +0,224] | 1,00 |

**Conclusões (texto para a análise crítica e o relatório)**
1. **A hipótese se confirma.** Em todos os modelos flexíveis, o 5-fold estratificado supera o temporal em +0,05 a +0,12, e o temporal fica perto do holdout (diferença ≤ 0,04 em todos, exceto a árvore). No modelo linear, as três estimativas coincidem (0,720 / 0,716 / 0,733). O viés está no **embaralhamento**, e ele cresce com a capacidade do modelo.
2. **Mecanismo:** RF e MLP chegam a PR-AUC de treino ~1,0, ou seja, memorizam as fraudes do treino. Fraudes vêm em rajadas próximas no tempo (EDA, Fig. 3); no k-fold embaralhado, fraudes "irmãs" caem no treino e na validação, e a memorização vira acerto. No corte temporal, as fraudes da validação são de outro momento, e a vantagem some. A logística, que não consegue memorizar (treino 0,72), não ganha nada com o embaralhamento.
3. **O 5-fold estratificado superestima o desempenho futuro dos modelos flexíveis em ~0,08–0,12 de PR-AUC.** Ele continua útil para medir **estabilidade** (desvio baixo, ~0,01–0,03), mas não para estimar o desempenho em produção nem para ordenar modelos de capacidades diferentes.
4. **O holdout tem pouco poder estatístico:** com 52 fraudes, o IC 95% de cada modelo tem ~±0,13 de largura. A RF tem o melhor holdout, mas **a vantagem sobre a logística, o MLP e o KNN não é significativa** (os ICs pareados incluem o zero). Só são significativamente piores que a RF: Árvore, KNN com SMOTE e AdaBoost balanceado.
5. **A CV temporal também é ruidosa** (desvio de ~0,07–0,14), porque cada bloco tem só 33–66 fraudes e os blocos diferem entre si (o split 3 é difícil para todos). Nenhuma estimativa isolada é confiável; o relatório deve mostrar as três e o IC.

**Consequência para a A7 (decisão):** a otimização **não chegou a ser concluída** (nenhum trial foi salvo antes da queda), então não há configuração superajustada a refazer. Ao reiniciar, o objetivo da otimização passa a ser a **PR-AUC média da CV temporal no desenvolvimento** (`TUNE_CV = "temporal"` em `src/train.py`), e não a do estratificado. Motivo: o estratificado premia justamente a memorização (por exemplo, folhas de 1 amostra na RF), que não se sustenta no futuro. Custo: o objetivo temporal é mais ruidoso, e o TPE pode perseguir ruído; por isso a melhor configuração é sempre conferida no holdout contra a padrão, e o ganho reportado é o do holdout.

**Arquivos alterados:** `src/validation.py`, `src/train.py`, `docs/CONTEXTO.md`.

---

## S2.5 — A6 completa: comparação de modelos

**AdaBoost (Etapa 2.1)**
- A variante "sem tratamento" foi **substituída** por `adaboost-balanced`: AdaBoost com estimador base `DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)`, 200 estimadores, `learning_rate=0.5`.
- **Ela degenera:** o boosting para com **3 estimadores**. O stump balanceado otimiza o erro ponderado por classe e marca fraude em excesso. Medido com os pesos do AdaBoost, o erro sobe de 0,027 → 0,096 → 0,318 e chega a 1,0 na 4ª rodada. O SAMME exige erro < 0,5 e encerra. PR-AUC: 0,276 (estratificado), 0,390 (temporal), 0,096 (holdout). Conclusão para o relatório: **`class_weight` no estimador base conflita com a reponderação do AdaBoost**; para AdaBoost, o desbalanceamento deve ser tratado por reamostragem (SMOTE) ou pelos pesos iniciais do `fit`.
- `adaboost-smote` foi reexecutado do zero (tinha morrido na queda). Treino de ~199 s no desenvolvimento com a CPU disputada (137 s isolado no probe): **nenhum treino passou de 10 min**, então a subamostra não foi necessária. O processo levou ~55 min de relógio porque recebeu menos de um núcleo (a otimização da RF ocupava ~15 dos 20).

**Tabela final da A6** (experimento `comparacao`; holdout no limiar 0,5; tempos medidos com processos concorrentes, valem como ordem de grandeza)

| Run | Estratificado 5-fold | Temporal 5-fold | Holdout [IC 95%] | Recall | Precisão | F1 | ROC-AUC | Treino (s) | Predição (ms / mil tx) |
|---|---|---|---|---|---|---|---|---|---|
| **random-forest** | 0,8541 ± 0,0122 | **0,7741 ± 0,1225** | **0,7589** [0,619; 0,870] | 0,6731 | 0,9459 | 0,7865 | 0,9273 | 15,0 | 1,9 |
| logreg (baseline) | 0,7200 ± 0,0299 | 0,7160 ± 0,1426 | 0,7330 [0,601; 0,839] | 0,9038 | 0,0221 | 0,0432 | 0,9742 | 0,3 | 0,4 |
| mlp | 0,8431 ± 0,0191 | 0,7546 ± 0,1141 | 0,7194 [0,577; 0,833] | 0,7308 | 0,6032 | 0,6609 | 0,9120 | 11,8 | 0,7 |
| knn-under | 0,7556 ± 0,0558 | 0,7091 ± 0,1149 | 0,7161 [0,575; 0,836] | 0,7308 | 0,6667 | 0,6972 | 0,9114 | 0,2 | 26,6 |
| mlp-smote | 0,8496 ± 0,0118 | 0,7313 ± 0,0723 | 0,7069 [0,565; 0,821] | 0,7308 | 0,5846 | 0,6496 | 0,9299 | 15,0 | 1,7 |
| adaboost-smote | 0,8014 ± 0,0188 | 0,7714 ± 0,0971 | 0,6962 [0,572; 0,808] | 0,7885 | 0,2971 | 0,4316 | 0,9741 | 198,6 | 16,6 |
| decision-tree | 0,7549 ± 0,0268 | 0,6592 ± 0,1033 | 0,6915 [0,549; 0,825] | 0,7885 | 0,0940 | 0,1680 | 0,8932 | 5,3 | 0,4 |
| knn-under-smote | 0,7323 ± 0,0466 | 0,6450 ± 0,1405 | 0,6047 [0,449; 0,746] | 0,8077 | 0,2320 | 0,3605 | 0,9186 | 1,0 | 36,5 |
| adaboost-balanced | 0,2757 ± 0,0648 | 0,3903 ± 0,1997 | 0,0960 [0,061; 0,135] | 0,8846 | 0,0178 | 0,0349 | 0,9580 | 3,4 | 0,5 |

Matrizes de confusão no holdout (limiar 0,5; TN, FP, FN, TP): logreg 43.273 / 2.076 / 5 / 47 · RF 45.347 / 2 / 17 / 35 · MLP 45.324 / 25 / 14 / 38 · KNN 45.330 / 19 / 14 / 38 · AdaBoost com SMOTE 45.252 / 97 / 11 / 41 · Árvore 44.954 / 395 / 11 / 41.

**Escolha da variante de desbalanceamento por modelo** (pelo 5-fold **temporal** no desenvolvimento, nunca pelo holdout)

| Modelo | Escolha | Evidência |
|---|---|---|
| Logística, Árvore, RF | `class_weight="balanced"` | o modelo aceita; regra do projeto |
| AdaBoost | **SMOTE(0,1)** | vence o stump balanceado nos 5 splits temporais (0,771 × 0,390); o "sem tratamento" foi descartado por custo (~1.000 s por treino) |
| MLP | **sem SMOTE** | pelo estratificado o SMOTE parecia ajudar (0,850 × 0,843), mas pelo temporal atrapalha (0,731 × 0,755; vence só 2 de 5 splits). As fraudes sintéticas interpolam fraudes do mesmo período |
| KNN | **só a subamostra 1:50** | o SMOTE piora nos 5 splits temporais (0,645 × 0,709) |

**Notebook e figuras**
- [notebooks/03_comparacao.ipynb](../notebooks/03_comparacao.ipynb), gerado por [scripts/build_nb03.py](../scripts/build_nb03.py) com os textos de [scripts/nb03_texto.json](../scripts/nb03_texto.json): `python scripts/build_nb03.py . scripts/nb03_texto.json`. Só **lê** o MLflow (stack de pé). Executado do zero com kernel limpo (`nbconvert --execute`): **0 erros**, 34 células, interpretação abaixo de cada gráfico, salvo sem saídas.
- Figuras em `reports/figs/`:

| Arquivo | Conteúdo |
|---|---|
| `cmp_01_pr_auc.png` | Barras: média ± desvio da PR-AUC no 5-fold estratificado, com o holdout marcado |
| `cmp_02_folds.png` | PR-AUC por fold, estratificado × temporal (melhor variante de cada família) |
| `cmp_03_curvas_pr.png` | Curvas PR sobrepostas no holdout (uma por família) |
| `cmp_04_confusao.png` | Matrizes de confusão no holdout, limiar 0,5 |
| `cmp_05_treino_validacao.png` | Treino × validação estratificada × validação temporal (overfitting/underfitting) |
| `cmp_06_custo.png` | Tempo de treino × PR-AUC no holdout |
| `cmp_07_estratificado_temporal_holdout.png` | **Nova:** as três estimativas por modelo, com desvio e IC 95% |

**Análise crítica (resumo; texto completo no notebook, seção 10)**
1. A **Random Forest** lidera em todas as estimativas (holdout 0,759; temporal 0,774; estratificado 0,854) e fica entre os 4 primeiros em todos os splits temporais (1º em dois). A vantagem sobre logística, MLP e KNN **não é significativa** no holdout (52 fraudes).
2. O baseline é competitivo no futuro (holdout 0,733; 1º no split temporal mais recente, 0,867), é o mais barato e não tem overfitting; perde na faixa útil da curva PR (precisão ~0,8 aos 60% de recall, contra ~0,95 da RF).
3. Todas as curvas PR desabam entre 75% e 80% de recall: 10 a 14 das 52 fraudes recebem score < 0,1 em todos os modelos. É o teto de recall com precisão útil nesta base.
4. No limiar 0,5, a RF é conservadora (2 falsos alarmes, recall 0,67) e a logística agressiva (2.076 falsos alarmes, recall 0,90). O limiar será escolhido na A8.

**Finalistas da A7:** **Random Forest** e **MLP**. O AdaBoost com SMOTE (2º no temporal, 0,771) ficou de fora por custo (~13 vezes o treino da RF; cada trial seriam ~5 treinos de 1,5 a 3 min) e pelo holdout mais baixo entre os fortes (0,696). A logística não entra (só teria a regularização para ajustar).

**Arquivos alterados:** `src/train.py` (`adaboost-balanced` no lugar de `adaboost`), `scripts/build_nb03.py`, `scripts/nb03_texto.json` (novo), `notebooks/03_comparacao.ipynb` (novo), `reports/figs/cmp_01` a `cmp_07` (novos), `docs/CONTEXTO.md`.

---

## S2.6 — A7: otimização de hiperparâmetros

**Estratégia e justificativa**
- **Optuna TPE** nos dois finalistas (RF e MLP): o TPE usa os trials anteriores para propor os próximos, o que rende mais por trial quando cada trial é caro (5 treinos). **Random Search** (o `RandomSampler` do Optuna) também na RF, com o **mesmo espaço, os mesmos splits, o mesmo objetivo e o mesmo orçamento** (30 trials), como base de comparação: só o amostrador muda.
- **Objetivo:** PR-AUC média do **5-fold temporal no desenvolvimento** (S2.4: o estratificado premia memorização). O holdout não é usado na busca; ele só confere a melhor configuração no fim.
- **O 1º trial de cada estudo é a configuração padrão da A6** (`enqueue_trial`): o ganho é medido nos mesmos splits. Conferido: o trial 0 reproduziu exatamente o temporal da A6 (RF 0,7741; MLP com SMOTE 0,7313).
- **Detalhe da comparação:** o TPE começa com 10 trials aleatórios, e com a mesma semente (42) eles coincidem com os do Random Search. As duas buscas só divergem a partir do trial 10; a diferença de resultado vem do modelo do TPE.
- **Robustez:** os trials ficam num SQLite do Optuna (`data/optuna.db`, fora do git). Se o processo cair, o mesmo comando retoma o estudo (lição da queda de energia).
- **30 trials por estratégia** foi o limite pelo tempo disponível: ~1 a 2 min por trial na RF e ~1,5 min no MLP com a CPU disputada. Duração: TPE da RF 56 min, Random Search da RF 32 min, TPE do MLP 49 min.

**Espaços de busca**

| Modelo | Hiperparâmetro | Espaço | Padrão da A6 |
|---|---|---|---|
| RF | `n_estimators` | 100–400, passo 50 | 200 |
| RF | `max_depth` | {None, 8, 12, 16, 24} | None |
| RF | `min_samples_leaf` | 1–20 (escala log) | 1 |
| RF | `max_features` | {"sqrt", 0,3, 0,5} | "sqrt" |
| RF | `class_weight` | {"balanced", "balanced_subsample", None} | "balanced" |
| MLP | proporção do SMOTE | 0,01–0,5 (log; 0,01 ≈ quase sem SMOTE) | 0,1 |
| MLP | `hidden_layer_sizes` | {(32), (64, 32), (128, 64), (128, 64, 32)} | (64, 32) |
| MLP | `alpha` | 1e-6–1e-2 (log) | 1e-4 |
| MLP | `learning_rate_init` | 1e-4–1e-2 (log) | 1e-3 |
| MLP | `batch_size` | {100, 200, 500} | 200 |

**Resultados**

| Modelo | Estratégia | Temporal: padrão → melhor | Melhor configuração | Holdout: padrão → otimizado |
|---|---|---|---|---|
| **RF** | **Optuna TPE** | 0,7741 → **0,8009** (+0,027; trial 22) | 400 árvores, `max_depth=24`, `min_samples_leaf=2`, `max_features=0.3`, **`class_weight=None`** | 0,7589 → **0,7601** (+0,001) |
| RF | Random Search | 0,7741 → 0,7991 (+0,025; trial 22) | 400 árvores, `max_depth=12`, `min_samples_leaf=10`, `max_features="sqrt"`, `class_weight="balanced_subsample"` | (não avaliada: a melhor das duas é a do TPE) |
| **MLP** | **Optuna TPE** | 0,7313 → **0,7587** (+0,027; trial 22) | (64, 32), `alpha=3,1e-5`, `learning_rate_init=1,4e-4`, `batch_size=500`, SMOTE 0,017 | 0,7069 → **0,7219** (+0,015) |

RF otimizada (run `16177e0b…`, experimento `otimizacao`): holdout 0,7601 [IC 0,626; 0,873], temporal 0,8009 ± 0,1048, estratificado 0,8619 ± 0,0077, ROC-AUC 0,9515 (era 0,9273), recall 0,692 e precisão 0,947 no limiar 0,5, treino de 82 s, 2,8 ms por mil transações.
MLP otimizado (run `52fb2360…`): holdout 0,7219 [0,580; 0,833], temporal 0,7587 ± 0,0974, estratificado 0,8498 ± 0,0231, recall 0,731 e precisão 0,704 no limiar 0,5.

**Leituras**
1. **TPE × Random Search (RF):** o melhor valor final é praticamente igual (0,8009 × 0,7991), mas o TPE **explora melhor a região boa**. Nos trials 10–29, quando os dois já divergiram, a média do TPE foi 0,794 e a do Random Search 0,781; o TPE teve 14 trials ≥ 0,795, e o Random Search, 4. Com orçamento pequeno, os dois chegam perto do topo; o TPE chega lá com mais consistência.
2. **O que importa na RF é o `class_weight`** (importância fANOVA de 0,97). Sem pesos de classe, a média dos trials é 0,797 (19 trials); com `balanced`, 0,773, e com `balanced_subsample`, 0,777. **O `class_weight="balanced"` da A6 atrapalhava a RF:** com ~500 vezes mais peso que uma legítima (razão legítimas/fraudes no treino), cada fraude vira uma região inteira da árvore, e o modelo memoriza fraudes específicas que não se repetem no futuro. O TPE descobriu isso e concentrou 19 dos 30 trials em `None`. Os outros hiperparâmetros quase não importam (≤ 0,02).
3. **No MLP, o que importa é a taxa de aprendizado** (importância de 0,67): os trials com `learning_rate_init` < 3e-4 têm média de 0,740, contra 0,728 acima disso. Treinar mais devagar reduz a memorização (PR-AUC de treino no temporal de ~1,0 para 0,977). A proporção de SMOTE escolhida (0,017) é praticamente "sem SMOTE", o que confirma a A6.
4. **O ganho no holdout é pequeno e está dentro do ruído:** RF +0,001, MLP +0,015, contra ganhos de +0,027 no temporal. Parte do ganho na CV é otimismo de seleção (o melhor de 30 trials num objetivo com desvio de ~0,10) e parte não se transfere para o período do holdout. **A otimização melhorou a calibração dos scores** (ROC-AUC da RF de 0,927 para 0,951), **não o ranking no topo**, que é o que a PR-AUC mede.
5. **Figuras:** `reports/figs/opt_random-forest_historico.png` (TPE × Random Search, melhor até cada trial), `opt_random-forest_importancia.png`, `opt_mlp-smote_historico.png` e `opt_mlp-smote_importancia.png`. Parâmetros em [src/best_params.json](../src/best_params.json).

**Arquivos alterados:** `src/train.py` (`tune` com SQLite, `TUNE_CV`, `_completed`, `_default_value`), `src/best_params.json` (novo), `reports/figs/opt_*.png` (4, novos), `docs/CONTEXTO.md`.

---

## S2.7 — A8: campeão, limiar, teste final e promoção

**Candidatos** (holdout com IC 95%; temporal no desenvolvimento)

| Run | Holdout [IC 95%] | Temporal | Treino (s) |
|---|---|---|---|
| **random-forest-tuned** | **0,7601** [0,626; 0,873] | **0,8009 ± 0,1048** | 82 |
| random-forest (A6) | 0,7589 [0,619; 0,870] | 0,7741 ± 0,1225 | 15 |
| logreg (baseline) | 0,7330 [0,601; 0,839] | 0,7160 ± 0,1426 | 0,3 |
| mlp-smote-tuned | 0,7219 [0,580; 0,833] | 0,7587 ± 0,0974 | 39 |
| mlp (A6) | 0,7194 [0,577; 0,833] | 0,7546 ± 0,1141 | 12 |

**Campeão: Random Forest otimizada** (`random-forest-tuned`: 400 árvores, `max_depth=24`, `min_samples_leaf=2`, `max_features=0.3`, `class_weight=None`).
Justificativa: (1) maior PR-AUC no holdout (0,760) e (2) maior PR-AUC no temporal (0,801), ou seja, é a melhor nas duas estimativas de desempenho futuro; (3) foi a mais consistente entre os blocos temporais (S2.5); (4) na faixa útil da curva PR, mantém precisão de ~0,95 até ~73% de recall. **Ressalva honesta:** pelos ICs, a vantagem sobre a logística (+0,027) e o MLP não é estatisticamente significativa com 52 fraudes; a escolha se apoia na **consistência** (holdout, temporal e estratificado apontam o mesmo modelo), não num único número. O custo é aceitável: 82 s de treino e 2,8 ms por mil transações.

**Limiar**
- **Critério:** o limiar que **maximiza o F1 da classe fraude** na curva PR do holdout (`validation.choose_threshold`), calculado com os scores do modelo treinado só no desenvolvimento. Motivo: equilibra fraudes perdidas e falsos alarmes sem depender de um custo arbitrário. Alternativas registradas para o relatório: recall ≥ 0,80 exigiria limiar 0,039, com precisão de 0,31; recall ≥ 0,75, limiar 0,216, com precisão de 0,81.
- Limiar exato 0,46851786 (F1 0,826). **Valor publicado: `0.468518`** (6 casas; tag `threshold` da versão do modelo e `FRAUD_THRESHOLD`). Com o arredondamento, uma fraude com score exatamente na fronteira fica de fora; **todas as métricas abaixo usam o valor publicado**.
- **Matriz de confusão no holdout** (52 fraudes, 45.401 linhas), em `reports/figs/a8_02_confusao_antes_depois.png`:

| | TN | FP | FN | TP | Recall | Precisão | F1 |
|---|---|---|---|---|---|---|---|
| Antes (limiar 0,5) | 45.347 | 2 | 16 | 36 | 0,692 | 0,947 | 0,800 |
| Depois (limiar 0,468518) | 45.347 | 2 | 15 | 37 | 0,712 | 0,949 | 0,813 |

  A RF otimizada já tinha scores bem separados; o limiar novo recupera uma fraude sem nenhum falso alarme a mais. A curva (`a8_01_limiar.png`) mostra um platô: entre ~0,22 e ~0,47, o recall fica em ~0,73 e a precisão sobe de 0,81 para 0,95; abaixo de ~0,2, o recall só sobe com perda grande de precisão.

**Treino final e promoção** (`python -m src.train final`)
- Retreina a RF com os parâmetros de `src/best_params.json` em **todo o `train.csv`** (desenvolvimento + holdout), registra no experimento `campeao` (run `04c8c42009eb4a08b260d7936b09fc4f`), cria a **versão 6** do modelo `fraudazero`, grava a tag `threshold=0.468518` e move **`champion` → versão 6**. Determinístico (`random_state=42`): o holdout reproduziu a PR-AUC da A7 (0,7601).

**Teste final no `stream.csv`** (`python -m src.train stream_test`, **executado uma única vez**, depois de tudo escolhido; métricas `stream_*` na run do campeão; figuras `a8_stream_confusion_matrix.png` e `a8_stream_pr_curve.png`)

| Métrica (stream: 56.962 transações, 75 fraudes) | Valor |
|---|---|
| **PR-AUC** | **0,7892** [IC 95% bootstrap: 0,696; 0,872] |
| ROC-AUC | 0,9737 |
| Recall | 0,7067 (53 de 75) |
| Precisão | 0,9636 |
| F1 | 0,8154 |
| Matriz (TN, FP, FN, TP) | 56.885 / **2** / 22 / 53 |

Leitura: o número de generalização (0,789) ficou **acima** do holdout (0,760) e dentro do IC dele, sem sinal de degradação no período mais recente. A prevalência do stream (0,13%) fica entre a do holdout (0,11%) e a do desenvolvimento (0,19%). Recall e precisão repetem o padrão do holdout: ~71% das fraudes detectadas, com 2 falsos alarmes em ~57 mil transações. No stream, os limiares 0,5 e 0,468518 dão exatamente o mesmo resultado (nenhum score entre os dois).

**Produção sem mudar o contrato**
- **API:** `POST /reload` carregou a versão 6. `/health` → `{"status":"ok","model_loaded":true}`. `/predict` com a 1ª fraude do stream → `{"score":0.9333,"is_fraud":true,"threshold":0.468518}`; com a 1ª legítima → `score` 2,0e-6, `is_fraud: false`. Os scores da API são idênticos aos do teste offline (0,93329 e 2,03e-6), e a API confirma `champion` = versão 6.
- **Limiar na API:** aplicado pela variável `FRAUD_THRESHOLD`, que o contrato já previa. Para a verificação, o contêiner `api` foi recriado com `FRAUD_THRESHOLD=0.468518` no ambiente do shell; **nenhum arquivo do repositório mudou**. **Pendência (frente B):** fixar o valor no repositório trocando o padrão em `docker-compose.yml` (`FRAUD_THRESHOLD: ${FRAUD_THRESHOLD:-0.468518}`) e em `.env.example`. Sem isso, um `make up` do zero volta para 0,5. No stream, isso não muda nenhuma previsão, mas o `threshold` devolvido pela API e o slider do dashboard ficariam com o valor errado.
- **A DAG não sobrescreve o campeão:** disparada depois da promoção, terminou verde nas 4 tarefas. O log da tarefa `baseline` diz "alias champion mantido na versão 6" (o baseline registra a versão 7 sem alias), e a `reload_api` recarregou a versão 6.
- Smoke test: 2 passed. Ruff (`src`, `scripts`, `notebooks`, `dags`, `tests`): limpo.

**Arquivos alterados:** `scripts/a8_campeao.py` (novo: candidatos, limiar e figuras), `src/best_params.json` (`champion`, `threshold`), `src/train.py` (`final`, `stream_test`), `src/validation.py` (`choose_threshold`), `reports/figs/a8_01_limiar.png`, `a8_02_confusao_antes_depois.png`, `a8_stream_confusion_matrix.png`, `a8_stream_pr_curve.png` (novos), `.gitignore` (`data/optuna.db`), `docs/CONTEXTO.md`.


---

## S2.8 — Verificações antes do commit (sem retreino)

Feitas só com artefatos que já existem no MLflow e no disco. Nenhum modelo foi treinado.

### 1. Com que dados o campeão foi treinado, e onde o limiar foi escolhido

- **Versão 6 (em produção):** treinada em **desenvolvimento + holdout**, ou seja, todo o `train.csv` sem as cópias exatas: **227.005 linhas** (parâmetro `n_train_rows` da run `04c8c420…`; em `evaluate_and_log`, `final = clone(model).fit(X, y)`).
- **Limiar (0,468518):** escolhido nos scores do holdout de **outro modelo**, com os mesmos hiperparâmetros, treinado **só no desenvolvimento** (181.604 linhas; `holdout_scores.csv` da run `random-forest-tuned`, `16177e0b…`). O `holdout_scores.csv` da run do campeão é idêntico (conferido), porque também vem do modelo de desenvolvimento.
- **Por que são modelos diferentes:** o holdout não pode escolher o limiar e ao mesmo tempo estar no treino do modelo que o usa (o score seria in-sample). Retreinar com os 20% mais recentes depois de tudo escolhido é a prática padrão: o modelo de produção aprende com o período mais próximo do futuro.

**Por que o limiar continua válido no modelo retreinado**
1. **Mesma receita, determinística:** os mesmos hiperparâmetros, `random_state=42` e 25% a mais de dados. Na RF, o score é a fração de árvores (folhas com ≥ 2 amostras) que vota fraude, e a escala dessa fração não muda com o tamanho do treino.
2. **As distribuições de score são equivalentes.** Modelo de desenvolvimento no holdout × versão 6 no stream (as duas são previsões fora da amostra):

| | Dev no holdout | v6 no stream |
|---|---|---|
| Legítimas: percentil 99,9 do score | 0,064 | 0,050 |
| Legítimas com score ≥ limiar | 0,0044% (2) | 0,0035% (2) |
| Fraudes com score ≥ limiar | 71,2% | 70,7% |
| Fraudes com score < 0,1 | 21,2% | 21,3% |
| Mediana do score das fraudes | 0,668 | 0,830 |

   As legítimas continuam concentradas perto de 0, e a fração de fraudes acima do limiar é praticamente a mesma. A v6 dá scores um pouco **mais altos** às fraudes que detecta (mediana 0,83), o que as afasta do limiar.
3. **O limiar está num platô, não numa fronteira sensível:** no holdout, recall e precisão quase não mudam entre ~0,22 e ~0,5 (`a8_01_limiar.png`). No stream, os limiares 0,468518 e 0,5 dão o mesmo resultado (nenhum score entre os dois), e só 12 das 56.962 transações têm score entre 0,3 e 0,6.
4. **Confirmação empírica:** no teste único do stream, a v6 com esse limiar repetiu o desempenho do holdout (recall 0,707 × 0,712; precisão 0,964 × 0,949; 2 falsos alarmes nos dois).

**Ressalva:** a comparação acima mistura dois efeitos, o modelo e o período (holdout × stream). Para isolar o efeito do retreino, seria preciso pontuar o stream também com o modelo de desenvolvimento. Esse modelo não foi salvo (o MLflow guarda só o modelo final), então isso exigiria retreiná-lo (~1–2 min, 400 árvores em 181 mil linhas). Não foi feito, por instrução.

### 2. Experimentos do MLflow

| Experimento | Runs | O que contém | Parâmetros, métricas, artefatos, modelo |
|---|---|---|---|
| `baseline` | 7 | Regressão logística (`logreg-baseline`). Versões do protocolo: as 4 mais antigas são da sessão 1 (k-fold no `train.csv` inteiro, 33 métricas); a `3f733249` é do protocolo novo (35 métricas); as 2 mais recentes vieram da DAG depois da validação temporal (59 métricas) | Todas com parâmetros, métricas, artefatos (matriz de confusão, curva PR, `cv_folds.csv`; as novas também `holdout_scores.csv` e `cv_temporal_folds.csv`) e **modelo salvo** |
| `comparacao` | 9 | Uma run por modelo da A6 | Todas com 7–10 parâmetros, 59 métricas, 5 artefatos e **modelo salvo** |
| `otimizacao` | 95 = 5 de topo + 90 trials | `random-forest-optuna-tpe`, `random-forest-random-search`, `mlp-smote-optuna-tpe` (parâmetros da busca, melhor valor, ganho, `trials.csv`; uma run-filha por trial com parâmetros e métricas) e `random-forest-tuned`, `mlp-smote-tuned` (a melhor configuração no protocolo completo) | As runs `*-tuned` têm parâmetros, 59 métricas, 5 artefatos e **modelo salvo**. As runs de estudo e os trials **não têm modelo** (não faz sentido salvar 90 modelos); o modelo da otimização é o das runs `*-tuned` |
| `campeao` | 1 | `random-forest-champion`: o `final` da A8 + o teste único no stream | Parâmetros, 69 métricas (incluindo `stream_*`), 8 artefatos e **modelo salvo** = versão 6 registrada |
| `Default` | 0 | Experimento padrão do MLflow, vazio | — |

Os três experimentos exigidos (baseline, comparação e otimização) existem separados, com parâmetros, métricas, artefatos e modelo salvo. Modelo registrado `fraudazero`: versões 1 a 8. **`champion` = versão 6.** As versões 7 e 8 vieram de execuções da DAG (baseline) depois da promoção e não moveram o alias.

### 3. Segundo ponto de operação no holdout: recall ≥ 0,80 (só registro; produção continua em 0,468518)

**Interpretação:** "o menor limiar com recall ≥ 0,80", tomado ao pé da letra, seria o limiar 0 (tudo vira fraude: recall 1,0 e precisão 0,0011). O ponto útil é o **maior limiar que ainda garante recall ≥ 0,80**, ou seja, o ponto da curva com recall ≥ 0,80 e a melhor precisão possível. É esse que está registrado.
Mesmo modelo e mesmos scores da escolha do limiar oficial (modelo de desenvolvimento, holdout com 52 fraudes e 45.401 linhas).

| Ponto de operação | Limiar | TN | FP | FN | TP | Recall | Precisão | F1 |
|---|---|---|---|---|---|---|---|---|
| **Produção (F1 máximo)** | 0,468518 | 45.347 | 2 | 15 | 37 | 0,7115 | 0,9487 | 0,8132 |
| **Recall ≥ 0,80** | **0,038875** | 45.254 | **95** | 10 | **42** | **0,8077** | **0,3066** | **0,4444** |
| Referência: limiar 0,5 | 0,5 | 45.347 | 2 | 16 | 36 | 0,6923 | 0,9474 | 0,8000 |

**Leitura:** subir o recall de 0,71 para 0,81 detecta **5 fraudes a mais** (37 → 42) ao custo de **93 falsos alarmes a mais** (2 → 95): ~19 alertas falsos por fraude adicional, e a precisão cai de 0,95 para 0,31. É o "cotovelo" da curva PR (S2.5: 10 a 14 fraudes têm score muito baixo em todos os modelos). O limiar exato é 0,03887500; logo acima dele (0,03917) o recall já cai para 0,788. Por isso o valor registrado foi arredondado **para baixo** (0,038875). Se o negócio preferir recall (por exemplo, com analistas revisando alertas), este é o segundo ponto a oferecer. Ele **não** foi aplicado.

### 4. Reprodutibilidade do `scripts/a8_campeao.py`

- Rodado de novo com `random-forest-tuned random-forest`: limiar 0,468518, as mesmas métricas, e **`src/best_params.json`, `a8_01_limiar.png` e `a8_02_confusao_antes_depois.png` idênticos byte a byte** (sha256 iguais antes e depois).
- O script **não tem aleatoriedade**: não treina nada e não usa semente. Ele lê o `holdout_scores.csv` de uma run já registrada e aplica funções determinísticas (`precision_recall_curve`, `choose_threshold`). A reprodutibilidade vem de ler sempre o mesmo artefato. **Atenção:** ele usa a run **mais recente** com aquele nome. Se o `tune random-forest` for rodado de novo, a run nova passa a ser a lida (o resultado seria igual se a busca chegasse à mesma configuração).
- **Dois detalhes cosméticos na saída do terminal, sem efeito nos arquivos:** a linha "limiar (máx. F1) = 0.468518" deveria dizer "limiar publicado"; e a linha de referência "recall >= 0.7" mostra o ponto do limiar exato (precisão 0,95, recall 0,7308), não o do publicado. Não corrigi, porque a instrução era não alterar código nesta verificação.

---

## Passagem de bastão (fim da sessão 2)

**Pronto:** protocolo sem vazamento (desenvolvimento / holdout / stream), A6 (9 variantes, notebook 03), diagnóstico estratificado × temporal × holdout, A7 (RF e MLP; TPE × Random Search), A8 (RF otimizada como `champion` v6, limiar 0,468518, stream PR-AUC 0,789). Comandos: `compare`, `diagnose`, `tune`, `final`, `stream_test` (`python -m src.train <comando>`), `scripts/a8_campeao.py`, `scripts/build_nb03.py`.

**Falta (frente A)**

| Tarefa | O que falta | Insumos prontos |
|---|---|---|
| **A9** | Curva de aprendizado da RF campeã (PR-AUC × tamanho do treino, no temporal) e o parágrafo de diagnóstico | Over/underfitting já discutido no notebook 03 (seção 8) e na S2.6; métricas `train_*` e `cv_temporal_train_*` no MLflow |
| **A10** | `reports/resultados.md` com os números finais e o run de origem de cada um | Tabelas das S2.4–S2.7; todas as figuras em `reports/figs/` (`eda_*`, `val_*`, `cmp_*`, `opt_*`, `a8_*`) |
| Notebook A7/A8 (opcional) | A7 e A8 estão documentadas aqui e nas figuras, não num notebook | `scripts/a8_campeao.py` e o MLflow |
| `final` na DAG | A B6 deve trocar a tarefa `baseline` por `final` | O `final` leva ~12 min com 400 árvores (holdout + 5-fold estratificado + 5-fold temporal + treino final). Se pesar na DAG diária, a frente A pode criar um modo que só treina e registra |

**Pendências das outras frentes**

| Frente | Pendência |
|---|---|
| **B** | Simulador reiniciado não gera dados novos por horas (`tx_id` repetido + `ON CONFLICT DO NOTHING`). Ver a "Nota para a frente B" acima |
| **B** | `FRAUD_THRESHOLD` padrão = **0.468518** no `docker-compose.yml` e no `.env.example` |
| **B (B6)** | Trocar `baseline` por `final` na DAG (lê `src/best_params.json`; move o `champion` só para o campeão) |
| **B/C** | `ruff check .` aponta 11 avisos em `api/`, `app/`, `embedded/`, `subscriber/` (`BLE001`, `S110`, `I001`) |
| **C** | Dashboard e relatório: usar o limiar 0,468518 e as métricas do campeão (S2.7). Números para o relatório: S2.2 a S2.7 |
| **README** | Conflito comum da porta 8080 (sessão 1, etapa 2) |
