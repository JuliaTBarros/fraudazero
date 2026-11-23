# CONTEXTO — FraudaZero, frente A (base de Dados e ML) + B1

Diário de trabalho da branch `dados/base-ml` (etapas 0 a 8 do plano combinado com o Irvin).
Cada etapa registra: o que foi feito, arquivos, decisões e justificativas, números reais, problemas, pendências e a sugestão de commit.

## Estado atual

- **Etapas 0 a 8 concluídas e validadas.** Marco M1 atingido: simulador → MQTT → subscriber → API (baseline) → Postgres → dashboard funcionando, DAG verde (4 tarefas), baseline no MLflow como `champion`, smoke test 2/2.
- **Próximo passo (próxima pessoa):** A6, comparação de modelos. Tudo o que é preciso está na seção [Passagem de bastão](#etapa-8--passagem-de-bastão).
- **Irvin:** revisar e commitar na ordem da seção [Commits sugeridos](#commits-sugeridos). **Nada foi commitado.** A [verificação final pré-commit](#verificação-final-pré-commit) passou sem precisar corrigir nada.
- **Cópia de trabalho:** `C:\dev\fraudazero` (branch `dados/base-ml`). A cópia no OneDrive está **obsoleta** (não foi alterada nem apagada).
- **Atenção nesta máquina:** a porta 8080 está ocupada por outro projeto seu; o Airflow está rodando em **8081** por um override local (ver etapa 2). A stack continua de pé; `make down` para desligar.

---

## Commits sugeridos

Na ordem abaixo, cada commit fica consistente sozinho (o código de um não depende de arquivos de commits posteriores).

| # | Mensagem | Arquivos |
|---|---|---|
| 1 | `Fixa as versões da pilha do modelo iguais no treino e na API` | `requirements.txt`, `docker/airflow/requirements.txt` |
| 2 | `Roda a DAG uma vez por dia em vez de a cada hora` | `dags/fraud_pipeline.py` |
| 3 | `Adiciona pré-processamento no Pipeline, validação reutilizável e baseline completo no MLflow` | `src/features.py`, `src/validation.py`, `src/train.py` |
| 4 | `Adiciona a EDA do bootstrap dataset com 7 visualizações interpretadas` | `notebooks/01_eda.ipynb`, `reports/figs/eda_01_classes.png` … `reports/figs/eda_07_pca.png` (7) |
| 5 | `Adiciona a validação do baseline: holdout temporal, k-fold, estabilidade e LOO` | `notebooks/02_experimentos.ipynb`, `reports/figs/val_01_baseline_pr_holdout.png`, `reports/figs/val_02_baseline_confusao_holdout.png`, `reports/figs/val_03_baseline_estabilidade_folds.png` |
| 6 | `Documenta decisões, resultados e passagem de bastão da base de ML` | `docs/CONTEXTO.md` |

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
