# FraudaZero — contexto para o Claude Code

Projeto da disciplina **Aprendizado de Máquina** (CESAR School, 2026.6, unidade 2a), feito em grupo.
Detecção de fraude em transações de cartão, de ponta a ponta: simulador → MQTT → Postgres → Airflow → MLflow → API → dashboard, tudo em Docker Compose.

O que fazer e em que ordem: leia PLANO_IMPLEMENTACAO.md (tarefas por frente, marcos, critérios de pronto). Pergunte à pessoa qual é a frente dela e trabalhe só nela.

## Regras de ouro

1. **Prazo curto (2 semanas). Faça o mais simples que funciona.** Nada de abstrações, camadas, padrões ou configurações "para o futuro". Se uma solução tem 20 linhas e outra tem 200, use a de 20.
2. **O professor avalia o sistema rodando e o relatório, não o código.** Prioridade: `make up` funciona do zero, a demo integrada funciona, os números do relatório são reais.
3. **Não invente métricas, resultados ou versões.** Toda métrica vem do MLflow ou de uma execução real.
4. **Consulte a documentação atual** (MCP Context7) antes de escrever código de Airflow, MLflow, Streamlit ou FastAPI. Cite a versão fixada ao pesquisar.
5. Pergunte antes de: adicionar serviço novo ao Compose, trocar uma tecnologia do stack, adicionar dependência pesada, ou mudar os contratos abaixo.

## Stack (versões fixadas, não atualizar no meio do projeto)

- Python 3.11 · Docker Compose (execução com um único comando)
- Eclipse Mosquitto 2 (broker MQTT) · Postgres 16 (3 bancos: `fraudazero`, `airflow`, `mlflow`)
- Apache Airflow **2.10.5** (LocalExecutor, treino em venv próprio `/opt/train-venv`)
- MLflow **2.x (<3)** com backend Postgres e artefatos em `./mlruns`
- scikit-learn **1.6.1** — a mesma versão no treino (Airflow) e na API, senão o modelo não carrega
- FastAPI (API) · Streamlit (dashboard) · paho-mqtt ≥ 2 (usa `CallbackAPIVersion.VERSION2`)

## Estrutura e donos

| Pasta | Conteúdo | Frente |
|---|---|---|
| `src/common/` | `config.py` (env vars) e `schema.py` (lista `FEATURES`), usados por todos | todos |
| `src/train.py` | CLI: `ingest`, `prepare`, `baseline`, `compare`, `tune` | Dados e ML |
| `notebooks/` | EDA e experimentos | Dados e ML |
| `embedded/` | Simulador do sistema embarcado (publica no MQTT) | Infra |
| `subscriber/` | Consome MQTT, chama a API, grava no Postgres | Infra |
| `dags/` | DAG `fraudazero_pipeline` | Infra |
| `docker/`, `docker-compose.yml`, `Dockerfile`, `Makefile` | Execução | Infra |
| `api/` | FastAPI de inferência | API/Dashboard |
| `app/` | Dashboard Streamlit | API/Dashboard |
| `reports/` | Relatório no template SBC | API/Dashboard |

Mexa só na pasta da sua frente. Mudou algo em `src/common/` ou nos contratos? Avise o grupo no PR.

## Contratos (mudar só com acordo do grupo)

**MQTT** — tópico `fraudazero/transactions`, QoS 1, JSON:

```json
{"tx_id": "tx-0-1", "ts": 1760000000.0, "features": {"Time": 0.0, "V1": 0.1, "...": 0.0, "V28": 0.0, "Amount": 12.5}, "label": 0}
```

`label` existe só para monitorar métricas ao vivo no dashboard. **Nunca** use `label` como entrada do modelo.

**API** — `GET /health` → `{"status","model_loaded"}` · `POST /predict` com `{"features": {...}}` → `{"score","is_fraud","threshold"}` · `POST /reload` recarrega o modelo `models:/fraudazero@champion`.

**Postgres** (banco `fraudazero`) — tabela `transactions(tx_id, received_at, features jsonb, label, score, is_fraud, latency_ms)`, criada pelo subscriber.

**Dados** — `data/raw/creditcard.csv` (Kaggle, não versionado). `prepare` ordena por `Time` e gera `data/processed/train.csv` (primeiros 80%) e `data/stream/stream.csv` (últimos 20%, que o simulador publica).

## Regras de ML (valem nota: Modelagem e Validação 30%)

- Métrica principal: **PR-AUC**; reporte também recall, precisão e F1 da classe fraude e matriz de confusão. **Nunca** acurácia como métrica principal (a base tem ~0,17% de fraudes).
- Desbalanceamento (SMOTE ou `class_weight`) **somente dentro do `Pipeline`**, nos folds de treino. Escalonar e reamostrar antes do split é vazamento de dados.
- Validação: holdout temporal + stratified k-fold com análise de estabilidade. Leave-One-Out não se aplica (~285 mil linhas): justificar no relatório.
- Modelos a comparar: KNN (em subamostra), Árvore de Decisão, Random Forest, AdaBoost, MLP. Baseline: regressão logística. Modelo fora da disciplina só com aprovação do professor.
- Hiperparâmetros com Optuna ou Random Search. Experimentos separados no MLflow: `baseline`, `comparacao`, `otimizacao`. Sempre registrar parâmetros, métricas, artefatos e modelo.
- EDA: mínimo de 5 visualizações, cada uma com interpretação escrita.

## Comandos

```bash
make up       # sobe tudo (docker compose up --build -d)
make down     # derruba
make logs     # logs de todos os serviços
make reset    # derruba e apaga volumes (recomeça do zero)
make train    # dispara a DAG fraudazero_pipeline
make check    # valida o docker-compose.yml
make test     # smoke test (precisa dos serviços de pé)
```

Endereços locais: Airflow `:8080` (admin/admin) · MLflow `:5001` · API `:8000/docs` · Dashboard `:8501` · Postgres `:5433` · MQTT `:1883`.

## Convenções

- Código e identificadores em inglês; comentários, README e relatório em português do Brasil.
- Config só via variáveis de ambiente (`src/common/config.py`). Sem segredos no repositório.
- Commits pequenos, branches curtas por frente (`dados/…`, `infra/…`, `api/…`), PR com revisão de um colega.
- Notebooks sem saída commitada (`nbstripout`).
- Formatação com `ruff format` (um hook já roda ao editar arquivos `.py`).
- Visual do dashboard: tema em `.streamlit/config.toml` e tokens/regras globais em `app/style.css`. Todo componente segue os dois: nada de cor fixa no Python, cards com `st.container(border=True, key="card-...")`, CSS só via classes `.st-key-*`.

## Definição de pronto

- [ ] `git clone` limpo + `make up` + dataset em `data/raw/` = sistema funcionando, sem passos manuais além do README
- [ ] Demo integrada: simulador → MQTT → Postgres → inferência na API → dashboard
- [ ] Relatório no template SBC (`reports/`), skill `relatorio-sbc`
- [ ] README completo (nomes, GitHub, instruções, link do Google Sites)

## Armadilhas conhecidas

- RAM: Airflow + MLflow + Postgres + serviços juntos pedem ~6–8 GB. Se faltar, reduza o intervalo de publicação do simulador antes de cortar serviços.
- Airflow 2.x e 3.x têm APIs diferentes: use sempre a documentação da **2.10**.
- `docker/postgres/init.sql` só roda em volume vazio: use `make reset` se mudar.
- Linux: se o Airflow não conseguir escrever em `data/`, rode `sudo chown -R 50000:0 data mlruns`.
- No macOS a porta 5000 é ocupada pelo AirPlay: por isso o MLflow está em `5001`.
