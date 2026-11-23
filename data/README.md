# Dados

Os arquivos de dados não são versionados (a base tem ~150 MB).

1. Baixe o *Credit Card Fraud Detection* (ULB/Kaggle):
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```
   ou baixe manualmente em https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. O arquivo deve ficar em `data/raw/creditcard.csv`.
3. A tarefa `prepare` da DAG gera:
   - `data/processed/train.csv`: primeiros 80% por `Time` (bootstrap dataset: EDA e treino)
   - `data/stream/stream.csv`: últimos 20% (fluxo publicado pelo simulador via MQTT)
