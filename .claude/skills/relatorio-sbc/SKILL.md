---
name: relatorio-sbc
description: Use ao escrever, revisar ou compilar o relatório do projeto FraudaZero no template SBC (pasta reports/). Garante o formato obrigatório e que os números venham de execuções reais.
---

# Relatório do FraudaZero (template SBC)

## Regra que não pode ser quebrada
O relatório deve usar **o template SBC fornecido pelo professor**, sem alterar margens, fontes, espaçamento ou número de colunas. Relatório fora do formato SBC recebe **nota 0 em Comunicação (20% da nota)**. Documentos livres (Word, Google Docs) não valem.

- O template fica em `reports/` (o grupo deve copiar para lá os arquivos fornecidos com a especificação). Se não estiverem lá, **pare e peça** os arquivos; não improvise um template parecido.
- Não troque a classe do documento, não adicione pacotes que mexam em formatação (`geometry`, `setspace`, `fontspec` etc.) nem use `\vspace` para empurrar conteúdo.
- Entregável final: PDF em `reports/`. Compile com `latexmk -pdf` ou `tectonic`, ou o grupo usa o Overleaf. Verifique que o PDF gerado abre e tem as figuras.

## Estrutura sugerida
Resumo/Abstract · 1 Introdução (problema, dataset, objetivos) · 2 Dados e EDA · 3 Metodologia (tratamento, validação, métricas, hiperparâmetros) · 4 Modelagem e resultados · 5 Arquitetura MLOps (MQTT, Airflow, MLflow, API, dashboard, Docker) · 6 Limitações · 7 Conclusão · Referências.

## Conteúdo exigido pela especificação (cada item vale nota)
- EDA com estatísticas descritivas e **mínimo de 5 visualizações, cada uma interpretada**; decisões de tratamento justificadas.
- Validação: holdout temporal, k-fold estratificado, análise de estabilidade, overfitting/underfitting e generalização; justificar por que Leave-One-Out não se aplica.
- Baseline justificado + comparação de KNN, Árvore, Random Forest, AdaBoost e MLP, com visualizações (matriz de confusão, curvas PR) e **análise crítica**.
- Otimização de hiperparâmetros: estratégia, espaço de busca e resultados.
- Descrição da arquitetura integrada, decisões técnicas e **limitações da solução**.

## Como tratar números
- Toda métrica, tabela e figura vem do MLflow (`http://localhost:5001`), dos notebooks ou de uma execução real. **Nunca invente ou arredonde "de cabeça".**
- Se um número ainda não existe, escreva `TODO(número)` no texto e liste a pendência no final da resposta.
- Cite no texto o experimento e o run do MLflow de onde o valor saiu.
- Escreva em português do Brasil, sem enfeite: frases curtas, cada figura com legenda e interpretação.

## Antes de dar o relatório como pronto
- [ ] Compila no template original, sem warnings de formatação
- [ ] Todas as figuras citadas no texto e interpretadas
- [ ] Nenhum `TODO` restante
- [ ] Nomes dos integrantes e link do repositório presentes
