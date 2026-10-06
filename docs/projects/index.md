# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Luana Prado Lopes Guimaraes | luanaplg@al.insper.edu.br | LuanaPLGuimaraes |
| Laura Pontiroli Machado | laurapm@alinsper.edu.br | laupontiroli |

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação | [Classificação](classification/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/pt/2026.2/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

## Dataset

| | |
|---|---|
| **Nome** | Predicting Electric Vehicle Purchases — Kaggle Playground Series S6E9 |
| **Fonte (URL)** | https://www.kaggle.com/competitions/playground-series-s6e9 |
| **Licença / termos de uso** | Regras da competição Kaggle (conferir aba "Rules" antes da entrega final) |
| **Amostras** | 668.665 (treino) |
| **Features** | 13 (7 numéricas + 6 categóricas) |
| **Variável alvo** | `Will_Buy_EV` (binária: Yes/No) |
| **Tarefa escolhida** | Classificação |

Dataset tabular sobre comportamento de compra de veículos elétricos, combinando dados
demográficos (idade, renda, tipo de cidade) e comportamentais (ansiedade de autonomia,
disponibilidade de carregador em casa/trabalho, subsídio disponível). É uma tarefa de
classificação binária não trivial: o target é desbalanceado (~82% não compra / ~18%
compra), e algumas features (ex. `Subsidy_Available`, `Home_Charging_Possible`) têm relação
óbvia com o alvo, então parte do trabalho é avaliar risco de vazamento vs. sinal legítimo.

## Status

- [x] **1. EDA** — proposta em andamento
- [ ] **2. Classificação**
- [ ] **3. Generativo**

## Registro de decisões

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-09-15 | Dataset: EV Purchase Prediction (Kaggle S6E9). Fase 2: Classificação (não Regressão) | Target `Will_Buy_EV` já é categórico/binário — classificação é a tarefa natural |