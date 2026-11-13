# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em duas partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Luana Prado Lopes Guimaraes | luanaplg@al.insper.edu.br | LuanaPLGuimaraes |
| Laura Pontiroli Machado | laurapm@al.insper.edu.br | laupontiroli |

## As entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação | [Classificação](classification/index.md) |

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

- [x] **1. EDA** — entregue
- [x] **2. Classificação** — entregue

## Registro de decisões

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-09-15 | Dataset: EV Purchase Prediction (Kaggle S6E9). Fase 2: Classificação (não Regressão) | Target `Will_Buy_EV` já é categórico/binário — classificação é a tarefa natural |
| 2026-11-10 | Classificação: corte de decisão = prevalência de "Yes" no ajuste (0,175), em vez de 0,5 | Espelha o desbalanceamento (17,5% de "Yes"); na validação, o recall sobe de 64% para 91% com F1 praticamente igual — deixar escapar um comprador foi tratado como o erro mais caro |
| 2026-11-10 | Classificação: métricas em destaque passam a ser PR-AUC (principal, inalterada), recall e ROC AUC/curva ROC | Recall mostra o efeito do corte escolhido; ROC AUC permite comparar com outros grupos da mesma competição |
| 2026-11-13 | Classificação: boosting (HistGradientBoosting) incluído na avaliação do teste como referência extra, fora do critério de sucesso | Mostra o ganho possível com outro tipo de modelo sem alterar o critério definido antes de treinar a rede |
| 2026-11-13 | O projeto passa a ter duas entregas (EDA e Classificação): a entrega 3 (Generativo) não será realizada e saiu do site | A entrega deixou de existir |
