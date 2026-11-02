---
project: classification
task: classificação
dataset: "https://www.kaggle.com/competitions/playground-series-s6e9"
team: ["Luana Prado Lopes Guimaraes", "Laura Pontiroli Machado"]
ai_use: "Uso assistido de IA (Claude) para discutir critérios metodológicos, revisar código e auxiliar a organização do texto. Todos os números, decisões e interpretações foram definidos e escritos pela equipe."
---

# 2. Classificação

!!! abstract "Entrega 2 de 3 do [Projeto](../index.md)"

    [Projects → Classification](https://insper.github.io/ann-dl/){:target='_blank'}

!!! info "Equipe"

    | Nome completo | E-mail | GitHub |
    |---------------|--------|--------|
    | Luana Prado Lopes Guimaraes | luanaplg@al.insper.edu.br | LuanaPLGuimaraes |
    | Laura Pontiroli Machado | laurapm@al.insper.edu.br | laupontiroli |

    Dataset, decisões e status: [página do projeto](../index.md).

Esta entrega parte do [EDA](../eda/index.md), que já caracterizou o dataset em profundidade. Aqui resumimos e apontamos para lá em vez de repetir a análise.

## 0. Ponto de partida

O dataset e as decisões da equipe ficam na [página do projeto](../index.md). Esta entrega parte do [EDA](../eda/index.md), que já caracterizou os dados em profundidade; aqui resumimos e apontamos para lá em vez de repetir a análise.

## Escolha do dataset

Usamos o dataset **Playground Series S6E9 – Predicting Electric Vehicle Purchases**, do Kaggle ([link](https://www.kaggle.com/competitions/playground-series-s6e9)). O arquivo de treino tem 668.665 linhas, 13 variáveis explicativas, um `id` e o alvo `Will_Buy_EV` (Yes/No). A tarefa é uma classificação binária que não é trivial por três motivos: o alvo é desbalanceado (17,46% "Yes"), as variáveis misturam tipos numéricos e categóricos, e os dados têm particularidades que o EDA documentou (valores acumulados no piso de renda e deslocamento, e uma variável que praticamente entrega a resposta).

## Descrição do dataset

- **Alvo:** `Will_Buy_EV` (compra de veículo elétrico, Yes/No), com 82,54% "No" e 17,46% "Yes".
- **Numéricas (7):** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work` e `Environmental_Concern_Level`.
- **Categóricas (5):** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible` e `Subsidy_Available`.
- **Descartada:** `Range_Anxiety_Level`, por quase determinar o alvo (vazamento).

**Pré-processamento.** Reutilizamos o pipeline definido no [plano de pré-processamento do EDA](../eda/index.md#8-plano-de-pre-processamento), implementado em `eda/code/preprocess.py`: tratamento dos valores no piso (renda = 30.000 e deslocamento = 5 km) com indicador de ausência e imputação pela mediana, winsorização por IQR, padronização das numéricas e one-hot das categóricas. `Range_Anxiety_Level` e `id` são descartados. O pipeline é ajustado somente nas linhas de ajuste, nunca na validação nem no teste. `Subsidy_Available` é mantida (justificativa na etapa 1).

**Métrica principal: PR-AUC da classe "Yes"** (área sob a curva precisão-recall, calculada como *average precision*). Escolhida *antes* de treinar a rede porque:

- o alvo é desbalanceado (17,46% "Yes"): a acurácia engana, já que o Dummy acerta 82,54% sem aprender nada;
- ela avalia o ordenamento das probabilidades em todos os cortes, sem depender do corte de 0,5, o que mantém a comparação justa quando a rede usar ponderação de classes (que desloca as probabilidades);
- um modelo que ordena ao acaso obtém PR-AUC igual à proporção de "Yes" (17,46%), e o perfeito obtém 100%.

Reportamos também precisão, recall, F1, acurácia balanceada e ROC-AUC como métricas secundárias.

**Critério de sucesso (definido antes de treinar a rede):** a PR-AUC da rede no teste, em média de 3 seeds, deve superar a da melhor baseline no teste em pelo menos 1 ponto percentual, e essa diferença deve ser maior que o desvio padrão entre as seeds. O limiar de 1 ponto é uma escolha de julgamento: as duas baselines diferem em apenas 0,08 ponto de PR-AUC na validação, então diferenças bem menores que 1 ponto não se distinguem de ruído.

!!! note "Desbalanceamento"

    Nas baselines nenhum balanceamento foi aplicado. A decisão sobre pesos na perda ou reamostragem para a rede será tomada e registrada nas etapas 2 e 4.

## 1. Baselines

Os três modelos usam o mesmo pipeline e foram avaliados na validação (o teste não foi tocado), com corte de decisão 0,5 e sem balanceamento de classes. Valores em %:

| Modelo | Acurácia | Acur. balanceada | Precisão | Recall | F1 | ROC AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Dummy (classe majoritária) | 82,54 | 50,00 | 0,00 | 0,00 | 0,00 | 50,00 | 17,46 |
| Regressão logística | 89,24 | 79,44 | 71,21 | 64,39 | 67,63 | 93,57 | 73,72 |
| Random forest | 89,33 | 79,20 | 72,01 | 63,63 | 67,56 | 93,50 | 73,80 |

**Interpretação.**

- O Dummy acerta 82,54% apenas por dizer sempre "No", o que mostra que a acurácia sozinha não serve nesse caso. Sua PR-AUC de 17,46% é o patamar do acaso.
- Regressão logística e random forest empatam (diferença de 0,08 ponto de PR-AUC e 0,07 de F1). Um modelo simples e quase linear chega ao mesmo resultado que um modelo complexo, o que sugere que a relação entre as variáveis e o alvo é em grande parte direta. Os hiperparâmetros da random forest (100 árvores, profundidade máxima 20) não foram otimizados, então o empate vale para essa configuração.
- Com corte 0,5, o recall é de ~64%: cerca de um terço dos compradores reais não é encontrado, embora o ROC AUC seja alto (~93,5). Esse corte será revisitado na etapa 5.
- **Subsídio:** sem `Subsidy_Available`, a PR-AUC cai de ~73,7 para ~58,8 (regressão logística) e o F1 de ~67,6 para ~51,1. A variável carrega muita informação e é mantida.

## 2. A rede

<!-- Exigido: camadas, larguras, ativações e inicialização justificadas; nº de parâmetros vs. nº de linhas de treino; 1 unidade de saída + BCEWithLogitsLoss (a rede devolve logits); rede e loop de treino em code/. -->

(a preencher)

## 3. Treinamento

<!-- Exigido: otimizador (SGD+momentum/Adam/AdamW), learning rate, batch e épocas justificados; early stopping na validação restaurando os melhores pesos; curvas de treino e validação por época, LIDAS no texto (converge? onde começa o overfitting?). -->

(a preencher)

## 4. Regularização e hiperparâmetros

<!-- Exigido: pelo menos 2 técnicas (weight decay, dropout, BatchNorm/LayerNorm) comparadas com a rede sem nenhuma; busca com pelo menos 8 configurações escolhidas pela VALIDAÇÃO, em tabela ordenada. -->

(a preencher)

## 5. Avaliação no teste

<!-- Exigido: retreinar a config escolhida e avaliar UMA vez; matriz de confusão; precisão/revocação/F1 por classe; F1 macro; ROC-AUC; curva PR e PR-AUC; corte escolhido na validação e justificado pelo custo do erro; tabela rede + baselines no mesmo teste; 3 seeds (média ± desvio); a diferença é maior que a variação?; critério da etapa 0 atingido? -->

(a preencher)

## 6. Análise de erros

<!-- Exigido: erro por subgrupo (uma categoria da EDA e faixas do alvo/numérica); piores exemplos com hipótese por padrão; importância por permutação vs. o que a EDA sugeria. -->

(a preencher)

## 7. Conclusão

<!-- Exigido: "A rede valeu a pena?"; limitações; quem é afetado por um erro do modelo; próximos passos. -->

(a preencher)

## Referências

(a preencher)

## Resumo dos resultados

| # | Item | Resultado |
|---|------|-----------|
| 1 | Dataset, tarefa, alvo e métrica principal | Playground Series S6E9; classificação binária; `Will_Buy_EV`; PR-AUC da classe "Yes" |
| 2 | Linhas de treino · validação · teste | 427.945 · 106.987 · 133.733 |
| 3 | Melhor baseline e sua métrica no teste | (a preencher na etapa 5) |
| 4 | Rede escolhida: camadas × unidades, nº de parâmetros, regularização | (a preencher) |
| 5 | Otimizador, learning rate, batch, época de parada | (a preencher) |
| 6 | Métrica principal da rede no teste (média ± desvio, 3 seeds) | (a preencher) |
| 7 | Critério de sucesso atingido? | (a preencher) |
| 8 | Subgrupo com maior erro e seu valor | (a preencher) |