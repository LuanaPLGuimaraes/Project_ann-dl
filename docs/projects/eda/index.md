---
project: eda
ai_use: "none"
---

# 1. EDA — Análise Exploratória

!!! abstract "Entrega 1 de 3 do [Projeto](../index.md)"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

!!! info "Equipe"

    | Nome completo | E-mail | GitHub |
    |---------------|--------|--------|
    | Luana Prado Lopes Guimaraes | luanaplg@al.insper.edu.br | LuanaPLGuimaraes |
    | Laura Pontiroli Machado | laurapm@alinsper.edu.br | laupontiroli |

    Dataset, decisões e status: [página do projeto](../index.md).

!!! tip "O que esta entrega decide"

    O EDA não é um álbum de gráficos: é onde a equipe **escolhe o dataset** e descobre o que
    vai atrapalhar o treino depois — desbalanceamento, vazamento, escalas incompatíveis com a
    ativação, ausências não aleatórias. Cada achado aqui deve virar uma linha do plano de
    pré-processamento no fim da página, e é esse plano que as duas entregas
    seguintes executam.

    As aulas de **Classes → Data** no
    [site da disciplina](https://insper.github.io/ann-dl/){:target='_blank'} dão a estrutura:
    tipos, distribuições, qualidade, desbalanceamento, vazamento, split e pré-processamento.

## 1. Dataset

- **Nome:** Playground Series — Season 6, Episode 9 (Predicting Electric Vehicle Purchases)
- **Fonte:** Kaggle, competição "Playground Series S6E9"
- **Licença:** dataset sintético disponibilizado para a competição, uso autorizado para fins de estudo/competição conforme as regras do Playground Series
- **Dimensões:** 668.665 amostras no total, divididas em 534.932 (treino) e 133.733 (teste), split 80/20 estratificado pelo alvo, `random_state=42`. 13 features + `id` (identificador) + alvo `Will_Buy_EV`
- **Por que este dataset:** Dataset de competition interessante que se enquadrava como interesse pelas duas da dupla. Outros datasets foram considerados e analisados mas esse se destacou como maior interesse para ambas.

## 2. Estrutura e tipos

534.932 amostras no conjunto de treino, 13 features (fora `id` e o alvo).

| Feature | Tipo | Cardinalidade / faixa | Observação |
|---------|------|-----------------------|------------|
| id | Identificador | 534.932 valores únicos | Não é feature |
| Age | Numérica discreta | 45 valores únicos | Idade em anos inteiros |
| Annual_Income_USD | Numérica contínua | 12.406 valores únicos | Alta granularidade, renda |
| Daily_Commute_km | Numérica contínua | 794 valores únicos | Distância de deslocamento diário |
| Number_of_Cars_Owned | Numérica discreta | 4 valores únicos | Contagem de carros |
| Charging_Stations_Near_Home | Numérica discreta | 15 valores únicos | Contagem |
| Charging_Stations_Near_Work | Numérica discreta | 20 valores únicos | Contagem |
| Environmental_Concern_Level | Ordinal discreta | 5 níveis (1–5) | Lida como `float64` pelo pandas, mas é escala ordenada, não contínua |
| Gender | Categórica nominal | Female, Male, Other | Sem ordem natural |
| City_Type | Categórica nominal | Suburban, Urban, Rural | Sem ordem adotada |
| Current_Car_Type | Categórica nominal | SUV, Sedan, Truck, Hatchback | Sem ordem natural |
| Home_Charging_Possible | Categórica binária | 2 categorias (Yes/No) | - |
| Subsidy_Available | Categórica binária | 2 categorias (Yes/No) | Possível risco de vazamento — retomar na seção de riscos |
| Range_Anxiety_Level | Categórica ordinal | Low, Medium, High | Lida como `object` pelo pandas |


## 3. Variável alvo

- **Proporção por classe:** `No` = 82,54% / `Yes` = 17,46%
- **Razão entre maior e menor classe:** ≈ 4,73
- **Baseline:** um classificador que sempre responde `No` acerta **82,54%** — a classificação precisa superar esse valor

![Distribuição da variável alvo](figures/fig01-distribuicao-alvo.svg)
/// caption
**Figura 1** — Distribuição da variável alvo (`Will_Buy_EV`) no conjunto de treino.
///

O alvo está desbalanceado numa razão de aproximadamente 4,73:1 entre as classes `No` e
`Yes`. Um classificador trivial que sempre responde `No` (a classe majoritária) já acerta
82,54% das amostras sem aprender nada sobre os dados — esse é o nosso *baseline*. Qualquer
modelo de classificação treinado nas próximas entregas só tem valor real se superar essa
acurácia; caso contrário, ele não está aprendendo nada além da proporção das classes.

## 4. Análise univariada

### Medidas de posição e dispersão

| Feature | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| Age | 47,02 | 12,87 | 25,0 | 36,0 | 47,0 | 58,0 | 69,0 |
| Annual_Income_USD | 84.754,35 | 28.623,91 | 30.000,0 | 67.380,0 | 84.870,0 | 102.751,0 | 188.549,0 |
| Daily_Commute_km | 32,15 | 18,73 | 5,0 | 17,2 | 33,6 | 47,4 | 98,7 |
| Number_of_Cars_Owned | 1,71 | 0,73 | 1,0 | 1,0 | 2,0 | 2,0 | 4,0 |
| Charging_Stations_Near_Home | 4,96 | 3,93 | 0,0 | 2,0 | 4,0 | 7,0 | 14,0 |
| Charging_Stations_Near_Work | 7,17 | 5,18 | 0,0 | 3,0 | 6,0 | 10,0 | 19,0 |
| Environmental_Concern_Level | 2,93 | 1,43 | 1,0 | 2,0 | 3,0 | 4,0 | 5,0 |

### Formato (histogramas)

Das 7 features acima, só 3 têm alta cardinalidade (`Age` = 45 valores únicos,
`Annual_Income_USD` = 12.406, `Daily_Commute_km` = 794) — essas são as únicas cujo formato
torna mais difícil de entender apenas pela tabela de posição/dispersão acima, por isso elas recebem
histograma individual. As demais, por terem poucos valores distintos (contagens de 4 a 20
categorias), já estão descritas pela tabela.

![Distribuição de Age](figures/fig02-age.svg)
/// caption
**Figura 2** — Distribuição de `Age` no conjunto de treino.
///

![Distribuição de Annual_Income_USD](figures/fig03-annual-income-usd.svg)
/// caption
**Figura 3** — Distribuição de `Annual_Income_USD` no conjunto de treino.
///

![Distribuição de Daily_Commute_km](figures/fig04-daily-commute-km.svg)
/// caption
**Figura 4** — Distribuição de `Daily_Commute_km` no conjunto de treino.
///

- **`Age`**: distribuição aproximadamente uniforme entre 25 e 69 anos, sem assimetria nem
  concentração em nenhuma faixa específica.
- **`Annual_Income_USD`**: formato com leve assimetria à direita, mas com uma concentração
  anômala de 9,20% das amostras exatamente no valor mínimo (30.000), destoando do restante
  da curva.
- **`Daily_Commute_km`**: mesmo padrão, de forma ainda mais intensa, 21,59% das amostras
  caem exatamente no valor mínimo (5,0 km), o que é  improvável para uma medida contínua de distância.

!!! warning "Achado: valor mínimo pode ser não-resposta disfarçada"

    As concentrações exatas no valor mínimo em `Annual_Income_USD` e `Daily_Commute_km`
    não aparecem como `NaN`, mas o padrão (pico isolado, exatamente no piso da
    escala, destoa do resto da distribuição) sugere que esses valores representam
    respostas ausentes definidas como o mínimo da escala, em vez de dados reais. O
    tratamento desse achado será definido posteriormente.

## 5. Análise bivariada e correlações

Relação entre as features e o alvo, e entre as features.

!!! danger "Correlação alta demais com o alvo é suspeita"

    Uma feature que prevê o alvo quase perfeitamente costuma ser **vazamento**: informação
    que só existe depois do fato que você quer prever. Investigue antes de comemorar.

## 6. Qualidade dos dados

### Valores ausentes

| Feature | % ausente | Padrão (aleatório?) | Tratamento planejado |
|---------|-----------|---------------------|----------------------|
| | | | |

Ausência raramente é aleatória. Se falta mais em um grupo do que em outro, o próprio "estar
ausente" carrega informação.

### Duplicatas e inconsistências

Linhas repetidas, categorias escritas de formas diferentes, unidades misturadas, datas
impossíveis.

### Outliers

Como foram detectados e o que será feito com eles — e por quê. Remover outlier é decisão de
modelagem, não faxina.

## 7. Riscos de vazamento

Liste as fontes de vazamento identificadas e como cada uma será contida.

``` mermaid
flowchart LR
    raw[Dados brutos] --> split{{split treino/teste}}
    split -->|treino| fit["fit_transform<br/>(estatísticas saem só daqui)"]
    split -->|teste| apply[transform]
    fit --> model[Modelo]
    apply --> model
```

| Risco | Onde aparece | Contenção |
|-------|--------------|-----------|
| Estatísticas calculadas antes do split | | Ajustar transformadores só no treino |
| | | |

## 8. Plano de pré-processamento

A saída desta entrega. Uma linha por transformação, ligando cada uma a um achado acima.

| # | Transformação | Features | Motivo (seção) |
|---|---------------|----------|----------------|
| 1 | | | |

## 9. Estratégia de split

Proporções, estratificação, e o que impede uma mesma entidade de cair nos dois lados
(agrupamento por usuário, por data, por sessão).

## Results summary

| # | Métrica | Valor |
|---|---------|-------|
| 1 | Amostras | |
| 2 | Features (antes / depois do encoding) | |
| 3 | Features com ausentes | |
| 4 | Maior % de ausência em uma feature | |
| 5 | Linhas duplicadas | |
| 6 | Razão de desbalanceamento do alvo | |
| 7 | Acurácia (ou erro) do baseline trivial | |
| 8 | Maior correlação feature–alvo | |
| 9 | Amostras treino / teste após o split | |

## Conclusão

O que o dataset permite e o que ele impede. Se algum achado inviabiliza a tarefa pretendida,
é aqui que a equipe muda de rumo — ainda dá tempo.

## Referências
