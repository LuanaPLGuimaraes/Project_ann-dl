---
project: eda
task: classificação
dataset: "Playground Series S6E9 — Predicting Electric Vehicle Purchases (Kaggle)"
team: ["Luana Prado Lopes Guimaraes", "Laura Pontiroli Machado"]
ai_use: "Uso assistido de IA para estruturar a análise, revisar código, discutir critérios metodológicos e redigir o relatório. Todos os números, decisões e interpretações foram definidos e escritos pela equipe."
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
82,54% das amostras sem aprender nada sobre os dados, sendo esse o nosso *baseline*. Qualquer
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
`Annual_Income_USD` = 12.406, `Daily_Commute_km` = 794), essas são as únicas cujo formato
torna mais difícil de entender apenas pela tabela de posição/dispersão acima, por isso elas recebem
histograma individual. As demais, por terem poucos valores distintos, já estão descritas pela tabela.

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


Todas as estatísticas desta seção foram calculadas **só no conjunto de treino** (534.932 linhas).
A taxa geral de `Yes` no treino é **17,46%** (Seção 3), e é a referência de todas as comparações abaixo.

### 5.A Numérica × numérica

**Método.** Usamos a correlação de **Spearman**. Quatro das sete numéricas são contagens ou escalas
ordinais (`Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`,
`Environmental_Concern_Level`), e `Annual_Income_USD` e `Daily_Commute_km` têm picos
exatamente no valor mínimo (9,20% e 21,59%, Seção 4) que distorcem Pearson. Spearman mede relação
monotônica sem supor normalidade e é menos sensível a esses picos. Na prática os dois métodos
concordam: o par mais correlacionado tem ρ = +0,543 (Spearman) e r = +0,510 (Pearson), e as demais
células diferem em no máximo 0,03.

![Matriz de correlação de Spearman](figures/fig05-correlacao.svg)
/// caption
**Figura 5** — Correlação de Spearman entre as 7 numéricas (treino).
///

**Conclusão (Fig. 5).** O par mais correlacionado é **`Charging_Stations_Near_Home` ×
`Charging_Stations_Near_Work` (ρ = +0,543)**. É uma correlação moderada, **abaixo do limiar de 0,7** que
usamos para chamar um par de redundante: nenhum dos 21 pares passa dele. O segundo maior é
`Annual_Income_USD` × `Environmental_Concern_Level` (ρ = +0,075), e todos os demais ficam abaixo de
|0,05|. Portanto **não removemos nenhuma numérica por redundância**; as duas contagens de estações de
carregamento ficam, porque cada uma tem relação quase nula com o alvo (ρ = −0,022 e −0,014) e a
colinearidade moderada não prejudica uma rede neural.

![Scatter dos pares mais correlacionados](figures/fig06-scatter-pares.svg)
/// caption
**Figura 6** — Dispersão dos 3 pares numéricos mais correlacionados (amostra de 5.000 linhas do treino).
///

**Conclusão (Fig. 6).** No par das estações de carregamento a nuvem
mostra tendência positiva, em grade, porque as duas variáveis são contagens inteiras; nos outros dois pares
(ρ = +0,075 e +0,043) não há padrão visível. Isso **confirma** a conclusão da Figura 5: só um par tem
relação apreciável, e ela é moderada.

**Correlação com o alvo.** As numéricas mais associadas a `Will_Buy_EV` são
**`Environmental_Concern_Level` (ρ = +0,461)** e **`Annual_Income_USD` (ρ = +0,224)**.
`Daily_Commute_km` tem ρ = −0,044 e as quatro restantes têm |ρ| ≤ 0,022, isto é, nenhuma
associação monotônica detectável com o alvo individualmente.

### 5.B Categórica × alvo

Em cada figura, a linha tracejada é a taxa geral de `Yes` (17,46%). Barras longe dela indicam
categorias que mudam a probabilidade de compra.

![Subsidy_Available e Home_Charging_Possible vs alvo](figures/fig07-cat-vs-alvo-1.svg)
/// caption
**Figura 7** — Proporção de `Yes` por categoria de `Subsidy_Available` e `Home_Charging_Possible`.
///

**Conclusão (Fig. 7).**

- **`Subsidy_Available`:** sem subsídio (198.952 linhas) a taxa de `Yes` é de apenas **0,57%**; com
  subsídio (335.980 linhas) é de **27,47%**, cerca de 48 vezes maior. Estimamos, a partir dessas taxas e
  contagens, que cerca de 99% dos compradores do treino têm subsídio disponível. Praticamente ninguém compra
  sem ele, o que faz dessa a feature categórica mais forte do dataset.
- **`Home_Charging_Possible`:** 12,69% sem carregador em casa contra 19,59% com ele (+6,9 p.p., ×1,54).
  É um efeito real, mas moderado.

**Decisão sobre `Subsidy_Available`.** Não a classificamos como vazamento, porque **não é derivada do
alvo**: a descrição do dataset original a trata como um 
fator econômico do comprador, não resultado da compra (Seção 7).
É uma condição que existe antes da decisão de compra
e tem leitura econômica direta. Mas a relação é tão forte que o modelo pode se apoiar quase só nela e
mascarar as outras features. Por isso **vamos treinar com e sem `Subsidy_Available`** na entrega de
Classificação e comparar as métricas. O `preprocess.py` já aceita as duas versões (`use_subsidy=True/False`).

![Range_Anxiety_Level e City_Type vs alvo](figures/fig08-cat-vs-alvo-2.svg)
/// caption
**Figura 8** — Proporção de `Yes` por `Range_Anxiety_Level` e `City_Type`.
///

**Conclusão (Fig. 8).**

- **`Range_Anxiety_Level`:** a taxa cai de **18,90%** (Low, 483.239 linhas, 90,3% do treino) para
  **4,18%** (Medium, 49.920 linhas) e **0,11%** (High, 1.773 linhas). A relação é monotônica e quase
  determinística nos níveis Medium e High: quem tem ansiedade de autonomia acima de Low praticamente nunca
  compra.
- **Decisão: descartamos `Range_Anxiety_Level`.** (a descrição do dataset original a trata como um segundo alvo, seção 7). Uma feature construída a partir do que
  queremos prever carrega o rótulo para dentro do modelo e inflaria as métricas sem generalizar. Ela fica
  visível nesta figura só como evidência da decisão; fora desta seção, não entra em nenhum modelo nem nas projeções
  PCA, t-SNE e UMAP.
- **`City_Type`:** Rural 19,34%, Suburban 18,13%, Urban 16,07%. A amplitude é de **3,27 p.p.**, efeito
  pequeno.

![Current_Car_Type e Gender vs alvo](figures/fig09-cat-vs-alvo-3.svg)
/// caption
**Figura 9** — Proporção de `Yes` por `Current_Car_Type` e `Gender`.
///

**Conclusão (Fig. 9).** `Current_Car_Type` varia de 15,75% (Truck) a 18,12% (SUV), amplitude de
**2,37 p.p.**, e `Gender` varia de 17,25% (Male) a 17,76% (Other), apenas **0,51 p.p.** Nenhuma das duas
tem poder preditivo relevante sozinha. `Gender = Other` tem só 4.250 linhas (0,8% do treino), então a taxa dessa
categoria é a menos estável.

### 5.C Numérica × categórica

![Annual_Income_USD por City_Type](figures/fig10-box-annual-income-usd.svg)
/// caption
**Figura 10** — `Annual_Income_USD` por `City_Type` (treino).
///

**Conclusão (Fig. 10).** As medianas são praticamente iguais: Rural 84.768, Suburban 85.475 e
Urban 84.748 (diferença máxima de 727, menos de 1%). O espalhamento difere pouco: o IQR é de 38.422 em
Rural contra 34.738 em Suburban e 34.091 em Urban (de 10% a 13% maior em Rural). Os grupos **não diferem em
posição e diferem levemente em espalhamento**, com a zona rural mais dispersa. A renda não depende do
tipo de cidade.

![Daily_Commute_km por Current_Car_Type](figures/fig11-box-daily-commute-km.svg)
/// caption
**Figura 11** — `Daily_Commute_km` por `Current_Car_Type` (treino).
///

**Conclusão (Fig. 11).** As medianas ficam entre 33,0 km (Sedan) e 34,2 km (SUV), amplitude de 1,2 km. O IQR
vai de 29,40 (Hatchback) a 32,57 (Truck), e o desvio-padrão de 18,56 a 19,20. Os grupos **não diferem em
posição e quase não diferem em espalhamento**; Truck é o mais disperso. Os 21,59% de linhas no
piso de 5,0 km estão em todos os grupos e achatam o limite inferior das caixas.

![Annual_Income_USD por Will_Buy_EV](figures/fig12-box-annual-income-usd.svg)
/// caption
**Figura 12** — `Annual_Income_USD` por `Will_Buy_EV` (treino).
///

**Conclusão (Fig. 12).** A mediana da renda é **95.966** para `Yes` (93.423 linhas) e **82.823** para `No` (441.509 linhas), uma diferença de 13.143 (cerca de 16% maior entre compradores). O espalhamento é parecido: IQR de 34.987 contra 33.258 e desvio-padrão de 26.273 contra 28.217. Os grupos diferem **em posição, não em espalhamento**: quem compra tem renda mediana mais alta, mas a variabilidade é a mesma. Isso é coerente com ρ = +0,224 entre renda e alvo (Seção 5.A), uma associação real porém moderada, já que as duas caixas se sobrepõem bastante.

### 5.D Síntese da seção

- **Sem redundância forte entre numéricas.** O maior ρ é 0,543 (estações de carregamento em casa e no
  trabalho), abaixo de 0,7; nenhuma numérica é removida por correlação (Fig. 5).
- **O sinal numérico está em `Environmental_Concern_Level` (ρ = +0,461) e `Annual_Income_USD`
  (ρ = +0,224).** Compradores têm renda mediana de 95.966 contra 82.823 dos não compradores (Fig. 12).
  As demais numéricas têm |ρ| ≤ 0,044 com o alvo.
- **`Range_Anxiety_Level` é descartada** (derivada do alvo): 0,11% de `Yes` em High e 4,18% em Medium
  (Fig. 8).
- **`Subsidy_Available` é a categórica mais forte** (0,57% contra 27,47%, Fig. 7), mas não é derivada
  do alvo; será testada com e sem na Classificação.
- **`Gender`, `City_Type` e `Current_Car_Type` têm efeito pequeno** (amplitudes de 0,51, 3,27 e 2,37 p.p.,
  Figs. 8 e 9), e renda e deslocamento não variam com cidade ou carro (Figs. 10 e 11).
- **Implicação para a modelagem:** o desbalanceamento (17,46% de `Yes`) e a dependência de poucas features
  (`Subsidy_Available`, `Environmental_Concern_Level`, `Annual_Income_USD`) são os riscos principais.
  Comparar o desempenho com e sem `Subsidy_Available` mostra quanto o modelo depende dela.



## 6. Qualidade dos dados

Todos os números desta seção são do conjunto de **treino** (534.932 linhas, 15 colunas incluindo `id` e o alvo).
O teste só entra na checagem de categorias novas.

### 6.1 Valores ausentes

| Feature | Ausentes | % ausente | Padrão | Tratamento planejado |
|---------|----------|-----------|--------|----------------------|
| Todas as 15 colunas | 0 | 0,00% | não se aplica | nenhum (o pipeline mantém um `SimpleImputer` por segurança) |

**Nenhuma coluna tem valores ausentes**, então a coluna com mais faltantes tem 0 linhas (0,00%).
Isso não significa que os dados estejam completos: os pisos de renda e deslocamento (Seção 6.3) não aparecem como
`NaN` e foram investigados à parte.

### 6.2 Duplicatas e inconsistências

| Verificação | Resultado |
|-------------|-----------|
| Linhas duplicadas (todas as colunas) | 0 (0,00%) |
| Linhas duplicadas ignorando o `id` (features + alvo) | 0 (0,00%) |
| `id` | 534.932 valores únicos em 534.932 linhas (todos distintos) |
| Colunas constantes | nenhuma |
| Valores impossíveis (idade fora de 18–110, renda ≤ 0, deslocamento < 0 ou > 500 km, contagens < 0, `Environmental_Concern_Level` fora de 1–5) | 0 linhas em todas as regras |
| Valores não inteiros em colunas de contagem (`Age`, `Number_of_Cars_Owned`, `Charging_Stations_*`, `Environmental_Concern_Level`) | 0 |
| Categóricas com grafias duplicadas (após `strip` e `lower`) ou placeholders (`Unknown`, `N/A`, `?`) | nenhuma |
| Categorias no teste ausentes do treino | nenhuma, nas 6 categóricas |

Os limites de "valor impossível" são nossos: [citar a fonte, se a página do Kaggle descrever as faixas válidas; se não,
dizer que são limites de bom senso]. O fato de nenhuma categoria nova aparecer no teste não elimina `handle_unknown="ignore"`
no encoding: mantemos para o caso de o modelo receber dados novos, como o `test.csv` da competição.

### 6.3 Valores no piso

`Annual_Income_USD` tem **49.191 linhas (9,20%)** exatamente em 30.000, e `Daily_Commute_km` tem
**115.506 linhas (21,59%)** exatamente em 5,0 km (Seção 4). Nenhuma das duas colunas tem `NaN`. Testamos a hipótese de
que esses valores sejam respostas ausentes disfarçadas no mínimo da escala, comparando a taxa de `Yes` dentro e fora do piso:

| Feature | Linhas no piso | `Yes` no piso | `Yes` fora do piso | `Yes` geral |
|---------|----------------|---------------|--------------------|-------------|
| `Annual_Income_USD` (30.000) | 49.191 (9,20%) | **4,41%** | 18,79% | 17,46% |
| `Daily_Commute_km` (5,0 km) | 115.506 (21,59%) | 18,43% | 17,20% | 17,46% |

Nos dois pisos ao mesmo tempo estão 11.018 linhas (2,06%). Se os dois pisos fossem independentes, esperaríamos
cerca de 10.600. Em pelo menos um piso estão 153.679 linhas (28,73%).

**O que isso mostra.**

- **Renda.** A taxa de `Yes` no piso (4,41%) é quatro vezes menor que fora dele (18,79%). Uma não-resposta aleatória
  teria taxa perto da geral (17,46%). O grupo do piso se comporta como pessoas de renda baixa, o que é coerente com a
  correlação de +0,224 entre renda e alvo (Seção 5.A) e com valores truncados no mínimo. Portanto **esse grupo é informativo, e
  apagar seu valor com a mediana destruiria sinal.**
- **Deslocamento.** A taxa no piso (18,43%) é próxima da de fora (17,20%), e o deslocamento quase não se relaciona com o
  alvo (ρ = −0,044). Não há evidência de que o piso seja diferente das demais linhas.
- **Independência dos pisos.** O número de linhas nos dois pisos (11.018) é quase o esperado sob independência (cerca de
  10.600). Isso é contra a ideia de um respondente que pulou vários campos de uma vez.

**Conclusão e revisão do achado da Seção 4.** A hipótese de "não-resposta disfarçada" **não se sustenta para a renda**
e não tem apoio para o deslocamento. O mais provável é que os valores tenham sido truncados no mínimo da escala. Isso é
uma interpretação, não uma prova: não temos o gerador do dataset. Tratamento proposto (Seção 8, estratégia de faltantes e
outliers): **manter os valores originais no piso e acrescentar um indicador binário "no piso" para cada uma das duas
colunas**, sem imputação por mediana.

### 6.4 Outliers

Método: **IQR com k = 1,5** (limites Q1 − 1,5·IQR e Q3 + 1,5·IQR), calculado no treino.

| Feature | Limites | Outliers | % do treino |
|---------|---------|----------|-------------|
| `Age` | [3,00; 91,00] | 0 | 0,00% |
| `Annual_Income_USD` | [14.323,50; 155.807,50] | 2.926 | 0,55% |
| `Daily_Commute_km` | [−28,10; 92,70] | 37 | 0,01% |
| `Number_of_Cars_Owned` | [−0,50; 3,50] | 10.725 | 2,00% |
| `Charging_Stations_Near_Home` | [−5,50; 14,50] | 0 | 0,00% |
| `Charging_Stations_Near_Work` | [−7,50; 20,50] | 0 | 0,00% |
| `Environmental_Concern_Level` | [−1,00; 7,00] | 0 | 0,00% |

- **Renda e deslocamento:** são contínuas, e os outliers são caudas altas (renda máxima 188.549, deslocamento máximo
  98,7 km), sem valores absurdos. **Winsorizamos** nos limites do IQR, em vez de remover linhas: são poucas (0,55% e 0,01%) e o
  `StandardScaler`, que usaremos por causa da rede neural, é sensível a caudas. Remover linhas seria uma decisão de
  modelagem, e não vemos erro de dado que a justifique.
- **`Number_of_Cars_Owned`:** os 10.725 "outliers" são todos os que têm **4 carros**. A regra do IQR não se aplica a uma variável
  discreta com 4 valores e IQR de 1: o valor 4 é legítimo. **Não tratamos.**
- **Idade, estações e preocupação ambiental:** nenhum outlier.
- **Linhas afetadas pela estratégia** (item 8 da tabela final): [preencher com o `preprocess_output.txt`].

### 6.5 Colunas descartadas

| Coluna | Motivo | Evidência |
|--------|--------|-----------|
| `id` | Identificador, não é feature | 534.932 valores únicos em 534.932 linhas |
| `Range_Anxiety_Level` | Derivada do alvo (vazamento) | [fonte]; `Yes` em 18,90% (Low), 4,18% (Medium) e 0,11% (High), Seção 5.B |
| (nenhuma constante) | | 0 colunas constantes |

`Subsidy_Available` **não** é descartada: tem relação muito forte com o alvo (0,57% contra 27,47% de `Yes`), mas não é
derivada dele. Será treinada com e sem a coluna na entrega de Classificação (Seção 5.B).


## 7. Riscos de vazamento

| Risco | Onde aparece | Evidência | Contenção |
|-------|--------------|-----------|-----------|
| Estatísticas calculadas antes do split | Qualquer estatística (média, desvio, quantis) usada pra normalizar/imputar | — | Ajustar (`fit`) transformadores só no treino (Seção 8) |
| `Subsidy_Available` — forte preditor, talvez suspeita de vazamento | Taxa de `Will_Buy_EV`: 27,5% (`Yes`) vs 0,5% (`No`) — quase separação total | A descrição do dataset original no Kaggle lista "disponibilidade de subsídio" como fator econômico do comprador (junto com preocupação ambiental), não como resultado da compra — reduz a suspeita de vazamento literal, pode ser apenas uma feature de forte impacto | Manter como feature, mas documentar a ressalva; monitorar se o modelo de classificação depende dela de forma desproporcional |
| `Range_Anxiety_Level` — suspeita de vazamento | Taxa de `Will_Buy_EV`: 18,8% (`Low`) vs 4% (`Medium`) vs 0,1% (`High`) — quase separação total (Seção 5) | A descrição do dataset original trata `Range_Anxiety_Level` como um **segundo alvo calculado**, derivado do mesmo processo que gera `Will_Buy_EV, não sendo uma feature de entrada independente | Iremos testar o modelo de classificação com e sem essa feature. Se a performance cair demais sem ela, é sinal de que o "aprendizado" vinha do vazamento, não dos dados |

## 8. Plano de pré-processamento

A saída desta entrega. Uma linha por transformação, ligando cada uma a um achado acima.

| # | Transformação | Features | Motivo (seção) |
|---|---------------|----------|----------------|
| 1 | | | |

## 9. Estratégia de split

- **Proporção:** 80% treino / 20% teste — 534.932 amostras de treino, 133.733 de teste
- **Estratificação:** por `Will_Buy_EV`, preservando a proporção de classes (82,54% `No` / 17,46% `Yes`) em ambos os conjuntos
- **Semente fixa:** `random_state=42`, usada para garantir reprodutibilidade
- **Agrupamento:** cada linha representa um comprador potencial distinto indicado por `id` que tem 534.932 valores únicos, batendo exatamente com o número de linhas do treino, então não há repetição da mesma entidade (sem coluna de usuário, sessão ou data que indicasse necessidade de agrupamento)
- **Ordem das operações:** o split foi feito **antes** de qualquer estatística de pré-processamento (médias, desvios, quantis usados nas Seções 4, 6 e 8 vêm só do conjunto de treino), evitando vazamento entre treino e teste

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
