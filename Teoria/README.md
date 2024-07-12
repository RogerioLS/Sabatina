### **Data Prep**

#### `Tratamento de missing`

Antes mesmo de entender o que é tratamento de missing temos que entender o que pode gerar esses dados ou do porque existem missing?
Dando uma resposta bem pragmática, os missing podem acontecer por dois fatores, `erro técnico` ou `erro humano`.
Lendo um [medium](https://medium.com/datapsico/valores-missing-parte-1-4382bb026660) bem interessante falando do tipos de missing:

- `Missings completamente aleatórios (MCAR)`
Os completamente aleatórios ocorreram por erro técnico ou obra do acaso. Algo que se perdeu no caminho e do qual não fazemos ideia do porquê.
- `Missings aleatórios (MAR)`
Os aleatórios existem porque a existência de uma outra variável aumenta a probabilidade dessa resposta não existir.
- `Missings não aleatórios (MNAR)`
Os não aleatórios não são aleatórios - existem porque a variável em si é a causa de não existir um escore dela mesma.

Com isso esses tipos de missings podem ser importante para entendermos como fazermos uma boa imputação desses valores faltantes seja em nosso banco de dados ou em nosso modelos.

> [!IMPORTANT]  
> Você sabia?
> 
> - NA significa “No answer” (“Sem resposta”) ou “Not applicable” (“Não se aplica”).

Voltando para a nossa abordagem inicial, como podemos tratar missings?
Na literatura ou no nosso dia a dia não existe uma bala de prata que vá trazer a solução perfeita para todos os casos, por isso vale muito aplicar todos os métodos possíveis e ir avaliando qual melhor se encaixa para o contexto da analise ou modelo de ML, por isso vou sitar alguns:
- `Remoção de linhas:` Excluir todas as linhas que contêm valores missing. Talvez seja útil quando a quantidade de dados faltantes é pequena.
- `Remoção de colunas:` Excluir colunas inteira se a maioria dos valores estiverem faltando.
- `Imputação com valor constante medidas de tendência central e medidas de dispersão:` Substitui valores missing por uma constante, como zero ou a média/mediana/moda, amplitude(A)/intervalo-interquartil/variância/desvio-padrão/coeficiente de variação.
- `Imputação baseada em k-vizinhos mais próximos (KNN):` Usa valores dos k-vizinhos mais próximos para imputar valores missing.
- `Imputação multivariada (MICE):` Usa modelos estatísticos para prever valores missing com base em outras variáveis.
- `Interpolação:` Usa métodos de interpolação linear ou polinomial para preencher valores missing em séries temporais.

Essas são os possíveis métodos que podemos utilizar para se tratar os missings, lembrando que podem existir casos que ainda não foram listados nesse resumo, e que o tratamento de missing vai de acordo com o que você está modelando.

#### `Tratamento de outliers`

Um ponto importante é que antes de falarmos de tratamento de outliers, vale dizer o que são outliers?
Em uma definição bem abrangente podemos dizer que outliers, são observações ou pontos de dados que se encontram significativamente afastados da maioria dos dados em um conjunto de dados. Eles podem surgir devido a variabilidade inerente ao sistema, erros de medição ou colta de dados, ou devido á presença de fenômenos anômalos. A identificação de outliers é crucial, pois eles podem distorcer a análise estatística e influenciar significativamente os resultados de modelos de machine learning.

Seguindo essa lógica de rastreabilidade de outliers temos os tipos:
- `Univariados:` Anomalias que ocorrem em uma única variável.
- `Multivariados:` Anomalias que ocorrem em um contexto multidimensional.
- `Artificiais:` Anomalias que ocorrem em falhas de medição, inferência erradas, ou falhas de processamento de dado.
- `Naturais:` Anomalias que ocorrem quando o dado é somente uma exceção atípica e discrepante em relação ao conjunto de observações.

> [!IMPORTANT]  
> Essas são possíveis características de outliers:
> - ``Distância:`` Eles estão a uma distância considerável dos outros pontos de dados.
> - `Raridade:` São raros ou incomuns dentro do conjunto de dados.
> - `Influência:` Podem ter um impacto desproporcional em estatísticas como média e variância.

Pensando na nossa abordagem principal, como podemos realizar tratamento de outliers?
Você vai me ver falar muito de frase `Não existe bala de prata` tudo depende do contexto que você está trabalhando, então o que sempre sugiro é testar.

E para se tratar os outliers encontrado na sua base de dados, aqui temos alguns métodos utilizados.

- `Mediana:` Substituir outliers pela mediana dos dados.
```python
median = df['feature'].median()
df['feature'] = np.where(np.abs(stats.zscore(df['feature'])) > 3, median, df['feature'])

```
- `Desvio Padrão:` Essa medida de dispersão de um conjunto de dados diz que quanto menor o desvio padrão mais homogêneos são os dados e ao contrário disso quanto maior o desvio mais os dados espalhados estão.
Outra informação importante é sobre o teorema das probabilidades que afirma que quanto maior a amostra, mais a distribuição amostral da sua média aproxima-se de uma distribuição normal.
Quando consideramos valores que são distribuídos de acordo com a curva de Gauss podemos fazer algumas inferências importantes para tratarmos os dados. Uma dessas inferências é que 96,6% dos dados estão contidos em 3 desvios padrões e o que está fora desse intervalo de confiança pode ser considerado como sendo um outlier.
<div align="center">

![Desvio Padrão](https://miro.medium.com/v2/resize:fit:300/1*RULOf2Y0yGyqbb09V8xMjQ.png)

</div>

- `Z-score:` nos diz o quanto cada valor em nossa distribuição se distancia da média em termos de desvio padrão.

<div align="center">

![Z-score](https://miro.medium.com/v2/resize:fit:219/1*Af8Wm1o8y7iBJjOGTPXO2A.png)![Z-score-imple](https://miro.medium.com/v2/resize:fit:449/1*KZja2D6znAtjxvxeOJN2Tg.png)

</div>

- `Desvio absoluto mediano:` um conjunto de dados é a distância entre cada dado em relação a mediana.

<div align="center">

![MDAM](https://miro.medium.com/v2/resize:fit:361/1*rP_fYzL3F3Ztr_Cxify4Uw.png)![MDAM_imple](https://miro.medium.com/v2/resize:fit:567/1*Wne1ME1Q2Yx8gxctLdOueA.png)

</div>

- `Método de Tukey:` o método de Tukey ou bloxplot consiste em definir os limites inferior e superior a partir do interquartil (IQR) e dos primeiros (Q1) e terceiros (Q3) quartis.
Os quartis são separatrizes que dividem um conjunto de dados em 4 partes iguais. O objetivo das separatrizes é proporcionar uma melhor ideia da dispersão do conjunto de dados, principalmente da simetria ou assimetria da distribuição.
O limite inferior é definido pelo primeiro quartil menos o produto entre o valor 1.5 e o interquartil.
$$𝐿𝑖𝑛𝑓 = 𝑄1 − (1.5 ∗ 𝐼𝑄𝑅)$$
O limite superior é definido pelo terceiro quartil mais o produto entre o valor 1.5 e o interquartil.
$$𝐿𝑠𝑢𝑝 = 𝑄3 + (1.5 ∗ 𝐼𝑄𝑅)$$

<div align="center">

![quartil_imple](https://miro.medium.com/v2/resize:fit:490/1*W8E3Ek_dmaISZBOyTAmrMA.png)

</div>

- `Isolation Forest:` Foi o primeiro algoritmo de detecção de anomalias usando a estratégia de isolamento.
Esse algoritmo é basicamente uma floresta aleatória em que cada arvore de decisão é cultivada aleatoriamente. As arvores irão dividir e subdividir os dados baseado em um valor aleatório de corte até que todos os dados eventualmente estejam todos cortados e separados. Os dados mais discrepantes serão isolados mais rapidamente do que os demais, podendo assim ser identificado como um outlier.

<div align="center">

![isolation](https://miro.medium.com/v2/resize:fit:511/1*WSnHJrbeZefheqjxbno2jw.png)
![isolation_imple](https://miro.medium.com/v2/resize:fit:461/1*ODyLL3OaDe_KPX6rjD0poQ.png)
</div>


#### `Categorização de variáveis contínuas e discretas`

Categorização para essas tipos de variáveis são o processo de transformar variáveis contínuas ou discretas em variáveis categóricas.

Variáveis contínuas são aquelas que podem assumir qualquer valor dentro de um intervalo, com por exemplo altura, peso, temperatura e tempo. Formas para se categorizar variáveis contínuas:
- `Binning (Discretização):` É o processo de dividir a faixa de valores contínuos em intervalos (bins) e atribuir um rótulo categórico a cada intervalo.
- `Equal-width Binning:` Divide os dados em intervalos de igual largura.
- `Equal-frequency Binning:` Divide os dados de modo que cada intervalo contenha aproximadamente o mesmo número de pontos de dados.
- `Custom Binning:` Utiliza intervalos definidos pelo usuário com base no conhecimento do domínio ou características dos dados.

```python
import pandas as pd

# Supondo um DataFrame com uma variável contínua 'idade'
df = pd.DataFrame({'idade': [23, 45, 12, 67, 34, 25, 37, 50, 29, 61]})

# Equal-width Binning
df['idade_binned'] = pd.cut(df['idade'], bins=3, labels=["Jovem", "Adulto", "Idoso"])

# Equal-frequency Binning
df['idade_binned_freq'] = pd.qcut(df['idade'], q=3, labels=["Jovem", "Adulto", "Idoso"])

# Custom Binning
bins = [0, 18, 35, 50, 100]
labels = ["Criança/Adolescente", "Jovem Adulto", "Adulto", "Idoso"]
df['idade_custom_binned'] = pd.cut(df['idade'], bins=bins, labels=labels)

```

Variáveis discretas são aquelas que assumem valores distintos e finitos, como o número de filhos, contagem de produtos vendidos, ou número de eventos. Uma forma para se categorizar variáveis discretas:
- `Agrupamento:` Agrupar valores discretos em categorias mais amplas. Pode ser feito de forma similar ao binning para variáveis contínuas, mas geralmente com base em lógica de negócio ou conhecimento do domínio.
```python
# Supondo um DataFrame com uma variável discreta 'num_filhos'
df = pd.DataFrame({'num_filhos': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Agrupamento customizado
bins = [0, 2, 4, 6, 10]
labels = ["Nenhum ou Poucos", "Alguns", "Muitos", "Grandes Famílias"]
df['num_filhos_binned'] = pd.cut(df['num_filhos'], bins=bins, labels=labels)
```

> [!IMPORTANT]  
> Variáveis quantitativas podem ser classificadas como discretas ou contínuas:
> - `Variáveis Discretas:` Assumem valores específicos e contáveis (ex.: número de filhos).
> - `Variáveis Contínuas:` Podem assumir qualquer valor dentro de um intervalo (ex.: altura, peso).
> - `Variáveis Categóricas:` Representam categorias ou grupos:
>	- `Nominais:` Sem ordem (ex.: cor dos olhos).
>	- `Ordinais:` Com ordem (ex.: classificação de serviço).

#### `PCA`

A Análise de Componentes Principais, ou PCA, é uma técnica fundamental no domínio da análise de dados e aprendizado de máquina.
O PCA é um método estatístico que transforma dados de alta dimensão em um formato de menor dimensão, preservando as informações mais importantes. Isto é conseguido através da identificação de novos eixos, chamados componentes principais, ao longo dos quais os dados variam mais. Esses componentes são ortogonais entre si, o que significa que não estão correlacionados, o que os torna uma ferramenta poderosa para redução de dimensionalidade.

E quando podemos utilizar esse método:
Imagine que você tem um conjunto de dados com muitos recursos ou variáveis. Cada recurso contribui para a complexidade geral dos dados, tornando difícil analisar, visualizar ou construir modelos. A alta dimensionalidade pode levar a vários problemas:

- `Complexidade Computacional:` À medida que o número de recursos aumenta, os recursos computacionais e o tempo necessário para análise e modelagem crescem exponencialmente.
- `Overfitting:` Os modelos podem se tornar excessivamente complexos e ajustar o ruído nos dados, levando a uma generalização deficiente em dados novos e não vistos.
- `Dificuldade de visualização:` Torna-se um desafio visualizar e compreender dados em mais de três dimensões.
- `Redundância:` alguns recursos podem estar altamente correlacionados, o que significa que transmitem informações semelhantes. Essa redundância pode ser eliminada sem perda significativa de informações.

Seguindo o raciocínio leva três etapas para se fazer o calculo:
- `Padronização de Dados:` antes de realizar o PCA, é fundamental padronizar os dados. Isso significa centralizar os dados subtraindo a média e escalá-los dividindo pelo desvio padrão. A padronização garante que todos os recursos tenham igual importância na análise.

- `Matriz de Covariância:` o PCA depende do cálculo da matriz de covariância. A covariância entre duas variáveis ​​mede como elas mudam juntas. A matriz de covariância para um conjunto de dados com n recursos é uma matriz nxn que resume as relações entre todos os pares de recursos.

- `cálculo de autovalor e autovetor:` a próxima etapa é calcular os autovalores e autovetores da matriz de covariância. Esses autovalores representam a quantidade de variância explicada por cada autovetor (componente principal). Autovalores e autovetores são conceitos matemáticos relacionados a transformações lineares e matrizes. No contexto da PCA, desempenham um papel central na identificação dos componentes principais. Aqui está o que eles significam:
	- `Autovalor:` Um autovalor (λ) representa um escalar que indica quanta variância é explicada pelo autovetor correspondente. No PCA, os autovalores quantificam a importância de cada componente principal. São sempre não negativos e o autovalor correspondente a um componente principal mede a proporção da variância total nos dados explicada por esse componente.
	- `Autovetor:` Um autovetor (v) é um vetor associado a um autovalor. No PCA, os autovetores representam as direções ao longo das quais os dados mais variam. Cada autovetor aponta em uma direção específica no espaço de recursos e corresponde a um componente principal. Os autovetores são normalmente normalizados, o que significa que seu comprimento é 1.


Para trazer mais clareza vamos calcular o PCA (Análise de Componentes Principais) passo a passo usando um exemplo simples. Suponha que temos um pequeno conjunto de dados com duas variáveis (X e Y):

\[
\begin{array}{|c|c|c|}
\hline
   & X  & Y  \\
\hline
 1 & 2  & 4  \\
 2 & 3  & 5  \\
 3 & 5  & 7  \\
 4 & 6  & 8  \\
 5 & 8  & 9  \\
\hline
\end{array}
\]


`Passo 1: Centralizar os dados`

Primeiro, centralizamos os dados subtraindo a média de cada variável.

- Média de X: (2 + 3 + 5 + 6 + 8) / 5 = 4.8
- Média de Y: (4 + 5 + 7 + 8 + 9) / 5 = 6.6

\[
\begin{array}{|c|c|c|c|c|}
\hline
   & X & Y & \text{X Centralizado} & \text{Y Centralizado} \\
\hline
1 & 2 & 4 & 2 - 4.8 = -2.8 & 4 - 6.6 = -2.6 \\
2 & 3 & 5 & 3 - 4.8 = -1.8 & 5 - 6.6 = -1.6 \\
3 & 5 & 7 & 5 - 4.8 =  0.2 & 7 - 6.6 =  0.4 \\
4 & 6 & 8 & 6 - 4.8 =  1.2 & 8 - 6.6 =  1.4 \\
5 & 8 & 9 & 8 - 4.8 =  3.2 & 9 - 6.6 =  2.4 \\
\hline
\end{array}
\]


`Passo 2: Calcular a matriz de covariância`

A matriz de covariância é calculada a partir dos dados centralizados.

\[ 
\begin{bmatrix}
\text{Cov(X, X)} & \text{Cov(X, Y)} \\
\text{Cov(Y, X)} & \text{Cov(Y, Y)}
\end{bmatrix}
\]

Para este exemplo, a matriz de covariância é:

- Cov(X, X) = \(\frac{1}{4} \sum_{i=1}^{5} (X_i - \bar{X})^2\)
- Cov(X, Y) = \(\frac{1}{4} \sum_{i=1}^{5} (X_i - \bar{X})(Y_i - \bar{Y})\)
- Cov(Y, Y) = \(\frac{1}{4} \sum_{i=1}^{5} (Y_i - \bar{Y})^2\)

\[
\begin{array}{|c|c|c|c|c|}
\hline
\text{X Centralizado} & \text{Y Centralizado} & (X - \bar{X})(X - \bar{X}) & (X - \bar{X})(Y - \bar{Y}) & (Y - \bar{Y})(Y - \bar{Y}) \\
\hline
-2.8 & -2.6 & 7.84 & 7.28 & 6.76 \\
-1.8 & -1.6 & 3.24 & 2.88 & 2.56 \\
 0.2 &  0.4 & 0.04 & 0.08 & 0.16 \\
 1.2 &  1.4 & 1.44 & 1.68 & 1.96 \\
 3.2 &  2.4 & 10.24 & 7.68 & 5.76 \\
\hline
\text{Total} & \text{Total} & 22.8 & 19.6 & 17.2 \\
\hline
\end{array}
\]

- Cov(X, X) = $\frac{22.8}{4} = 5.7$
- Cov(X, Y) = $\frac{19.6}{4} = 4.9$
- Cov(Y, Y) = $\frac{17.2}{4} = 4.3$

Matriz de covariância:

\[ 
\begin{bmatrix}
5.7 & 4.9 \\
4.9 & 4.3
\end{bmatrix}
\]

`Passo 3: Calcular os autovalores e autovetores`

Calculamos os autovalores e autovetores da matriz de covariância.

- Autovalores: λ1 = 10.1, λ2 = 0
- Autovetores correspondentes:

\[ 
\mathbf{v_1} = \begin{bmatrix}
0.79 \\
0.61
\end{bmatrix}
\]

\[ 
\mathbf{v_2} = \begin{bmatrix}
-0.61 \\
0.79
\end{bmatrix}
\]

`Passo 4: Projetar os dados nos componentes principais`

Multiplicamos os dados centralizados pelos autovetores para obter as componentes principais.

\[
\begin{bmatrix}
\text{PC1} \\
\text{PC2}
\end{bmatrix}=
\begin{bmatrix}
0.79 & 0.61 \\
-0.61 & 0.79
\end{bmatrix}
\begin{bmatrix}
X_{\text{centralizado}} \\
Y_{\text{centralizado}}
\end{bmatrix}
\]

Calculando para cada ponto:

- Para (2, 4): PC1 = 0.79*(-2.8) + 0.61*(-2.6) = -3.6
- Para (3, 5): PC1 = 0.79*(-1.8) + 0.61*(-1.6) = -2.3
- Para (5, 7): PC1 = 0.79*(0.2) + 0.61*(0.4) = 0.5
- Para (6, 8): PC1 = 0.79*(1.2) + 0.61*(1.4) = 2.1
- Para (8, 9): PC1 = 0.79*(3.2) + 0.61*(2.4) = 5.3

Assim, temos os dados projetados nos novos componentes principais.

> [!IMPORTANT]  
> Vale lembrar que todo essa tratamento pode acaber gerando o que chamamos de [Maldição da dimensionalidade](https://medium.com/data-hackers/maldi%C3%A7%C3%A3o-da-dimensionalidade-655e4342d64).
> A Maldição da Dimensionalidade foi denominada pelo matemático R. Bellman em seu livro “Programação Dinâmica” em 1957. Segundo ele, a maldição da dimensionalidade é o problema causado pelo aumento exponencial do volume associado à adição de dimensões extras ao espaço euclidiano.

![Aumento de dimensões gera dados mais esparsos](https://miro.medium.com/v2/resize:fit:875/1*FxCli5YAOuMPxHR_Xwe1xg.png)


#### `Correlação / associação entre dados contínuos e entre dados discretos`

`Correlação x Associação`
De forma simples podemos pensar nas duas como a mesma “coisa”, porém estatisticamente não é “correto”. Quando falamos do relacionamento entre variáveis numéricas buscamos correlação e quando falarmos sobre categóricas e numéricas e/ou categóricas e categóricas buscamos associação.
Quando estamos avaliando duas variáveis, temos que lembrar que é conhecida como análise bivariada.
- Análise da relação entre duas variáveis, conhecida como análise bivariada;
- Medidas de correlação para análise da relação entre variáveis numéricas;
- Medidas de associação para análise da relação entre variáveis categóricas;

Um desenho ilustrativo para auxiliar o entendimento do fluxo de medidas de acordo com os dados observados.

![](https://miro.medium.com/v2/resize:fit:875/1*QGpZwlAnOvHP0PY_VIM_8A.png)

`Medidas de correlação`

- `Covariância:` como medida de variação conjunta entre variáveis numéricas.
A covariância é calculada pela fórmula cov(X,Y) = Σ(xi – x̄)(yi – ȳ) / (n-1), levando em conta os desvios de cada observação em relação à média, medindo o quanto esses desvios variam conjuntamente entre as duas variáveis.

- `Coeficiente de correlação de Pearson:` provavelmente o mais familiar para nós é a correlação de Pearson, que tem como objetivo medir o grau de correlação entre duas variáveis que possuem uma relação linear. <div align="center"> ![](https://miro.medium.com/v2/resize:fit:743/1*4KfFijGsdKSBUku77FTVmg.png) </div>
O r varia de -1 até 1, onde 1 é uma correlação positiva perfeita, -1 correlação negativa perfeita e 0 não existe nenhuma correlação linear entre as variáveis, abaixo desenhei alguns exemplos para que fique mais claro: <div align="center"> ![](https://miro.medium.com/v2/resize:fit:875/1*FRBKVBFxn2a2dIgOlzbwpw.png) </div>
Supondo que encontramos uma correlação de pearson de -1, isso significa dizer que a medida que a variável x aumenta a variável y diminui proporcionalmente, ou seja uma influencia negativamente na outra. 
	```python
	# Correlação de pearson entre duas variáveis 
	import pandas as pd
	df[['coluna X','coluna Y']].corr()
	```

- `Correlação de Spearman:` Spearman é basicamente uma variação de pearson, com diferença na forma como calculamos, tendo basicamente que ordenar as observações criando um novo ‘dataset’ antes de calcular o coeficiente.
Spearman é o que chamamos de não paramétrico ou seja diferente da regressão linear não pressupõe nada sobre a distribuição dos dados. <div align="center"> ![](https://miro.medium.com/v2/resize:fit:405/1*ywLSUs2QbR4OmlV5YWz34A.png) </div>
	```python
	# Correlação de spearman entre duas variáveis 
	# corr(method='spearman')
	import pandas as pd
	df[['coluna X','coluna Y']].corr(method='spearman')
	```

`Medidas de associação`

- `Coeficiente de Cramér's V:` Utilizaremos o Cramers’V, quando estivermos falando da associação entre duas variáveis categóricas.
O resultado advindo do teste Cramers’V , varia de 0 até 1, onde 0 indica que não existe nenhuma associação entre as variáveis, enquanto 1 indica uma associação perfeita.

  \[
  V = \sqrt{\frac{\chi^2}{n \times \min(k-1, r-1)}}
  \]

  Onde \(\chi^2\) é o valor do teste qui-quadrado, \(n\) é o tamanho da amostra, \(k\) é o número de colunas e \(r\) é o número de linhas.
  - \( V = 0 \): Nenhuma associação.
  - \( V = 1 \): Associação perfeita.
	```python
	from scipy.stats.contingency import association
	#1- Criar tabela de contingência:
	df_cont_cat = pd.crosstab(df['tech_company'],df['treatment'])
	association(df_cont_cat, method="cramer")
	```

- `Coeficiente de Contingência:`
  \[
  C = \sqrt{\frac{\chi^2}{\chi^2 + n}}
  \]
Onde \(\chi^2\) é o valor do teste qui-quadrado e \(n\) é o tamanho da amostra.
Varia entre 0 e 1, onde valores mais próximos de 1 indicam uma associação mais forte.

`Correlação entre Dados Contínuos e Discretos`

Quando temos uma variável contínua e uma variável discreta, o método mais comum é usar ``ANOVA`` (Análise de Variância) ou a `correlação ponto-bisserial`.

- `Ponto-Bisserial:` o coeficiente de correlação ponto-bisserial mede a associação entre uma variável númerica e contínua, por exemplo: Doença (Sim ou Não) em relação ao nível de açucar no sangue. Ela varia de -1 a 1 seguindo a mesma lógica de interpretação das anteriores. <div align="center"> ![](https://miro.medium.com/v2/resize:fit:689/0*KkfWBoYkfe2vHi9w.png) </div>
	```python
	from scipy import stats
	#precisamos converter a coluna de qualidade de string para número, vamos colocar 1 -good e 0 -bad
	df2['Quality_encoded'] = np.where(df2['Quality']=='good',1,0)
	stats.pointbiserialr(df2['Quality_encoded'], df2['Weight'])
	```

- `Análise de Variância (ANOVA):` A análise de variância mais conhecida como ANOVA é um teste estatístico bastante utilizado nas pesquisas, seja em áreas biológicas e de saúde ou exatas. Esse teste estatístico tem como objetivo verificar a diferença entre médias de três ou mais grupos baseado na análise das variâncias amostrais.
Ou seja, a ANOVA permite identificar se esses grupos possuem diferenças estatísticas significativas ou não a partir da comparação de vários grupos que avalia se há diferenças estatisticamente significativas entre as médias de três ou mais grupos (categorias da variável discreta).
Nesse exemplo temos a variável explanatória categórica (X) e a variável resposta contínua (Y). Observe na tabela abaixo os dados do experimento:  <div align="center"> ![](https://blog.proffernandamaciel.com.br/wp-content/uploads/2022/09/anova1.png) 
**Tabela 1 – Dados referente a variável comprimento (cm) dividida por quatro tratamentos de água salina com concentrações:
T4: 25% de água salina, T3: 50% de água salina, T2: 75% de água salina e T1 apenas com água de torneira.** </div>
Realizando todos os cálculos que estão no quadro 1, temos a seguinte resposta:  <div align="center"> ![](https://blog.proffernandamaciel.com.br/wp-content/uploads/2022/09/anova2.png)
**Tabela 2 – Resultado do cálculo de análise de variância (ANOVA)** </div>
Após realizamos os cálculos da análise de variância, o que temos que fazer é visitar nossas hipóteses e verificar qual se hipótese vamos rejeitar ou aceitar. As hipóteses são:
H0: Não existe diferença entre os grupos (p-valor > 0,05).
H1: Existe diferença em pelo menos dois grupos (p-valor < 0,05)
Como o p-valor foi 0,38 não rejeitamos a hipótese H0 que informa que não existem diferenças entre os grupos (p-valor > 0,05). Ou seja, as diferentes concentrações de água salina não influenciaram no comprimento das plantas.

> [!IMPORTANT]  
> **Cuidado com Correlações e Associações espúrias !**
> Aqui fica outro tema de atenção sobre correlações, em alguns momentos podemos nos deparar com correlações entre variáveis, inclusive significativas, que ocorrem apenas por um acaso e com o mínimo de análise e contexto fica nítido que são absurdas. [Este site exemplifica alguns casos](http://www.tylervigen.com/spurious-correlations).

#### `Seleção de variáveis`

Para não ficar algo redundante e trazer a mesma coisa que foi visto acima, pois seleção de variáveis está totalmente conectado com correlação e associação das variáveis.
Vou trazer alguns métodos utilizando o sklearn, é bom lembrar que muitas dessas técnicas dependem inicialmente do algoritmo que você quer usar. Por exemplo, uma feature pode ser considerada importante para um modelo que considera relacionamentos lineares — LinearSVC, Regressão Linear — , ao mesmo tempo em que não é importante para um modelo que consegue identificar relacionamentos não lineares — como Decision Trees, Random Forest, etc.

Se tiver dúvida sobre qual estimador utilizar para seu modelo, [recomendo dar uma consultada nessa cheat sheet do scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html).

- `Sklearn SelectKBest:` Selecionar features pode ser feito através de testes estatísticos univariados, como na função SelectKBest do sklearn. Esta função seleciona as K melhores features do dataset com base em um teste estatístico. No exemplo do dataset iris, podemos usar SelectKBest para selecionar as 2 melhores features:
	```python
	from sklearn.datasets import load_iris
	from sklearn.feature_selection import chi2, SelectKBest
	data = load_iris()
	X = data.data
	y = data.target
	X = SelectKBest(chi2, k=2).fit_transform(X, y)
	```
	Após esse processo, o dataset terá apenas as duas features com melhor pontuação no teste chi2. SelectKBest também funciona para problemas de regressão com testes como f_regression e mutual_info_regression. No entanto, escolher o número ideal de K pode ser difícil e empírico. O SelectPercentile, uma alternativa, seleciona um percentual X% das melhores features.
	[Saiba mais sobre SelectKBest na documentação do sklearn](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)

- `Sklearn RFE:` O resultado dele é bem consistente, e seu principal trade-off — o tempo — não chega a ser uma fator tão negativo em meus projetos.
Assim como seu nome diz — Recursive Feature Elimination — , o RFE funciona da seguinte forma: **ele irá treinar seu modelo utilizando todo seu conjunto inicial**, com todas as features e data points que vierem nele. **Após o primeiro treino, o RFE irá verificar a importância das features** — utilizando atributos como `coef_` ou `feature_importances_` — **e, recursivamente, irá remover as features menos importantes** do dataset e treinar o modelo novamente. Ele fará isso até chegar a um número ideal de features. Veja abaixo uma aplicação do RFE, onde informo que quero remover uma feature de cada vez. Ou seja, cada vez que o modelo for treinado, ele irá remover uma feature. O parâmetro `n_features_to_selectpode` ser passado para informar a quantidade de features que quer selecionar. Se ele for nulo, o RFE escolherá metade do total de features.
	```python
	from sklearn.datasets import load_iris
	from sklearn.svm import LinearSVC
	from sklearn.feature_selection import RFEdata = load_iris()
	X = data.data
	y = data.target
	model = LinearSVC()
	rfe = RFE(model, step=1).fit(X, y)
	```
	Eu gosto de utilizar o RFE em modelos que possuem atributos coef_, como SVM, mas ele também funciona bem com Ensembles — atente-se aos pontos que falei no tópico de Feature Importance. Como disse anteriormente, o principal trade-off dessa função é o tempo: se você tiver um dataset com alta dimensionalidade — muitas features e/ou muitos data points — esse processo tende a demorar muito.
	Você pode diminuí-lo um pouco ao informar um número maior de steps, ou informar uma proporção de features para remover. Por exemplo: ao invés de remover uma feature de cada vez — step=1 — , você pode remover 5% das features — step=0.05.
- `Sklearn SelectFromModel:` O [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) é uma outra função do sklearn que funciona da seguinte forma: a partir de um modelo (fittado ou não), o SFM irá remover todas as features que não passem do threshold que você informa em seus argumentos. Essa função soou familiar? De fato, o funcionamento do SelectFromModel é bem parecido com o RFE, contudo, o SFM é menos robusto, já que ele baseia sua seleção apenas no threshold informado, enquanto o RFE recursivamente remove as features através de iterações.

---

Antes mesmo de entrarmos na abordagem dos tipos de modelos, como de Regressão, Classificação e Clusterização, quero deixar uma abordagem que me fez enxergar o que de fato é Machine Learning, ou aprendizado de máquina.

Aprendizado de máquina é a ciência (e a arte) da programação de computadores de modo que eles possam aprender com os dados. Aqui está uma definição mais generalizada:

> NOTE
> [Aprendizado de máquina é o] campo de estudo que possibilita aos computadores a habilidade de aprender sem explicitamente programá-los.
> -- Arthur Samuel, 1959

Agora um definição mais orientada à minha formação (engenharia)

> NOTE
> Alega-se que um programa de computador aprende pela experiência E em relação a algum tipo de tarefa T e alguma medida de desempenho P se o seu desempenho em T, conforme medido por P, melhora com a experiência E.
> -- Tom Mitchell, 1997

Espero que essas definições possam te auxiliar você no entendimento sobre o que é machine learning, Diante disso vamos falar de modelos?

### Modelos de Regressão

Antes mesmo de falar dos tipos de modelos de regressão, é importante relembrar que esses modelos, são do tipo de aprendizado supervisionado.
- `Aprendizado supervisionado:` o conjunto de treinamento que você fornece ao algoritmo inclui as soluções desejadas, chamadas de feature e target.

**Regressão**

A regressão linear é um método estatístico utilizado para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes.

Regressão linear simples, também chamada de mínimos quadrados ordinário (OLS), tenta minimizar a soma dos erros quadráticos.

`Algoritmo de Treinamento da Regressão Linear`
O calculo da Regressão Linear para encontrar um ponto que ainda não consta na base de dados pode ser definida 
pela expressão abaixo:

$y = \beta_0 + \beta_1.X + \epsilon$

$y$ $\Rightarrow$ é a variável dependente, ou seja, o valor previsto.

$Beta_0$ $\Rightarrow$ é o coeficiente que intercepta ou que corta o eixo y.
 
$Beta_1$ $\Rightarrow$ é o coeficiente que define a inclinação da reta.

$X$ $\Rightarrow$ é a variável independente, ou seja, a variável preditora.

Para representar a relação entre uma variável dependente ($y$) e uma variável independente ($x$), usamos o modelo
que determina uma linha reta com inclinação $Beta_1$ e intercepto $Beta_0$, com a variável aleatória (erro) $\epsilon$, 
considerada normalmente distribuída com $E(\epsilon) = 0$.

Para simplificar, vamos assumir a premissa de que o valor médio da variável $\epsilon$ para um dado valor de $x$ é $0$.

`CALCULO Beta1`
O calcula de inclinação é feito pela expressão

$\beta_1 =  \frac {n \Sigma xiyi - \Sigma xi \Sigma yi} {n \Sigma xi^2 - (\Sigma xi)^2}$

* $x$ $\Rightarrow$ posição no eixo $X$ do plano cartesiano.
* $y$ $\Rightarrow$ posição no eixo $Y$ do plano cartesiano.
* $i$ $\Rightarrow$ se refere ao i'ésimo valor de $X$ e $Y$.
* n $\Rightarrow$ número de pares ordenados utilizado na base.
* $\Sigma \Rightarrow$ letra grega que incida somatório.

`CALCULO Beta0`
O calcula do intercepto é feito pela expressão

$\beta_0 = \frac {\Sigma yi - \beta_1 \Sigma xi} {n}$

* $x$ $\Rightarrow$ posição no eixo $X$ do plano cartesiano.
* $y$ $\Rightarrow$ posição no eixo $Y$ do plano cartesiano.
* $i$ $\Rightarrow$ se refere ao i'ésimo valor de $X$ e $Y$.
* n $\Rightarrow$ número de pares ordenados utilizado na base.
* $\Sigma \Rightarrow$ letra grega que incida somatório.

Exemplo Manual de Regressão Linear

Vamos usar um conjunto de dados fictício com 5 pontos. A variável independente \( X \) e a variável dependente \( Y \) são:

\[
\begin{array}{|c|c|}
\hline
X & Y \\
\hline
1 & 2 \\
2 & 3 \\
3 & 5 \\
4 & 4 \\
5 & 6 \\
\hline
\end{array}
\]


1. Calcular as Médias de \( X \) e \( Y \):

\[
\bar{X} = \frac{1 + 2 + 3 + 4 + 5}{5} = 3
\]

\[
\bar{Y} = \frac{2 + 3 + 5 + 4 + 6}{5} = 4
\]

2. Calcular os Coeficientes da Regressão Linear ( $\beta_0$ e $\beta_1$ ):

$\beta_1$ (slope):

\[
\beta_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}
\]

Primeiro, vamos calcular as somas:

\[
\sum (X_i - \bar{X})(Y_i - \bar{Y}) = (1-3)(2-4) + (2-3)(3-4) + (3-3)(5-4) + (4-3)(4-4) + (5-3)(6-4)
\]

\[
= (-2)(-2) + (-1)(-1) + (0)(1) + (1)(0) + (2)(2) = 4 + 1 + 0 + 0 + 4 = 9
\]

\[
\sum (X_i - \bar{X})^2 = (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2
\]

\[
= (-2)^2 + (-1)^2 + (0)^2 + (1)^2 + (2)^2 = 4 + 1 + 0 + 1 + 4 = 10
\]

Agora podemos calcular $\beta_1$:

\[
\beta_1 = \frac{9}{10} = 0.9
\]

$\beta_0$ (intercept):

\[
\beta_0 = \bar{Y} - \beta_1 \bar{X} = 4 - 0.9 \times 3 = 4 - 2.7 = 1.3
\]

3. Equação da Regressão Linear:

\[
\hat{Y} = 1.3 + 0.9X
\]

4. Previsão dos Valores de \( Y \):

Agora podemos usar a equação da regressão para prever os valores de \( Y \) para cada valor de \( X \) no nosso conjunto de dados:

| $X$ | $Y$ observado | $\hat{Y}$ |
|-------|------------------|-------------|
| 1     | 2                | $1.3 + 0.9 \times 1 = 2.2$ |
| 2     | 3                | $1.3 + 0.9 \times 2 = 3.1$ |
| 3     | 5                | $1.3 + 0.9 \times 3 = 4.0$ |
| 4     | 4                | $1.3 + 0.9 \times 4 = 4.9$ |
| 5     | 6                | $1.3 + 0.9 \times 5 = 5.8$ |

Conclusão:

A equação da regressão linear para os dados fornecidos é:

\[
\hat{Y} = 1.3 + 0.9X
\]


`Métodos de avaliação do modelo`

Avaliar um modelo de regressão linear é fundamental para entender sua eficácia e fazer ajustes conforme necessário. Abaixo estão as principais formas de avaliar um modelo de regressão linear utilizando o algoritmo dos mínimos quadrados ordinários (OLS):

- `R² (Coeficiente de Determinação): `O R² mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Ele varia de 0 a 1, onde 1 indica que o modelo explica toda a variância dos dados
	\[
	R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
	\]
	- `O que mede:` Proporção da variância na variável dependente explicada pelas variáveis independentes.
	- `Quando usar:` Para avaliar a capacidade explicativa do modelo. Útil para comparar diferentes modelos com as mesmas variáveis dependentes.
	- `Limitações:` Pode ser enganoso em modelos com muitas variáveis independentes.
	- `Para calcular o coeficiente de determinação 𝑅² manualmente, siga estas etapas básicas:`
		1. Calcular a média dos valores observados ( 𝑦̄ ):
		𝑦̄ = 1 𝑛 ∑ 𝑖=1 𝑛 𝑦𝑖
		Onde 𝑦𝑖 são os valores observados e 𝑛 é o número de observações.
		2. Calcular a soma total dos quadrados (SST):
		𝑆𝑆𝑇 = ∑ 𝑖=1 𝑛 (𝑦𝑖 − 𝑦̄)²
		3. Calcular a soma dos quadrados dos resíduos (SSE):
		Suponha que você tenha ajustado um modelo e obteve previsões 𝑦̂𝑖 para cada observação 𝑦𝑖.
		𝑆𝑆𝐸 = ∑ 𝑖=1 𝑛 (𝑦𝑖 − 𝑦̂𝑖)²
		4. Calcular o coeficiente de determinação ( 𝑅² ):
		𝑅² = 1 − 𝑆𝑆𝐸 / 𝑆𝑆𝑇
		Aqui está um exemplo de cálculo passo a passo usando dados hipotéticos:
		Suponha que os valores observados 𝑦𝑖 sejam [10, 15, 12, 18, 20] e as previsões do modelo 𝑦̂𝑖 sejam [11, 14, 13, 17, 19].
		Passo 1: Calcular 𝑦̄:
		𝑦̄ = (10 + 15 + 12 + 18 + 20) / 5 = 75 / 5 = 15
		Passo 2: Calcular SST:
		𝑆𝑆𝑇 = (10 − 15)² + (15 − 15)² + (12 − 15)² + (18 − 15)² + (20 − 15)²
				= 25 + 0 + 9 + 9 + 25 = 68
		Passo 3: Calcular SSE:
		𝑆𝑆𝐸 = (10 − 11)² + (15 − 14)² + (12 − 13)² + (18 − 17)² + (20 − 19)²
				= 1 + 1 + 1 + 1 + 1 = 5
		Passo 4: Calcular 𝑅²:
		𝑅² = 1 − 𝑆𝑆𝐸 / 𝑆𝑆𝑇 = 1 − 5 / 68 = 0.9265

- `Erro Médio Absoluto (MAE):`MAE é a média das diferenças absolutas entre os valores previstos e os valores observados. Fornece uma ideia de quão grandes são os erros em média.
	\[
	MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
	\]
	- `O que mede:` Média das diferenças absolutas entre os valores previstos e os valores observados.
	- `Quando usar:` Quando se deseja uma medida simples e intuitiva do erro médio.
	- `Limitações:` Não diferencia entre erros positivos e negativos.
	- `Cálculo manual do Erro Médio Absoluto (MAE):`
	**Valores Observados**: [10, 15, 12, 18, 20]
	**Previsões do Modelo**: [11, 14, 13, 17, 19]
	`Passos do cálculo:`
	1. Calcular as diferenças absolutas entre os valores observados e as previsões:
   \[
	\begin{align*}
   |10 - 11| & = 1 \\
   |15 - 14| & = 1 \\
   |12 - 13| & = 1 \\
   |18 - 17| & = 1 \\
   |20 - 19| & = 1 \\
   \end{align*}
   \]
	2. Somar as diferenças absolutas:
   \[1 + 1 + 1 + 1 + 1 = 5\]
	3. Dividir a soma pelo número de observações ( n = 5  ):
   \[
	MAE = \frac{5}{5} = 1
	\]
	Resultado: Erro Médio Absoluto (MAE): 1


- `Erro Quadrático Médio (MSE):`MSE é a média dos quadrados das diferenças entre os valores previstos e os valores observados. Penaliza erros maiores mais severamente.
	\[
	MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
	\]
	- `O que mede:` Média dos quadrados das diferenças entre os valores previstos e os valores observados.
	- `Quando usar:` Quando se deseja penalizar mais severamente grandes erros.
	- `Limitações:` Sensível a outliers.
	- `Cálculo manual do Erro Quadrático Médio (MSE):`
	   Valores Observados: [10, 15, 12, 18, 20]
	   Previsões do Modelo: [11, 14, 13, 17, 19]
	   `Passos do cálculo:`
		1. Calcular as diferenças quadráticas entre os valores observados e as previsões:
		\[
		\begin{align*}
		(10 - 11)^2 & = 1 \\
		(15 - 14)^2 & = 1 \\
		(12 - 13)^2 & = 1 \\
		(18 - 17)^2 & = 1 \\
		(20 - 19)^2 & = 1 \\
		\end{align*}
		\]

		2. Somar as diferenças quadráticas:
		\[
		1 + 1 + 1 + 1 + 1 = 5
		\]

		3. Dividir a soma pelo número de observações ( n = 5 ):
		\[
		MSE = \frac{5}{5} = 1
		\]

		Resultado:
		Erro Quadrático Médio (MSE): 1

- `Raiz do Erro Quadrático Médio (RMSE):`RMSE é a raiz quadrada do MSE. É na mesma unidade da variável dependente, facilitando a interpretação.
	\[
	RMSE = \sqrt{MSE}
	\]
	- `O que mede:` Raiz quadrada do MSE, mantendo a unidade da variável dependente.
	- `Quando usar:` Quando se deseja interpretar o erro na mesma unidade da variável dependente.
	- `Limitações:` Sensível a outliers.
	- `Cálculo manual da Raiz do Erro Quadrático Médio (RMSE):`
	   Valores Observados: [10, 15, 12, 18, 20]
	   Previsões do Modelo: [11, 14, 13, 17, 19]
	   `Passos do cálculo:`
		1. Calcular as diferenças quadráticas entre os valores observados e as previsões:
		\[
		\begin{align*}
		(10 - 11)^2 & = 1 \\
		(15 - 14)^2 & = 1 \\
		(12 - 13)^2 & = 1 \\
		(18 - 17)^2 & = 1 \\
		(20 - 19)^2 & = 1 \\
		\end{align*}
		\]

		2. Somar as diferenças quadráticas:
		\[
		1 + 1 + 1 + 1 + 1 = 5
		\]

		3. Dividir a soma pelo número de observações ( \( n = 5 \) ) para obter o MSE:
		\[
		MSE = \frac{5}{5} = 1
		\]

		4. Calcular a raiz quadrada do MSE para obter o RMSE:
		\[
		RMSE = \sqrt{1} = 1
		\]

		Resultado
		Raiz do Erro Quadrático Médio (RMSE): 1

- `Erro Percentual Absoluto Médio (MAPE):`MAPE é a média dos erros percentuais absolutos entre os valores previstos e os valores observados. É uma medida relativa, expressa em porcentagem.
\[
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]
	- `O que mede:` Média dos erros percentuais absolutos entre os valores previstos e os valores observados.
	- `Quando usar:` Para medir o erro em termos percentuais.
	- `Limitações:` Pode ser enganoso se houver valores muito próximos de zero.
	- `Cálculo manual do Erro Percentual Absoluto Médio (MAPE):`
	Valores Observados: [10, 15, 12, 18, 20]
	Previsões do Modelo: [11, 14, 13, 17, 19]
	`Passos do cálculo:`
		1. Calcular os erros percentuais absolutos:
		\[
		\begin{align*}
		\left| \frac{10 - 11}{10} \right| \times 100\% & = 10\% \\
		\left| \frac{15 - 14}{15} \right| \times 100\% & = 6.67\% \\
		\left| \frac{12 - 13}{12} \right| \times 100\% & = 8.33\% \\
		\left| \frac{18 - 17}{18} \right| \times 100\% & = 5.56\% \\
		\left| \frac{20 - 19}{20} \right| \times 100\% & = 5\% \\
		\end{align*}
		\]

		2. Somar os erros percentuais absolutos:
		\[
		10\% + 6.67\% + 8.33\% + 5.56\% + 5\% = 35.56\%
		\]

		3. Dividir a soma pelo número de observações ( \( n = 5 \) ):
		\[
		MAPE = \frac{35.56\%}{5} = 7.11\%
		\]

		Resultado
		Erro Percentual Absoluto Médio (MAPE): 7.11%

- `Ajustado R² (Adjusted R²):`O R² ajustado leva em conta o número de variáveis independentes no modelo. É útil para comparar modelos com diferentes números de variáveis.
	\[
	R_{adj}^2 = 1 - \left( \frac{SS_{res} / (n - p - 1)}{SS_{tot} / (n - 1)} \right)
	\]
	- `O que mede:` Similar ao R², mas ajusta pela quantidade de variáveis independentes no modelo.
	- `Quando usar:` Para comparar modelos com diferentes números de variáveis independentes.
	- `Limitações:` Pode não penalizar suficientemente a complexidade do modelo em datasets muito grandes.
	- `Cálculo manual do coeficiente de determinação ajustado (Adjusted R²):`
	Valores Observados: [10, 15, 12, 18, 20]
	Previsões do Modelo: [11, 14, 13, 17, 19]
	Número de Observações ( n  ): 5
	Número de Preditores ( k ): 1
	`Passos do cálculo:`
		1. Média dos valores observados ( $\bar{y}$ ): 15
		2. Soma Total dos Quadrados (SST): 68
		3. Soma dos Quadrados dos Resíduos (SSE): 5
		4. Coeficiente de Determinação ( $R^2$ ): 0.9265
		5. Coeficiente de Determinação Ajustado ( $R^2_{ajustado}$ ):
		\[ 
		R^2_{ajustado} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)
		\]
		\[ 
		R^2_{ajustado} = 1 - \left( \frac{(1 - 0.9265)(5 - 1)}{5 - 1 - 1} \right)
		\]
		\[
		R^2_{ajustado} = 1 - \left( \frac{0.0735 \times 4}{3} \right)
		\]
		\[
		R^2_{ajustado} = 1 - \left( \frac{0.294}{3} \right)
		\]
		\[
		R^2_{ajustado} = 1 - 0.098
		\]
		\[
		R^2_{ajustado} = 0.902
		\]

- `Teste F:`O teste F avalia a significância global do modelo, verificando se pelo menos uma variável independente tem um coeficiente diferente de zero.
	\[
	F = \left( \frac{R^2 / p}{(1 - R^2) / (n - p - 1)} \right)
	\]
	- `O que mede:` Significância global do modelo.
	- `Quando usar:` Para verificar se pelo menos uma variável independente tem um efeito significativo na variável dependente.
	- `Limitações:` Não fornece informações sobre quais variáveis específicas são significativas.
	- `Cálculo manual do Teste F usando a fórmula:`
	Dados:
	Coeficiente de Determinação ( $R^2$ ): 0.85
	Número de Preditores ( $p$ ): 2
	Número de Observações ( $n$ ): 10
	`Passos do cálculo:`

		1. Calcular a parte superior da fração ( $\frac{R^2}{p}$ ):
		\[
		\frac{R^2}{p} = \frac{0.85}{2} = 0.425
		\]

		2. Calcular a parte inferior da fração ($\frac{1 - R^2}{n - p - 1}$):
		\[
		\frac{1 - R^2}{n - p - 1} = \frac{1 - 0.85}{10 - 2 - 1} = \frac{0.15}{7} \approx 0.0214
		\]

		3. Calcular o valor do Teste F:
		\[
		F = \frac{0.425}{0.0214} \approx 19.86
		\]

		Resultado
		Valor do Teste F: 19.86

- `Análise dos Resíduos:`
	- `Gráfico de Resíduos vs. Valores Ajustados:` Ajuda a identificar a homocedasticidade e a linearidade.
	- `Histograma dos Resíduos:` Ajuda a verificar a normalidade dos resíduos.
	- `Gráfico QQ:` Avalia a normalidade dos resíduos.


Um detalhe bastante importante para esse algoritmo é que o cientista de dados, esteja muito ciente de que quanto mais ele diminuir a função de custo 

`Premissas do Modelo de Regressão Linear`
Para que a regressão linear produza estimativas válidas e significativas, algumas premissas devem ser atendidas:
- Linearidade: A relação entre a variável dependente e as variáveis independentes é linear.
- Independência: As observações são independentes umas das outras.
- Homoscedasticidade: A variância dos resíduos é constante para todos os níveis das variáveis independentes.
- Normalidade dos Erros: Os resíduos do modelo seguem uma distribuição normal.
- Ausência de Multicolinearidade: As variáveis independentes não são altamente correlacionadas entre si.

`Quando Utilizar e Não Utilizar a Regressão Linear`
*Utilizar Quando:*
- A relação entre as variáveis é aproximadamente linear.
- O objetivo é interpretar a relação entre a variável dependente e as variáveis independentes.
- Os dados atendem às premissas do modelo de regressão linear.

*Não Utilizar Quando:*
- A relação entre as variáveis não é linear.
- Existem outliers significativos que influenciam o modelo.
- Há multicolinearidade entre as variáveis independentes.
- As premissas do modelo de regressão linear não são atendidas.

---

- O erro é a diferença entre o valor verdadeiro com o valor previsto pelo modelo, essa equação é chamada de função de erro (LOSS).
- `OBS:` Nem sempre o modelo de OLS consegue analisar de forma eficiente os dados, uma situação é quando o dado mostra multi-colinearidade, isto é, quando as variáveis de entrada estão correlacionadas entre si e também com a variável de resposta.

2. Regularização L1, L2, Elastic Net
	- jjkjk
3. Árvore de regressão
4. Análise de resíduos
5. Modelos lineares generalizados (GLM)

**Classificação**

**Agrupamento**


- Premissas de cada modelo;
- Quando utilizar e não utilizar;
- Metodos de regularização;
- Algoritmo de treinamento;
- Metodos de avaliação do modelo;
- O nome do modelo já o final do metodo do modelo;

---

Olhar as cartas dos fundos, e tentar extrair o máximo de gráficos,