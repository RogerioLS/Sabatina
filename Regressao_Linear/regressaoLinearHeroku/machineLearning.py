"""importar pacotes necessários"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from PIL import Image
from PIL import Image
import os
import math
from sklearn.linear_model import LinearRegression

@st.cache
def load_image(img):
	im = Image.open(os.path.join(img))
	return im

st.info(f'''EMENTA

Vamos separar em dois nivéis de apreendizado da Regressão Linear.

1º Realizaremos um modelo simples e explicando o passo a passo matématicamente o que acontece com o modelo até ele
encontrar a reta que melhor se encaixa entre os pontos.

* Calculo modelo Regressão Linear populacional.
* Calculo da correlação dos nossos dados.
    
2º Usaremos a lib do [`SCIKIT-LEARN`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) para demonstrar a sua praticidade.  
''')

st.markdown('''
# O que é Regressão Linear?

---

A Regressão Linear é uns dos algoritmos mas visto pelos cientista de dados no começo da carreira, e pra exemplificar de 
uma forma armonioza, vou exemplificar um tipo de problema que pode ser relvido com esse algorita.

Digamos que você queira saber o preço de venda de uma casa sua e acredita que existe um relacionamento entre as 
variáveis que você está considerando (área construída, número de quartos e localização) com esse preço. Seria possível 
fazer uma análise de regressão baseado nas outras casas da cidade, obter os pesos para os parâmetros em um modelo 
Regressão Linear e inferir qual o preço de venda que você deve colocar.

Ou até um mais simples que séria dizer o tamanho da asa de um pinguin dado a sua massa corporal, resumindo o objetivo 
da análise de regressão é explorar o relacionamento existente entre duas ou mais variáveis, visando obter informações 
sobre uma delas a partir dos valores conhecidas das outras.

Vale lembrar de um ponto importante, mas muito desconhecido, é que nos nossos problemas do cotidiano, 
muitas variáveis x e y aparentam estar relacionadas uma com a outra, porém de maneira não determinística.

Uma relação determínistica, por exemplo, é quando queremos saber a distância percorrida por um carro, mantendo 
velocidade constante $v$ ao longo 
de $\Delta t$ segundos. Nesse exemplo, sabemos que a distância percorrida será $\Delta s = v * \Delta t$, 
pois as variáveis estão relacionadas deterministicamente.

Vamos deixar um exemplo para tomar mais claro a explicação.

---''')

with st.spinner('Aguarde o gráfico ser criado...'):
    time.sleep(3)
    np.random.seed(42)
    det_x = np.arange(0, 10, 0.1)
    det_y = 2 * det_x + 3

    # transformando em data frame deterministicos
    series_det = pd.Series(det_y, det_x)
    df_det = pd.DataFrame(series_det, columns=['det_y'])
    df_det.reset_index(drop=False, inplace=True)
    df_det = df_det.rename(columns={'index': 'det_x'})

    feature_name_det = "det_x"
    target_name_det = "det_y"
    data_det, target_det = df_det[[feature_name_det]], df_det[target_name_det]

    # exemplo de plots não determinísticos
    non_det_x = np.arange(0, 10, 0.1)
    non_det_y = 2 * non_det_x + np.random.normal(size=100)

    # transformando em data frame não deterministicos
    series = pd.Series(non_det_y, non_det_x)
    df_non = pd.DataFrame(series, columns=['non_det_y'])
    df_non.reset_index(drop=False, inplace=True)
    df_non = df_non.rename(columns={'index': 'non_det_x'})

    feature_name_non = "non_det_x"
    target_name_non = "non_det_y"
    data_non, target_non = df_non[[feature_name_non]], df_non[target_name_non]

    plt.figure(figsize=(15, 5), constrained_layout=False)

    ax = plt.subplot(1, 2, 1)
    ax = sns.scatterplot(data=df_det, x=feature_name_det, y=target_name_det,
                         color="blue", alpha=0.5)
    ax.set_title("Determinístico")

    ax = plt.subplot(1, 2, 2)
    ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                         color="blue", alpha=0.5)
    ax.set_title("Não Determinístico")

    #plt.show()

    st.pyplot(plt)
st.success('Feito!')

code = """np.random.seed(42)
det_x = np.arange(0, 10, 0.1)
det_y = 2 * det_x + 3

# transformando em data frame deterministicos
series_det = pd.Series(det_y, det_x)
df_det = pd.DataFrame(series_det, columns=['det_y'])
df_det.reset_index(drop=False, inplace=True)
df_det = df_det.rename(columns={'index': 'det_x'})

feature_name_det = "det_x"
target_name_det = "det_y"
data_det, target_det = df_det[[feature_name_det]], df_det[target_name_det]

# exemplo de plots não determinísticos
non_det_x = np.arange(0, 10, 0.1)
non_det_y = 2 * non_det_x + np.random.normal(size=100)

# transformando em data frame não deterministicos
series = pd.Series(non_det_y, non_det_x)
df_non = pd.DataFrame(series, columns=['non_det_y'])
df_non.reset_index(drop=False, inplace=True)
df_non = df_non.rename(columns={'index': 'non_det_x'})

feature_name_non = "non_det_x"
target_name_non = "non_det_y"
data_non, target_non = df_non[[feature_name_non]], df_non[target_name_non]

plt.figure(figsize=(15, 5), constrained_layout=False)

ax = plt.subplot(1, 2, 1)
ax = sns.scatterplot(data=df_det, x=feature_name_det, y=target_name_det,
                     color="blue", alpha=0.5)
ax.set_title("Determinístico")

ax = plt.subplot(1, 2, 2)
ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                     color="blue", alpha=0.5)
ax.set_title("Não Determinístico")

plt.show()"""
dados = st.checkbox('Código da criação dos gráficos')

if dados:
    with st.spinner('Aguarde o código está sendo carregado...'):
        time.sleep(3)
        st.code(code, language='python')
    st.success('Feito!')

st.warning("""Olhando rapidamente você já consegue ver uma diferença importante, que apesar dos dois gráficos estarem 
mostrando pontos que se espalham sobre uma “reta virtual”, um deles não segue um padrão exato, determinístico. 
Parece que há algum tipo de aleatoriedade envolvida.
Ou seja não temos uma relação perfeita entre as variavéis, isso significa que temos um modelo probabilístico, 
que captura a aleatoriedade que é inerente de qualquer processo do mundo real.""")

st.markdown("""# Matemática em ação calculo da Regressão Linear populacional
---
O calculo da Regressão Linear para encontrar um ponto que ainda não consta na base de dados pode ser definida 
pela expressão abaixo:""")

st.latex(r'''
y = \beta_0 + \beta_1.X + \epsilon
''')

st.markdown("""

$y$ $\Rightarrow$ é a variável dependente, ou seja, o valor previsto.

$Beta_0$ $\Rightarrow$ é o coeficiente que intercepta ou que corta o eixo y.
 
$Beta_1$ $\Rightarrow$ é o coeficiente que define a inclinação da reta.

$X$ $\Rightarrow$ é a variável independente, ou seja, a variável preditora.

Para representar a relação entre uma variável dependente ($y$) e uma variável independente ($x$), usamos o modelo
que determina uma linha reta com inclinação $Beta_1$ e intercepto $Beta_0$, com a variável aleatória (erro) $\epsilon$, 
considerada normalmente distribuída com $E(\epsilon) = 0$.

Para simplificar, vamos assumir a premissa de que o valor médio da variável $\epsilon$ para um dado valor de $x$ é $0$.

Outro ponto é que vamos ter que criar novas colunas tais como:

$\Rightarrow$ $\Sigma xiyi$

$\Rightarrow$ $\Sigma xi^2$

$\Rightarrow$ $\Sigma xi^2$

---""")

# criando colunas necessárias para o calculo.
df_non['non_det_xy'] = df_non['non_det_x'] * df_non['non_det_y']
df_non['non_det_x^2'] = df_non['non_det_x'] ** 2
df_non['non_det_y^2'] = df_non['non_det_y'] ** 2

code_2 = '''
# criando colunas necessárias para o calculo.
df_non['non_det_xy'] = df_non['non_det_x'] * df_non['non_det_y']
df_non['non_det_x^2'] = df_non['non_det_x'] ** 2
df_non['non_det_y^2'] = df_non['non_det_y'] ** 2'''

dados_2 = st.checkbox('Código da criação das colunas')

if dados_2:
    with st.spinner('Aguarde o código está sendo carregado...'):
        time.sleep(3)
        st.code(code_2, language='python')
        st.write(df_non)
    st.success('Feito!')

st.markdown('''
##### CALCULO Beta1
O calcula de inclinação é feito pela expressão''')

st.latex(r'''
\beta_1 =  \frac {n \Sigma xiyi - \Sigma xi \Sigma yi} {n \Sigma xi^2 - (\Sigma xi)^2}
''')

st.markdown('''
* $x$ $\Rightarrow$ posição no eixo $X$ do plano cartesiano.
* $y$ $\Rightarrow$ posição no eixo $Y$ do plano cartesiano.
* $i$ $\Rightarrow$ se refere ao i'ésimo valor de $X$ e $Y$.
* n $\Rightarrow$ número de pares ordenados utilizado na base.
* $\Sigma \Rightarrow$ letra grega que incida somatório.
''')

n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])
sigmaXiYi = sum(df_non['non_det_xy'])
sigmaXi2 = sum(df_non['non_det_x^2'])

b1 = ((n * sigmaXiYi) - (sigmaXi * sigmaYi)) / ((n * sigmaXi2) - (sigmaXi ** 2))

code_3 = '''
n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])
sigmaXiYi = sum(df_non['non_det_xy'])
sigmaXi2 = sum(df_non['non_det_x^2']) 
b1 = ((n * sigmaXiYi) - (sigmaXi * sigmaYi)) / ((n * sigmaXi2) - (sigmaXi ** 2))'''

st.code(code_3, language='python')
st.markdown(f'''Resultado:\n
    Número de pares ordenados utilizado na base: {n} 
    Somatorio da coluna X: {sigmaXi}  
    Somatorio da coluna Y: {sigmaYi}
    Somatorio da coluna X e Y: {sigmaXiYi}
    Somatorio da coluna X ao quadrado: {sigmaXi2}
    Calculo b1: {b1}''')

st.markdown('''---''')

st.markdown('''
##### CALCULO Beta0
O calcula do intercepto é feito pela expressão''')

st.latex(r'''
\beta_0 = \frac {\Sigma yi - \beta_1 \Sigma xi} {n} 
''')

st.markdown('''
* $x$ $\Rightarrow$ posição no eixo $X$ do plano cartesiano.
* $y$ $\Rightarrow$ posição no eixo $Y$ do plano cartesiano.
* $i$ $\Rightarrow$ se refere ao i'ésimo valor de $X$ e $Y$.
* n $\Rightarrow$ número de pares ordenados utilizado na base.
* $\Sigma \Rightarrow$ letra grega que incida somatório.
''')

n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])

b0 = (sigmaYi - (b1 * sigmaXi)) / n

code_4 = '''
n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])
b0 = ((sigmaYi) - (b1 * sigmaXi)) / n'''

st.code(code_4, language='python')
st.markdown(f'''Resultado:\n
    Número de pares ordenados utilizado na base: {n} 
    Somatorio da coluna X: {sigmaXi}  
    Somatorio da coluna Y: {sigmaYi}
    Calculo b0: {b0}''')

st.markdown('''---''')

st.markdown('''Agora vamos ver o resultado final da nossa expressão e também iremos realizar o calculo para um dado
que não consta em nossa base.''')

st.markdown(f'''Expressão final ápos a realização dos calculos: \n
    y = {b0} + {b1} * X''')

x = 4.65
y = b0 + (b1 * x)

st.markdown(f'''Dados que não consta em nossa base: \n
    x = 4.65
    y = {b0} + ({b1} * {x})
    y = {y}''')

st.markdown('''Encontrando o ponto no grafico com os valores de X e Y''')

code_5 = '''
point = pd.DataFrame({'x': [4.65], 'y': [9.19]})
plt.figure(constrained_layout=True)
ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                     color="blue", alpha=0.5)

label = "x {0:.2f} e y {1:.2f}"
ax = point.plot(x='x', y='y', ax=ax, marker='o', markersize=4, color="red", label=label.format(x, y))
ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)

eixo_x = x
plt.axvline(x=eixo_x, ymin=0, ymax=0.47, color="black", linestyle="--")

eixo_y = y
plt.axhline(y=eixo_y, xmin=0, xmax=0.465, color="black", linestyle="--")

plt.annotate("Ponto Calculado", (.65, 9.5), fontsize=15)

ax.set_title("Não Determinístico")

plt.show()'''

dados_5 = st.checkbox('Código da criação do gráfico ponto')

if dados_5:
    with st.spinner('Aguarde o código está sendo carregado...'):
        time.sleep(3)
        st.code(code_5, language='python')
    st.success('Feito!')

with st.spinner('Aguarde o gráfico ser criado...'):
    time.sleep(3)
    point = pd.DataFrame({'x': [4.65], 'y': [9.19]})
    plt.figure(constrained_layout=True)
    ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                         color="blue", alpha=0.5)

    label = "x {0:.2f} e y {1:.2f}"
    ax = point.plot(x='x', y='y', ax=ax, marker='o', markersize=4, color="red", label=label.format(x, y))
    ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)

    eixo_x = x
    plt.axvline(x=eixo_x, ymin=0, ymax=0.47, color="black", linestyle="--")

    eixo_y = y
    plt.axhline(y=eixo_y, xmin=0, xmax=0.465, color="black", linestyle="--")
    plt.annotate("Ponto Calculado", (.65, 9.5), fontsize=15)
    ax.set_title("Não Determinístico")
    #plt.show()
    st.pyplot(plt)
st.success('Feito!')

st.markdown('''---''')

st.markdown('''Agora vamos criar uma função que já faz esse calculo e aplicar essa função sobre nossos
dados para encontra a reta de melhor ajuste''')

code_6 = '''
def linear_model(x, beta_um, beta_zero):
    """Linear model funcao y = a * x + b"""
    y = x * beta_um + beta_zero
    return y


reta_ajuste = linear_model(df_non['non_det_x'], b1, b0)

# Plot da reta de melhor ajuste
plt.figure(constrained_layout=True)
ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                     color="blue", alpha=0.5)

label = "Reta de melhor ajuste com b1 {0:.2f} * x + b0 {1:.2f}"
ax.plot(df_non['non_det_x'], reta_ajuste, color='r', linestyle='--', linewidth=2 ,label=label.format(b1, b0))
ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)
ax.set_title("Não Determinístico")
plt.show()'''

dados_6 = st.checkbox('Código da criação do gráfico reta melhor ajuste')

if dados_6:
    with st.spinner('Aguarde o código está sendo carregado...'):
        time.sleep(3)
        st.code(code_6, language='python')
    st.success('Feito!')


def linear_model(x, beta_um, beta_zero):
    """Linear model funcao y = a * x + b"""
    y = x * beta_um + beta_zero
    return y


reta_ajuste = linear_model(df_non['non_det_x'], b1, b0)

with st.spinner('Aguarde o gráfico ser criado...'):
    time.sleep(3)
    plt.figure(constrained_layout=True)
    ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                         color="blue", alpha=0.5)

    label = "Reta de melhor ajuste com b1 {0:.2f} * x + b0 {1:.2f}"
    ax.plot(df_non['non_det_x'], reta_ajuste, color='r', linestyle='--', linewidth=2, label=label.format(b1, b0))
    ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)
    ax.set_title("Não Determinístico")
    #plt.show()
    st.pyplot(plt)
st.success('Feito!')

st.markdown('''---''')

st.markdown('''Acredito que vale a pena fazermos retas que não performam bem no nosso modelo só para fins didaticos, 
pra isso vamos estipular alguns valores para $b1$ e $b0$ ''')

code_7 = '''
b_um = [1.01, 2.01, -1.59]
b_zero = [-2.1, -0.17, 17]

plt.figure(constrained_layout=True)
ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                     color="blue", alpha=0.5)

label = "b1 {0:.2f} * x + b0 {1:.2f}"
for b1, b0 in zip(b_um, b_zero):
    reta_ajuste = linear_model(df_non['non_det_x'], b1, b0)
    ax.plot(df_non['non_det_x'], reta_ajuste,
            label=label.format(b1, b0))

ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)
ax.set_title("Não Determinístico")
plt.show()'''

dados_7 = st.checkbox('Código da criação do gráfico multiplas retas')

if dados_7:
    with st.spinner('Aguarde o código está sendo carregado...'):
        time.sleep(3)
        st.code(code_7, language='python')
    st.success('Feito!')

b_um = [1.01, 2.01, -1.59]
b_zero = [-2.1, -0.17, 17]

with st.spinner('Aguarde o gráfico ser criado...'):
    time.sleep(3)
    plt.figure(constrained_layout=True)
    ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                         color="blue", alpha=0.5)

    label = "b1 {0:.2f} * x + b0 {1:.2f}"
    for b1, b0 in zip(b_um, b_zero):
        reta_ajuste = linear_model(df_non['non_det_x'], b1, b0)
        ax.plot(df_non['non_det_x'], reta_ajuste,
                label=label.format(b1, b0))

    ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)
    ax.set_title("Não Determinístico")
    #plt.show()
    st.pyplot(plt)
st.success('Feito!')

st.markdown('''---''')

st.markdown('''
# Calculo da correlação dos nossos dados

---

#### O que é correlação?

A função do coeficiente de correlação é determinar qual é a intensidade da relação que existe entre conjuntos de dados 
ou informações conhecidas. O valor do coeficiente de correlação pode variar entre -1 e 1 e o resultado obtido define se 
a correlação é negativa ou positiva''')

#image = Image.open('correlacao_p')
#st.image(image, caption='Tabela de correlação')
st.image(load_image('correlacao_p.png'),  caption='Tabela de correlação')

st.markdown('''
#### Expressão da correlação.

O calculo da expressão de correlação para encontrar se o nosso conjunto de dados tem uma forte ou baixa correlação
 pode ser definida pela expressão abaixo:''')

st.latex(r'''r(x,y) = \frac {Sxy} {\sqrt Sxx * Syy}''')

st.markdown('''
Sxx é o:
* Desvio-padrão de x:''')

st.latex(r'''Sxx = \frac {\Sigma xi^2 - (\Sigma xi)^2} {n}''')

st.markdown('''
Syy é o:
* Desvio-padrão de y:''')

st.latex(r'''Sxx = \frac {\Sigma yi^2 - (\Sigma yi)^2} {n}''')

st.markdown('''
Sxy é a:
* covariância de x, y:''')

st.latex(r'''Sxy = \frac {\Sigma xiyi - (\Sigma xi * \Sigma yi)} {n}''')

st.markdown('''
* $x$ $\Rightarrow$ posição no eixo $X$ do plano cartesiano.
* $y$ $\Rightarrow$ posição no eixo $Y$ do plano cartesiano.
* $i$ $\Rightarrow$ se refere ao i'ésimo valor de $X$ e $Y$.
* n $\Rightarrow$ número de pares ordenados utilizado na base.
* $\Sigma \Rightarrow$ letra grega que significa somatório.
''')

st.markdown('''---''')

n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])
sigmaXiYi = sum(df_non['non_det_xy'])
sigmaXi2 = sum(df_non['non_det_x^2'])
sigmaYi2 = sum(df_non['non_det_y^2'])

code_8 = '''
n = 100
sigmaXi = sum(df_non['non_det_x'])
sigmaYi = sum(df_non['non_det_y'])
sigmaXiYi = sum(df_non['non_det_xy'])
sigmaXi2 = sum(df_non['non_det_x^2'])
sigmaYi2 = sum(df_non['non_det_y^2'])'''

st.code(code_8, language='python')

st.markdown(f'''Resultado:\n
    Número de pares ordenados utilizado na base: {n}
    Somatorio da coluna X: {sigmaXi}
    Somatorio da coluna Y: {sigmaYi}
    Somatorio da coluna X e Y: {sigmaXiYi}
    Somatorio da coluna Y ao quadrado: {sigmaYi2}
    Somatorio da coluna X ao quadrado: {sigmaXi2}''')

st.markdown('''---''')

sxx = sigmaXi2 - (sigmaXi ** 2) / 100
code_9 = '''
sxx = sigmaXi2 - (sigmaXi ** 2) / 100
'''
st.code(code_9, language='python')
st.markdown(f'''Resultado:\n
    sxx = {sxx}''')

st.markdown('''---''')

syy = sigmaYi2 - (sigmaYi ** 2) / 100
code_10 = '''
syy = sigmaYi2 - (sigmaYi ** 2) / 100
'''
st.code(code_10, language='python')
st.markdown(f'''Resultado:\n
    syy = {syy}''')

st.markdown('''---''')

sxy = sigmaXiYi - (sigmaXi * sigmaYi) / 100
code_11 = '''
sxy = sigmaXiYi - (sigmaXi * sigmaYi) / 100
'''
st.code(code_11, language='python')
st.markdown(f'''Resultado:\n
    sxy = {sxy}''')

st.markdown('''---''')

r = sxy / math.sqrt(sxx * syy)
code_12 = '''
r = sxy / math.sqrt(sxx * syy)
'''
st.code(code_12, language='python')
st.markdown(f'''Resultado:\n
    Obtemos uma relação forte positiva de {np.around(r*100,2)}%, ou seja as variavéis X e Y se conversão''')

st.markdown('''---''')

st.markdown('''
# Utilizando a lib do sklearn''')

code_13 = '''
lm_model = LinearRegression()
lm_model.fit(non_det_x.reshape(-1, 1), non_det_y)

b1 = lm_model.coef_
b0 = lm_model.intercept_

# Plot da reta de melhor ajuste
plt.figure(constrained_layout=True)
ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                     color="blue", alpha=0.5)
plt.plot(non_det_x, ((non_det_x * b1) + b0),  color='r', linestyle='--', linewidth=2)
ax.set_title("Não Determinístico")
plt.show()
'''

st.code(code_13, language='python')

lm_model = LinearRegression()
lm_model.fit(non_det_x.reshape(-1, 1), non_det_y)

b1 = lm_model.coef_
b0 = lm_model.intercept_

with st.spinner('Aguarde o gráfico ser criado...'):
    time.sleep(3)
    plt.figure(constrained_layout=True)
    ax = sns.scatterplot(data=df_non, x=feature_name_non, y=target_name_non,
                         color="blue", alpha=0.5)
    plt.plot(non_det_x, ((non_det_x * b1) + b0), color='r', linestyle='--', linewidth=2)
    ax.set_title("Não Determinístico")
    #plt.show()
    st.pyplot(plt)
st.success('Feito!')

r_squared = lm_model.score(non_det_x.reshape(-1, 1), non_det_y)

code_14 = '''
# imprimindo o coeficiente de determinação
r_squared = lm_model.score(non_det_x.reshape(-1, 1), non_det_y)'''

st.code(code_14, language='python')

st.markdown(f'''Resultado:\n
    Coeficiente de Determinação: {np.around(r_squared*100,2)}%''')
