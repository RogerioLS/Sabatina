## 📚 Estrutura de Aprendizado em Modelos de Linguagem

Ótimo! Vamos estruturar o aprendizado de forma progressiva, desde os fundamentos matemáticos até a implementação dos algoritmos em Python. O conteúdo está dividido em etapas para facilitar o entendimento:

### 📊 1. Fundamentos Matemáticos
Antes de mergulhar nos modelos de linguagem, precisamos entender os pilares matemáticos:

- ***Álgebra Linear***: Vetores e matrizes são a base para representar dados e operações em modelos de linguagem. Operações como multiplicação de matrizes são usadas em camadas de redes neurais.
- ***Cálculo Diferencial e Integral***: Derivadas e gradientes são cruciais para otimizar funções de perda durante o treinamento de modelos. O gradiente descendente e suas variantes são algoritmos de otimização amplamente utilizados.
- ***Probabilidade e Estatística***: Modelos de linguagem frequentemente lidam com incertezas e distribuições de probabilidade. Conceitos como o Teorema de Bayes são fundamentais para a inferência estatística e a avaliação de modelos.

### Álgebra Linear
- Vetores, matrizes e operações (soma, multiplicação, transposição).

    <details>
    <summary><strong>🔹 Module 00 - Vetores e Matrizes</strong></summary>
    Um vetor é uma sequência ordenada de números. Por exemplo, em 2D, temos um <strong>vetor</strong> de duas coordenadas 
    <img src="https://latex.codecogs.com/svg.latex?[x,y]" />. Vetores podem representar palavras, características de um item, etc.

    <strong>Exemplo de vetor:</strong>
    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\mathbf{v}=[3,5,-2]" /></p>

    Em Python, podemos criar um vetor usando <em>Numpy:</em>
    ```python
        import numpy as np
        # Criando um vetor em 3D
        v = np.array([3, 5, -2])
        print("Vetor:", v)
    ```
    Já por outro lado uma <strong>matriz</strong> é uma coleção de vetores dispostos em um formato de tabela (linha e colunas).
    <strong>Exemplo de uma matriz 2x3 (linhas e 3 colunas):</strong>
    <p align="center">
    <img src="https://latex.codecogs.com/svg.latex?M%20=%20%5Cbegin%7Bbmatrix%7D%0A1%20%26%204%20%5C%5C%0A2%20%26%205%20%5C%5C%0A3%20%26%206%20%5Cend%7Bbmatrix%7D" /></p>
    
    Em Python, podemos criar um matriz usando <em>Numpy:</em>
    ```python
        import numpy as np
        # Criando uma matriz 2x3
        M = np.array([[1, 2, 3], [4, 5, 6]])
        print("Matriz:\n", M)
    ```
    </details>

    <details>
    <summary><strong>🔹 Module 01 - Operações com Vetores e Matrizes</strong></summary>
    <p><strong>Soma de Vetores:</strong><br>
    Podemos somar dois vetores da mesma dimensão, somando seus elementos correspondentes.</p>

    <p>Exemplo:<br>
    Se tivermos <strong>𝑣<sub>1</sub> = <img src="https://latex.codecogs.com/svg.latex?[3,5,-2]" /></strong> e <strong>𝑣<sub>2</sub> = <img src="https://latex.codecogs.com/svg.latex?[1,-3,4]" /></strong>, a soma será:</p>

    <p align="center"><strong>𝑣<sub>1</sub> + 𝑣<sub>2</sub> = <img src="https://latex.codecogs.com/svg.latex?[3+1,5+(-3),-2+4]" /> = <img src="https://latex.codecogs.com/svg.latex?[4,2,2]" /></strong></p>

    Em Python, podemos somar os vetores usando <em>Numpy:</em>
    ```python
        import numpy as np
        v1 = np.array([3, 5, -2])
        v2 = np.array([1, -3, 4])
        # Soma de vetores
        soma = v1 + v2
        print("Soma dos vetores:", soma)
    ```
    <strong>Multiplicação de Vetor por um Escalar:</strong>
    Multiplicar um vetor por um número (escalar) significa multiplicar cada elemento do vetor por esse número.</p>

    
    Multiplicando <strong>𝑣<sub>1</sub> = <img src="https://latex.codecogs.com/svg.latex?[3,5,-2]" /></strong> por 2:</p>

    <p align="center"><strong>2 ⋅ 𝑣<sub>1</sub> = <img src="https://latex.codecogs.com/svg.latex?[2*3,2*5,2*(-2)]" /> = <img src="https://latex.codecogs.com/svg.latex?[6,10,-4]" /></strong></p>

    <p><strong>Código:</strong></p>
    <pre><code class="python">
    # Multiplicando vetor por um escalar
    escalar = 2
    resultado = escalar * v1
    print("Multiplicação por escalar:", resultado)
    </code></pre>
    </details>




- Espaços vetoriais, normas, distância Euclidiana e produtos internos (dot product).
- Decomposição de matrizes (SVD, PCA para redução de dimensionalidade).

### Cálculo Diferencial e Integral
- Derivadas parciais e gradientes (essenciais para o backpropagation).
- Regras da cadeia para derivação de funções compostas.
- Otimização com Gradiente Descendente (GD e variantes: SGD, Adam, etc.).

### Probabilidade e Estatística
- Distribuições de probabilidade (normal, binomial, etc.).
- Teorema de Bayes e inferência estatística.
- Entropia, cross-entropy e Kullback-Leibler divergence (importantes para funções de perda).

### Teoria da Informação
- Codificação de Shannon, entropia e compressão de dados.
- Relação da teoria da informação com o aprendizado de representações.

## 🤖 2. Introdução aos Modelos de Linguagem (Language Models - LM)
Modelos de linguagem atribuem probabilidades a sequências de palavras. O objetivo principal é modelar a distribuição de probabilidade de uma sequência de texto:

P(w₁, w₂, ..., wₙ) = P(w₁) ⋅ P(w₂|w₁) ⋅ P(w₃|w₁, w₂) ⋯ P(wₙ|w₁, ..., wₙ₋₁)

### Modelos N-gram
- Definição e funcionamento (bigram, trigram, etc.).
- Limitações: explosão combinatória e falta de generalização.
- Smoothing (Laplace, Kneser-Ney).

### Bag of Words (BoW) e TF-IDF
- Representação de texto como vetores de frequência.
- Pontos fortes e fracos (perda da ordem das palavras).

## ⚡ 3. Modelos Baseados em Redes Neurais

### Word Embeddings
- **Word2Vec (CBOW e Skip-Gram):** Arquitetura, treinamento e função de perda.
- **GloVe (Global Vectors):** Baseado em matrizes de coocorrência.

### Redes Neurais Recorrentes (RNNs)
- Conceito de estados ocultos e processamento sequencial.
- LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Unit).

## 🌍 4. Modelos Baseados em Atenção e Transformers

### O Problema das RNNs e a Solução da Atenção
- Mecanismo de atenção (Attention Mechanism).
- Multi-Head Attention.

### Transformers
- Arquitetura: Encoder, Decoder e Self-Attention.
- Fórmulas matemáticas para cálculo da atenção.

## 🤯 5. Modelos de Linguagem de Grande Escala (LLMs)
Explorando arquiteturas modernas, como o GPT e o BERT:

### GPT (Generative Pre-trained Transformer)
- Arquitetura baseada em decoder-only.
- Causal Masking para prever a próxima palavra.

### BERT (Bidirectional Encoder Representations from Transformers)
- Encoder-only, aprendizado bidirecional.

## 🔬 6. Treinamento e Avaliação de Modelos
- Funções de perda: Cross-Entropy Loss e variantes.
- Técnicas de regularização: Dropout, weight decay, early stopping.
- Métricas de avaliação: Perplexity, BLEU Score, ROUGE, etc.

## 🚀 7. Aplicações Avançadas e Casos de Uso
- **Chatbots e Assistentes Virtuais.**
- **Summarization, Translation, Question Answering.**

## 📦 8. Implementação Final do Projeto em Python
Criação de um projeto completo com:
- Pré-processamento de dados.
- Construção do modelo do zero.
- Treinamento e validação.
- Deploy simples da API usando FastAPI ou Flask.

## 📌 Revisão e Estudos Avançados

- Revisar fundamentos matemáticos e estatísticos.
- Praticar a implementação dos algoritmos em Python.
- Explorar aplicações práticas e casos de MLOps.
