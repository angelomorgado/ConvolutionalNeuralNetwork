# Treinar redes convolucionais - Ângelo Morgado
Resumos baseados [neste artigo](https://victorzhou.com/blog/intro-to-cnns-part-2/)

## Introdução

- Neste resumo iremos ver como **treinar uma CNN**, incluindo derivar gradientes, implementar backpropagation de raíz (usando apenas o numpy), e, por fim, construir uma **training pipeline** completa.
- **Requisitos**: Conhecimento básico de CNN e Cálculo de múltiplas variáveis

## 1 - Contexto

- Usaremos o mesmo exemplo dos resumos de CNN. Iremos usar o como problema.
- A CNN feita nos resumos anteriores era bastante simples apenas consistindo na *Conv layer*, *Max Pooling layer* e da *Softmax layer*, demonstrado pelo diagrama a seguir:

![diagrama](https://victorzhou.com/media/cnn-post/cnn-dims-3.svg)

---

## 2 - Overview do treino

- Treinar uma rede neuronal normalmente consiste em duas fases:
  - A fase **forward**, onde o input é passado pelas camadas da rede para gerar o output.
  - A fase **backward**, onde os gradientes são *backpropagated* (*backprop*) e os pesos e biases são atualizadas.

- Seguiremos este mesmo padrão para treinar a CNN. Existem duas ideias específicas de implementação que seguiremos:
  - Durante a fase **forward** cada camada irá guardar em **cache** qualquer dado (como *inputs*, *valores intermédios*, etc.)
  - Durante a fase **backward**, cada camada irá **receber um gradiente** e também **retornar um gradiente**. Irá receber o gradiente da perda com respeito aos outputs ( *&part;L / &part;out* )e returnar o gradiente de perda com respeito aos seus inputs ( *&part;L / &part;in* ), isto porque estamos a andar de frente para trás, do output para o input.

- Estas duas ideias irão ajudar a nossa implementação de treino limpa e organizada. Olhando para código treinar uma CNN deverá parecer-se mais com [isto](https://replit.com/@Morgado/Convolution-network#trainingExample.py).
- Com o código assim organizado é mais fácil adicionar mais camadas caso seja preciso.

---

## 3 - Backprop: Softmax

- Começaremos do fim e iremos até ao início, visto que é assim que a backprop funciona.
- *nota: lembrar que a cross-entropy loss é o -ln(pc), onde pc é a probabilidade estimada pelo softmax da classe escolhida.*
- A primeira coisa a calcular é o input da camada anterior à da Softmax, ( *&part;L / &part;out<sub>s</sub>* ), onde *out<sub>s</sub>* é o output é o output da camada Softmax: um vetor de 10 probabilidades. Isto é bastante fácil, já que apenas o *pi* aparece na equação da perda:
  - ## *$\frac{\partial L}{\partial out_{s}}$ =  { 0 if i $\neq$ c } || { -$\frac{1}{p_{i}}$ if i = c }*
- Isso é o nosso gradiente inicial referenciado no código:
```
#Calculate initial gradient
gradient = np.zeros(10)
gradient[label] = -1 / out[label] 
```
- Com isto estamos quase prontos para implementar a backward - apenas precisamos de primeiro fazer a forward usando *cache* como mencionado acima, esse código pode ser visto [aqui](https://replit.com/@Morgado/Convolution-network#softmax.py).

- Como podemos reparar no código, **guardamos em cache** três coisas que serão úteis para implementar a fase backward:
  - A **forma do input** (Input shape) antes de lhe aplicarmos o *.flatten()*
  - O **input** depois de lhe aplicarmos o *.flatten()*
  - Os **totais**, que são os valores passados para a função de ativação softmax

- Agora poderemos começar a derivar os gradientes para a fase backprop. Já derivámos a fase de Softmax *$\frac{\partial L}{\partial out_{s}}$*
- Um facto que poderemos utilizar sobre essa fórmula é que só não é zero para *c*, a classe correta. Isso significa que poderemos ignorar tudo menos *out<sub>s</sub>(c)*, relebrando que a função out é a função softmax!
- Primeiro, iremos calcular o gradiente de *out<sub>s</sub>(c)* com respeito aos totais (Os valores passados como argumento da função de ativação Softmax). Seja *t<sub>i</sub>* o valor do total para a classe *i*. Então, poderemos escrever *out<sub>s</sub>(c)* como:
    - ## *out<sub>s</sub>(c) = $\frac{e^{t_{c}}}{\sum_{i}^{} e^{t_{i}}} = \frac{e^{t_{c}}}{S}$*
    - *onde S = $\sum_{i}^{} e^{t_{i}}$, ou seja, a soma de todos os totais. Resumindo, é o total da classe correta sobre a soma de todos os totais.*
- Agora, consideremos uma classe *k* tal que *k $\neq$ c*. Poderemos reescrever *out<sub>s</sub>(c)* como:
  - ## *out<sub>s</sub>(c) = $e^{t_{c}}S^{-1}$*
- E usamos novamente a regra da cadeia para derivar:
  - ## *$\frac{\partial out_{s}(c)}{\partial t_{k}}$ =  $\frac{\partial out_{s}(c)}{\partial S}$ * ( $\frac{\partial S}{\partial t_{k}}$ ) = $\frac{-e^{t_{c}}e^{tk}}{S^{2}}$*
- Relembrando que esta fórmula foi feita assumindo que *k $\neq$ c*, ou seja, fizemos a derivação para uma classe que não é correta, agora façamos a derivação para a classe c, desta vez usando a regra do quociente:
  - ## *$\frac{\partial out_{s}(c)}{\partial t_{c}}$ = $\frac{e^{t_{c}}(S - e^{t_{c}})}{S^{2}}$*
- Tendo agora as duas fórmulas necessárias para a implementação do backprop para o softmax implementemos isso, disponível neste código [aqui](https://replit.com/@Morgado/Convolution-network#softmax.py)
- No código temos que procurar pela classe correta, para fazer isso, iremos iterar sobre todas as classes, e aquela em que o gradiente não for 0 é a correta.
- Após saber isso calculamos o gradiente (dOut_dT) usando as formulas derivadas acima.
- Porém não para aqui, nós queremos os gradientes da perda dos pesos e das biases e do input:
  - Usaremos o gradiente dos pesos, *$\frac{\partial L}{\partial w}$*, para atualizar os pesos da camada;
  - Usaremos o gradiente das biases, *$\frac{\partial L}{\partial b}$*, para atualizar as biases da camada;
  - Usaremos o gradiente do input, *$\frac{\partial L}{\partial input}$*, usando o método *backprop()* para que a próxima camada o possa usar (por próxima refiro-me a anterior), este é o tal gradiente de retorno que vai ser usado na camada anterior.
- Para calcular esses 3 gradientes de loss, teremos que derivar mais 3 resultados: os gradientes dos totais com pesos, biases e input. A equação que nos irá ajudar será esta:
  - ## *t = w * input + b*
  - Não foi mencionado antes, mas esta fórmula representa a fórmula da reta e não é ao acaso, é a fórmula que já estamos habituados a usar desde redes neuronais.
- Estas fórmulas já foram previamente calculadas no resumo das redes neuronais:
  - *$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial out} * \frac{\partial out}{\partial t} * \frac{\partial t}{\partial w}$*
  - 
  - *$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial out} * \frac{\partial out}{\partial t} * \frac{\partial t}{\partial b}$*
  - 
  - *$\frac{\partial L}{\partial input} = \frac{\partial L}{\partial out} * \frac{\partial out}{\partial t} * \frac{\partial t}{\partial input}$*

- Colocando isto em código não é tão simples como aqui porém é possivel e pode ser visualizado [aqui](https://replit.com/@Morgado/Convolution-network#softmax.py), *nota: O código acrescentado terá o título "Weights, biases and input gradient"*
- Explicando o código, primeiro pré-calculamos o **dL_dT** já que o vamos utilizar várias vezes, após isso calculamos cada um dos gradientes:
  - **dL_dW**: Precisamos de arrays 2d para fazer multiplicação de matrizes (@) portanto usamos o método *np.newaxis* para criar uma nova dimensão. Assim, a multiplicação vai acabar por ficar com a shape (inputLen, nodes) que é o mesmo que self.weights.
  - **dL_dB**: Este é fácil já que *dT_dB* é 1.
  - **dL_dInputs**: Multiplicamos as matrizes com dimensões (inputLen, nodes) e (nodes,1) para ter um resultado com comprimento inputLen.
- Com todos os gradientes computados, tudo o que resta é treinar a camada Softmax. Iremos atualizar os pesos e bias usando a **descida de gradiente estocástica**, tal e qual nos resumos de redes neuronais e depois retornar **dL_dInputs**.
- Pode ser visualizado [aqui](https://replit.com/@Morgado/Convolution-network#softmax.py), onde diz *"Update weights/biases"*
- É importante reparar que acrescentamos a variável learnRate que nos indica a que velocidade iremos treinar os nossos pesos. E retornamos o input com o shape original porque estava flat
- Para testar iremos criar um método **train()** no código presente [aqui](https://replit.com/@Morgado/Convolution-network#cnn.py)

---

## 4 - Backprop: Max Pooling

- A camada Max Pooling não pode ser treinada porque não tem nenhum peso na verdade, mas teremos ainda que implementar o método **backprop()** pois teremos que calcular os gradientes para mandar para a *conv layer*. Comecemos pelo mesmo lugar da última camada, **guardar os valores do forward() em cache**. Tudo o que temos de dar cache é o input (porque não há pesos).
- O código de acrescentar no cache pode ser encontrado [aqui](https://replit.com/@Morgado/Convolution-network#maxpool.py) no método forward().
- Durante o forward, a camada Max Pooling pega no volume do input e diminui o seu comprimento e largura para metade ao pegar o valor máximo de blocos 2x2. 
- O backprop faz o oposto: **dobra o comprimento e a largura** do loss de gradiente ao designar cada valor gradiente para **onde o valor máximo estava originalmente** no bloco 2x2 correspondente.
- Aqui está um exemplo. Considerando esta fase forward:

![forward](https://victorzhou.com/media/cnn-post/maxpool-forward.svg)

A fase backprop da mesma camada iria ficar parecida a isto:

![backprop](https://victorzhou.com/media/cnn-post/maxpool-backprop.svg)

- Cada valor de gradiente irá ser designado para onde o valor máximo estava originalmente, e todos os outros valores serão assinalados com zero.

- A razão dos outros valores serem zero é bastante simples, esses valores não têm qualquer efeito já que **apenas o valor máximo** é tomado em conta.
- É possivel implementar isto de uma forma bastante fácil usando o método **iterateRegions()**.
- O código de backprop na maxpool pode ser encontrado [aqui](https://replit.com/@Morgado/Convolution-network#maxpool.py), no método backprop().
- Para cada pixel em cada região da imagem 2x2, copiamos o gradiente de **dL_dOut** para **dL_dInput** se esse era o valor máximo durante a fase forward.

---

## 5 - Backprop: Conv

- A fase de backpropagating da Conv layer é o core do treino de uma CNN.
- O caching da fase forward é bastante simples e pode ser visualisado neste código [aqui](https://replit.com/@Morgado/Convolution-network#conv.py) e estará assinalado com {NEW} na função forward
- *Para relembrar sobre a implementação: para simplicidade, **assumimos que o input da nossa conv layer era um array 2d**. Isto apenas funciona aqui porque estamos a utilizar como a primeira camada da nossa rede. Se fossemos construir uma rede maior que precisasse usar uma Conv3x3 muitas vezes, teríamos que utilizar o input como um **array 3d**. Isto porque na primeira camada só temos a imagem inicial, nas camadas à frente já trabalhamos com filtros o que obriga a tomar em conta a terceira dimensão.*
- Aqui o maior interesse será o gradiente de loss para os filtros na conv layer (camada de convolução), visto que nós precisamos disso para atualizar os pesos dos filtros. Já sabemos *à priori* o $\frac{\partial L}{\partial out}$ para a conv layer, então apenas precisamos encontrar o $\frac{\partial out}{\partial filters}$. Para calcular isso, teremos que nos perguntar: **como é que alterar os pesos dos filtros afetaria o output da conv layer?**
- A realidade é que **alterar quaisquer pesos nos filtros irá alterar a imagem de output para esse filtro**, visto que todos os pixeis de output usam todos os pesos dos pixeis durante a convolução. Para tornar isto mais fácil de imaginar, pensaremos num único pixel de output de cada vez: **como é que alterar o filtro alteraria o output de um único pixel?**
- Utilizemos este simples exemplo para ilustrar:

![example](https://victorzhou.com/media/cnn-post/conv-gradient-example-1.svg) 

- Aqui temos uma imagem 3x3 convolucionada com um filtro 3x3 cheio de zeros para produzir um output 1x1, o que aconteceria se aumentassemos o valor central do filtro para 1? o output iria ter um valor final de 80:

![result](https://victorzhou.com/media/cnn-post/conv-gradient-example-2.svg)

- De forma similar aumentar qualquer outro dos pesos do filtro por 1 iria aumentar o output pelo valor do pixel de input. Isto sugere que a derivada de um pixel de output com respeito a um peso de um filto específico é apenas o valor do pixel da imagem. Fazer as contas confirma isso.
- Com isto chegamos à fórmula final do gradiente da loss para pesos de filtros específicos
  - ## *$\frac{\partial L}{\partial filter(x,y)} = \sum_{i}\sum_{j}\frac{\partial L}{\partial out(i,j)} * \frac{\partial out(i,j)}{\partial filter(x,y)}$*
  - sendo filter(x, y) o peso na posição (x, y) no filtro e out(i, j) o valor do pixel output da posição (i, j)
- Armados da fórmula podemos agora implementá-la em código presente [aqui](https://replit.com/@Morgado/Convolution-network#conv.py) no método backprop().
- Aplicámos a equação derivada ao iterar sobre todas as regiões da imagem/ filtro e construir incrementalmente os gradientes de loss. Quando tivermos percorrido tudo, teremos que atualizar a variável **self.filters** usando SGD (Descida de gradiente estocática) como antes.
- Como já não há mais camadas à espera do gradiente de loss do input desta camada retornamos *None*.

---

## 6 - Treinar a CNN

- Iremos treinar a CNN por alguns epochs, ver o progresso ao longo do treino, e depois testar num conjunto de teste separado.
- O código de treino pode ser encontrado [aqui](https://replit.com/@Morgado/Convolution-network#conv.py)