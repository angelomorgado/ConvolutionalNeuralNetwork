# Redes convolucionais - Ângelo Morgado
Resumos baseados [neste artigo](https://victorzhou.com/blog/intro-to-cnns-part-1/)

## 1 - Introdução

  - As CNN têm sido alvo de muita atenção recentemente por causa da forma como têm revolucionado a área de visão computacional (Computer vision), e este documento irá explicar o que são, como funcionam e construir uma em código em Python usando apenas a biblioteca NumPy.

  - Um uso clássico das CNN's é a de classificação de imagens, e.g, olhar para a imagem de um animal e decidir se é um cão ou um gato. 
  - Existem várias razões para se usar CNN em vez de redes neuronais na classificação de imagens, tais razões são:
    - ### Razão 1: Imagens são grandes
      - As imagens usadas para problemas de visão computacional são geralmente 224x244 ou até maiores. Contruir uma rede neuronal para processar imagens do tamanho 224x224 a cores, ou seja, que tenham três canais RGB, teria que ter 224 * 224 * 3 = 150,528 recursos de entrada (input features). Uma camada escondida numa rede destas poderá ter 1024 nodos, então ter-se-ia de treinar 150,528 * 1024 = 150+ milhões de pesos apenas para a primeira camada. A rede seria enorme e quase impossível de treinar.
      - Fora que, não precisamos de todos os pesos. A coisa boa acerca das imagens é que os pixeis são mais úteis no contexto dos seus vizinhos. Seria um desperdício de recursos ter que olhar para todos os pixeis de uma imagem quando só importam alguns.
    - ### Razão 2: Posições podem variar
      - Se treinamos uma rede para detetar cães, seria expectável que ela detetasse um cão **independentemente** da sua posição na imagem. Numa rede neuronal se um cão aparecesse noutro lugar que não aquele em que a rede foi treinada, os pixeis errados iriam se ativar e, consequentemente, a rede neuronal iria se comportar de uma forma completamente diferente do que seria esperado.

## 2 - Dataset (Conjunto de dados)
  - Neste resumo usar-se-á o "Hello World!" de visão computacional, o problema de classificação de escrita digital MNIST. É bastante simples, dada uma imagem, classificá-la quanto um dígito.

![MNIST](https://victorzhou.com/static/16ddab2ee3bcd9e22d96f267e473a2f4/e4151/mnist-examples.webp)
 
  - Cada imagem no dataset do MNIST é 28x28 e contém um dígito centrado em tons de cinza, para simplificar a rede.
  - Uma rede neuronal normal seria capaz de resolver bem este trabalho devido às características das imagens, porém noutros problemas de classificação de imagens, as imagens não serão tão fáceis assim, inutilizando as redes neuronais normais.

## 3 - Convoluções

  - As redes neuronais convolucionais são apenas redes neuronais que usam **camadas convolucionais** a.k.a. *Conv layers*, que são baseadas na operação matemática de [convolução](https://en.wikipedia.org/wiki/Convolution).
  - *Conv layers* consistem num conjunto de **filtros**, que podem ser considerados apenas como matrizes bidimensionais que contêm números, tal e qual esta:

![Conv layer](https://victorzhou.com/687f9b9f24fc4f9a50b30e8a1f4d3686/vertical-sobel.svg)

  - Podemos usar uma imagem de input e um filtro para criar uma imagem de output ao *embrulhar* (**convolving**) o filtro com a imagem de input, isto é um processo que consiste em:
    -  1.Colocar o filtro em cima da imagem numa localização
    -  2.Fazer **multiplicação por elementos** (element-wise multiplication) entre os valores do filtro e os seus valores correspondentes da imagem
    -  3.Somar todos os produtos por elementos. A soma é o valor de output para o pixel destino da imagem de output
    -  4.Repetir para todos os pixeis
- Para entender melhor as CNN basta ver [este video](https://www.youtube.com/watch?v=zfiSAzpy9NM), que explica bastante bem.
 
Para ajudar a entender vejamos um exemplo de como funciona a convolução:
  - Consideremos esta imagem 4x4 em tons de cinza e um filtro 3x3:

![exemplo](https://victorzhou.com/media/cnn-post/convolve-example-1.svg)

  - Os números na imagem representam as intensidades dos pixeis, onde 0 é preto e 255 é branco. Ao convolacionar-mos a imagem com o filtro iremos gerar uma imagem output 2x2:

![imagem output](https://victorzhou.com/media/cnn-post/example-output.svg)

  - Para começar iremos aplicar o primeiro passo, ou seja, o overlay do filtro no canto esquerdo da imagem:

![overlay](https://victorzhou.com/media/cnn-post/convolve-example-2.svg)

  - Agora, seguindo o passo 2, é necessário fazer a multiplicação por elementos onde está o overlay do filtro, tal e qual a tabela seguinte:

| Valor da imagem (I) | Valor do filtro (F) | Produto (I * F) |
|---|---|---|
| 0 | -1 | 0 |
| 50 | 0 | 0 | 
| 0 | 1 | 0 |
| 0 | -2 | 0 |
| 80 | 0 | 0 |
| 31 | 2 | 62 |
| 33 | -1 | -33 |
| 90 | 0 | 0 |
| 0 | 1 | 0 |

  - A seguir, segundo o passo 3, somar todos os resultados:

*62 + (-33) = 29*

  - Finalmente, colocamos o resultado no pixel de destino da nossa imagem de output. Como o filtro está overlayed no canto superior esquerdo da imagem de output, o nosso pixel destino será também o que está no canto superior esquerdo da imagem de output, tal como mostra a próxima imagem:

![overlayed :)](https://victorzhou.com/media/cnn-post/convolve-output-1.svg)

  - Seguindo o passo 4, faz-se o mesmo para os restantes pixeis até fazer a imagem output por completo

![gif](https://victorzhou.com/69b4c1dd078ee363317bb8fa323eaace/convolve-output.gif)

### 3.1 - Qual a utilidade disto?

  - O que faz convular uma imagem com um filtro? O filtro usado anteriormente é chamado de *Sobel filter vertical*, que aplica o seguinte efeito nas imagens: 

![sobel vertical](https://victorzhou.com/static/44a1ff59f9a2c7f62cf9f56a8398efd0/fa73e/lenna%2Bvertical.webp)

  - O que o filtro de sobel faz é **detetar arestas**. O filtro vertical deteta arestas verticais e, o horizontal, deteta arestas horizontais. Com isto, torna-se fácil intrepertar uma imagem; um pixel com uma intensidade alta na imagem output indica que há uma aresta "forte" por ali na imagem original.
  - Assim, a imagem filtrada torna-se mais útil na classificação do que a imagem "raw". Em geral, **a convolução ajuda a encontrar características específicas localizadas na imagem**, de forma a facilitar a classificação, estas features poderão mais tarde ser usadas na rede neuronal.

### 3.2 - Padding

  - No exemplo anterior foi criada uma imagem 2x2 a partir de uma imagem 4x4, porém, quando queremos que a imagem output seja do mesmo tamanho que a input, existe uma técnica que nos ajuda nisso chamada **padding**. O padding consiste em em criar camadas de 0 à volta da imagem input. No caso anterior em que temos uma imagem de input 4x4 e um filtro 3x3 necessitaremos de 1 pixel de padding, tal como mostrado nesta imagem:

![padding](https://victorzhou.com/media/cnn-post/padding.svg)

  - Este padding que faz com que a imagem output tenha o mesmo tamanho que a input é chamado de **padding 'igual'**, enquanto não usar padding é chamado **padding 'válido'**.

### 3.3 - Conv layers

  - As CNN incluem as Conv layers que usam um conjunto de filtros para tornar imagens input em imagens output. O parâmetro primário de uma Conv layer é o seu **número de filtros**.
  - Para o CNN do MNIST. usar-se-á uma pequena conv layer com 8 filtros como camada inicial na nossa rede. Isto significa que irá tornar uma imagemde input 28x28 numa de output 26x26x8 de **volume** (26 porque estamos a usar padding válido):

![volume](https://victorzhou.com/media/cnn-post/cnn-dims-1.svg)

  - Cada um dos 8 filtros na conv layer produz um output de 26x26, então, empilhados os outputs, ocupam 26x26x8 de volume. Isto tudo acontece apenas com 3 * 3 (filtro) * 8 (Número de filtros) = **apenas 72 pesos**!
  - A escolha dos melhores filtros será trabalho do treino desta rede convolacional.

### 3.4 - Implementar convolução

   - Iremos implementar a parte de feedforward de uma conv layer, que, o que faz é transformar uma imagem input numa output. Neste exemplo usaremos um filtro 3x3 porém é bastante comum usar-se filtros 5x5 ou até mesmo 7x7.
   - O código pode ser encontrado [aqui](https://replit.com/@Morgado/Conv-layer#conv.py)
   - *Nota: Dividir por 9 na inicialização faz com que os valores não sejam tão díspares. Isto é de extrema importância porque valores díspares resultam num pior treinamento.*
   - O método **iterateRegions()** é um gerador de todos as regiões 3x3 possiveis da imagem. Isto vai ser útil na implementação da parte de voltar a trás da class mais tarde.
   - Estes métodos vão ser aplicados para cada pixel da imagem output até esta estar completa.
   - Aqui, [neste código](https://replit.com/@Morgado/Conv-layer#cnnExample.py) é possivel testar a CNN usando o MNIST.  

---

## 4 - Pooling (junção)

 - Pixeis vizinhos numa imagem tendem a ter valores de intensidade parecidos, então as conv layers vão tipicamente produzir valores similares para os pixeis vizinhos como output, consequentemente, **grande parte da informação contida na conv layer é redundante**. Por exemplo, se usarmos um filtro que encontre arestas, e encontrar-mos um pixel que seja uma aresta, é muito provavel encontrar outro encostado que também faça parte da **mesma** aresta. Logo, não estamos a encontrar nada de novo.

 - **Camadas de pooling** (junção) resolvem este problema. O que elas fazem resume-me a reduzir o tamanho do input dado **valores de pooling** juntos ao input. O pooling é feito habitualmente com operações simples como *max, min* ou *média*, o gif abaixo mostra um exemplo de max pooling (Usa-se a operação max):

![pooling](https://victorzhou.com/ac441205fd06dc037b3db2dbf05660f7/pool.gif)

Para fazer um *max pooling* atravessamos a imagem de 2x2 a 2x2 blocos (porque o pool size é 2) e o valor final desse pixel output será o maior valor dentre os 4.

Basicamente, **o pooling divide o comprimento e largura do input pelo pool size**. Para o CNN do MNIST utilizaremos um max pool com um pool size de 2 logo após a conv layer inicial. a pooling layer vai transformar a imagem de 26x26x8 para 13x13x8, tal como mostra a próxima imagem:

![reduction](https://victorzhou.com/media/cnn-post/cnn-dims-2.svg)

### 4.1 - Implementar pooling

 - Iremos criar a classe **MaxPool2** com os mesmos métodos da nossa classe convoloção, o código pode ser encontrado [aqui](https://replit.com/@Morgado/Conv-layer#maxpool.py)

 - Na classe mostrada acima, temos que na linha principal (linha 30), utilizamos o np.amax nos eixos 0 e 1, pois só queremos encontrar o maior valor nesses eixos, que representam o comprimento e a largura.

 - Podemos testar isso [neste código](https://replit.com/@Morgado/Conv-layer#cnnExample.py)

 - Como expectável após a operação de pooling a imagem tem 13x13x8 tal como visto na imagem acima.

---

## 5 - Softmax

 - Para completar a CNN, é necessário dar-lhe a habilidade de fazer previsões. Faremos isso utilizando a última camada habitual de problemas de classificação múltipla: A camada **softmax**, uma camada totalmente conectada (densa) que usa a [função softmax](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer) como função de ativação. Em suma, a função softmax transforma números valores numéricos em probabilidades.
 
 - Mais informações sobre softmax [aqui](https://victorzhou.com/blog/softmax/) 

 - Camadas completamente conectadas (densas) têm todos os seus nodos conectados a todos os nodos da camada anterior.

### 5.1 - Uso (Usage)

 - Usaremos a camada de softmax com 10 nodos, cada um representando cada classe (neste caso temos 10 já que existem 10 dígitos). Cada nodo da camada irá estar conectado a todos os inputs. Após a transformação softmax ser aplicada, **o dígito representado pelo nodo com maior probabilidade** será o output da CNN.

![output](https://victorzhou.com/media/cnn-post/cnn-dims-3.svg)

### 5.2 - Perda de entropia cruzada (Cross-Entropy Loss)

 - Com isto surge a dúvida, *porque transformar os outputs em probabilidades se no final se apenas se vai escolher o maior?*
 - O que o softmax vai fazer é ajudar a **quantificar a certeza da previsão**, que é bastante útil para treinar e avaliar a CNN, mais ainda, usar softmax permite usar **cross-entropy loss**, que toma conta da certeza de cada previsão.
 - A cross-entropy loss calcula-se da seguinte forma:
    - *L = -ln( p<sub>c</sub> )*
 - Onde *c* é a classe correta (neste caso o dígito correto), *p<sub>c</sub>* é a probabilidade da classe c prevista. Uma loss menor é mais favorável que uma maior.
 - Num caso realista teriamos que: *p<sub>c</sub> = 0.8* e *c = -ln(0.8) = 0.223*.

### 5.2 - Implementar o softmax

 - É possível visualisar o código [aqui](https://replit.com/@Morgado/Conv-layer#softmax.py)
 - Algumas notas acerca do código:
   - O **.flatten()** serve para tornar o array num array unidimensional, para simplificar, já que não precisamos da forma, basta termos os pesos.
   - O **np.dot()** multiplica o input pelos pesos (*self.weights*) elemento por elemento e depois soma o resultado.
   - O **np.exp(x)** calcula os exponenciais usados para o Softmax ( *e<sup>x</sup>* )


## 6 - Conclusão

 - **CNN** são mais úteis em certos problemas, como classificação de imagem
 - **Conv layers** que convolvem filtros com imagens para produzir outputs mais úteis na classificação
 - **Pooling layers** que ajudam a descartar todas as características que não sejam úteis
 - **Softmax layer** que transforma um valor em probabilidade que permite calcular **cross-entropy loss**.