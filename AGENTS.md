# Orientações para agentes — Adult + q01 (funções de ativação)

## Estado deste contrato

Este arquivo é o contrato de trabalho do estudo dirigido de Reconhecimento de
Padrões 2026.1. Ele foi reiniciado em 21/07/2026 após a decisão do estudante de
substituir a q04 pela **q01 — activations**.

O escopo anterior sobre ativações aprendíveis foi abandonado antes de qualquer
implementação ou treinamento experimental. Não reutilize suas variáveis,
hipóteses, IDs ou conclusões. A pasta `experiments/` foi esvaziada e
`PROJECT_STATUS.md` foi removido pelo estudante por estarem obsoletos. Artefatos
novos foram inaugurados em 21/07/2026 para Adult + q01; não os confunda com os
arquivos descartados.

O contrato está com o **protocolo pré-experimental confirmado**. Em 21/07/2026,
o estudante confirmou as três variáveis, sua ordem, hipóteses, controles,
métricas, repetições e regras de análise descritas abaixo. A única recomendação
não adotada foi consultar o professor sobre o enquadramento acadêmico de V3; a
variável permanece selecionada, com esse risco documentado. A Variável 1 foi
executada e analisada localmente em
21/07/2026: 12 runs primárias, uma repetição determinística, tabela agregada e
três gráficos reproduzíveis. Esses artefatos foram versionados no commit de
fechamento da V1. Em 23/07/2026, o estudante solicitou o início da V2. Sua
infraestrutura pré-experimental foi implementada e validada localmente, com
quatro smokes de duas épocas, e versionada no commit `26e4473`. Em 24/07/2026,
as 12 runs científicas da V2 foram executadas e analisadas sem consulta ao teste
oficial. H2 ficou inconclusiva. Em 24/07/2026, o estudante autorizou o commit de
encerramento `de000de`, que versiona esses resultados. Depois desse fechamento,
o estudante solicitou o início da V3. Sua infraestrutura pré-runs foi preparada
e validada com três smokes, depois versionada no commit autorizado `2c15768`.
As nove runs científicas da V3 foram executadas e analisadas localmente, ainda
sem teste oficial: H3a não foi contradita, H3b e H3c foram sustentadas e H3d
ficou inconclusiva. Os artefatos de resultado foram versionados no commit de
encerramento autorizado pelo estudante. Depois de congelar as três análises, o
estudante autorizou a avaliação oficial: os 33 checkpoints primários foram
avaliados em 24/07/2026, sem novo treinamento e sem alterar os CSVs científicos.
Depois disso, a análise conjunta foi concluída localmente a partir dos
artefatos já salvos. Seu gerador não carrega o teste: produziu 33 pares brutos,
resumo das 11 configurações, retornos marginais e três gráficos reproduzíveis.
A seleção continuou baseada exclusivamente na validação. Uma suíte ampla de
desenvolvimento executada depois chamou o loader do Adult test em um teste de
schema e também fez treinos curtos; esse acesso não executou checkpoints
oficiais, não gerou métricas de modelos e não alterou decisões ou resultados.

Prazo informado no enunciado: **24 de julho de 2026**. A bonificação depende da
profundidade do estudo de uma a três variáveis: até 0,5, 1,0 e 1,5 ponto,
respectivamente.

## Função e escopo confirmado

- Tarefa: classificação binária do conjunto Adult em
  `exercises/task_binary_classification.py`.
- Questão: q01 em `exercises/q01_activations.py`.
- Questão central do enunciado: uma função de ativação pode ser responsável pelo
  melhor desempenho global do sistema em determinada tarefa?
- Objetivo experimental: comparar formulações relacionadas a funções de
  ativação quanto à acurácia e aos FLOPs instrumentados na classificação Adult.
- Baseline experimental confirmada: a MLP atual
  `Linear(108,64) -> ReLU -> Linear(64,2)`, identificada como `F-RELU`.
- Métricas: acurácia de treino, validação e teste. A métrica primária é a média
  da acurácia de validação na época 100 entre seeds; o checkpoint principal é o
  da época 100. O teste foi consultado somente na fase final confirmada.
- Quantidade confirmada: três variáveis, investigadas nesta ordem e de forma
  separada: família da ativação, curvatura da Softplus e profundidade linear sem
  ativação.

Não implemente BERT, q02, q03, q04 ou uma tarefa diferente. Não recupere os
artefatos experimentais descartados. Não trate o smoke como teste de hipótese.

## Motivações registradas

- Adult permanece por ser uma tarefa concreta, pequena o suficiente para CPU e
  diretamente comparável por acurácia e FLOPs.
- q01 substitui q04 porque trabalha conceitos fundamentais e explicáveis:
  valor da ativação, derivada local, propagação de gradiente e efeito da
  não linearidade sobre a representação oculta.
- A investigação deve conectar equação, implementação do `forward`, regra de
  `backward`, dinâmica de treinamento e desempenho medido. Não basta trocar o
  nome da função no código e apresentar a acurácia final.

## Estado técnico observado até 24/07/2026

### q01

`ExTensor` acrescenta três ativações básicas ao engine:

```text
sigmoid(x)  = 1 / (1 + exp(-x))
swish(x)    = x * sigmoid(x)
softplus(x) = log(1 + exp(x))
```

Os três métodos estão implementados com formulações numericamente estáveis,
backward analítico e custos instrumentados de 4, 5 e 3 FLOPs por elemento. Há
testes de valor, extremos, gradientes, grafo e FLOPs em
`test/test_q01_activations.py`. O comando da q01 gera
`experiments/plots/q01_activations.png`; Matplotlib consta em
`requirements.txt`. A antiga referência ao módulo inexistente
`exercises.check` foi removida.

Para a V2, `softplus_beta(x, beta)` foi acrescentada como operação separada,
com `beta` positivo e fixo, forward estável
`logaddexp(0, beta*x) / beta`, derivada `sigmoid(beta*x)` e custo instrumentado
de 5 FLOPs por elemento. O caminho antigo `softplus` permanece inalterado para
preservar a V1.

### Classificação Adult

A implementação mantém ReLU como padrão. A V1 escolhe a família da ativação; a
V2 usa explicitamente `activation="softplus_beta"` e um `activation_beta`:

```text
z = fc1(x)
h = activation(z)
logits = fc2(h)
```

Configuração observada:

- 108 entradas, `hidden=64` e duas saídas;
- 7.106 parâmetros escalares;
- full-batch, Adam, `lr=1e-2`, 100 épocas e validação de 20%;
- tensores no formato `(features, amostras)`;
- loss de treino antes do `Adam.step` e loss de validação depois do passo;
- métricas e FLOPs retornados em histórico estruturado;
- seeds independentes para split e inicialização do modelo;
- teste oficial carregado somente com `--evaluate-test`.

O estresse confirmou que `softmax` seguido de `log` produzia loss infinita e
gradientes NaN em logits `+/-1000`. A cross-entropy foi estabilizada uniformemente
com log-sum-exp antes da baseline. Os casos correto e incorreto agora produzem,
respectivamente, loss 0 e 2000, com gradientes finitos.

O split atual, com seed 0, contém 26.049 amostras de treino e 6.512 de
validação. O encoder é ajustado no arquivo oficial de treino inteiro antes do
hold-out; portanto usa features da futura validação, sem seus rótulos. Por
decisão confirmada, preserve esse comportamento em todas as configurações e
registre-o como limitação; não o corrija silenciosamente durante uma varredura.

### Testes e FLOPs

A suíte ampla de desenvolvimento alcançou historicamente:

```text
pytest -q --ignore=test/test_model.py
286 passed
```

Ela não é a rota segura de reprodução final: inclui treinos curtos e
`test_test_split_matches_train_feature_space`, que chama o loader do Adult
test. Quatro placeholders do Transformer continuam fora do escopo. Não afirme
que a suíte completa passa nem que esse comando preserva a reserva do teste.

A validação segura e focada da análise conjunta é:

```text
pytest -q test/test_plot_joint.py
6 passed
```

Antes do commit de fechamento, quatro preflights bloquearam corretamente
`requirements.txt` modificado como fonte crítica. Depois do versionamento, sem
enfraquecer a trava, a suíte focada do avaliador e da análise terminou com
`17 passed`. Os artefatos oficiais também continuaram válidos em
`--verify-only`.

O contador de FLOPs:

- contabiliza os forwards instrumentados;
- contabiliza backward apenas das multiplicações matriciais;
- não contabiliza backward elementwise;
- não contabiliza operações NumPy do Adam;
- trata uma operação elementwise não especializada como um FLOP por elemento.

A convenção confirmada já foi implementada e testada para as primitivas da q01
e para Softplus-beta.
Relate sempre “FLOPs instrumentados”, nunca tempo, energia, memória ou custo
completo.

### Executor da V1

`experiments/run_v1.py` executa uma configuração congelada por vez. Ele:

- rejeita runs científicas com código não versionado ou fora da branch;
- nunca carrega o teste oficial nesta fase;
- salva histórico e diagnósticos em JSONL e pesos finais em NPZ verificável;
- valida hashes de dados, split, pesos, configuração, ambiente e FLOPs;
- não inclui smoke em `results.csv`;
- bloqueia variantes até concluir e reproduzir a baseline ReLU.

Os diagnósticos usam somente a validação fixa e ficam fora da janela de FLOPs.
Foram produzidas 13 runs científicas válidas, todas sem consulta ao teste.

### Executor e resultado da V2

`experiments/run_v2.py` executa uma configuração por vez e
`experiments/run_v2_all.py` fixa a ordem das 12 runs. O validador confere
configuração, beta, seed, métricas, FLOPs, log e checkpoint antes de aceitar uma
run. `experiments/plot_v2.py` já prepara tabela, três gráficos e a decisão de H2.

Os quatro níveis passaram em smokes isolados de duas épocas, com
`850.711.121` FLOPs por época e `94.632.384` FLOPs de inferência na validação.
Os smokes não entraram em `results.csv`, não avaliaram o teste oficial e não são
evidência de H2. A infraestrutura foi versionada no commit `26e4473`.

As 12 runs científicas foram validadas em um único contexto experimental:

- validação média: beta 0,5 = 84,7461%; beta 1 = 84,8843%; beta 2 = 85,0379%;
  beta 5 = 85,1966%;
- todos os níveis custaram 85,0711121 GFLOPs instrumentados por run;
- o vencedor central foi beta 2 e o extremo foi beta 5;
- a diferença central menos extremo foi -0,1587 p.p., com sinais pareados
  positivo/negativo em 1/2 seeds; H2 ficou inconclusiva;
- beta 1 reproduziu exatamente pesos e métricas da Softplus da V1, mas teve
  custo maior pela instrumentação da multiplicação e divisão por beta;
- tabela, análise e três gráficos estão em `experiments/v2/`.

### Executor e resultado da V3

`AdultLinearClassifier` implementa separadamente as três profundidades sem
alterar a `AdultMLP` de V1/V2. O forward encadeia somente `Linear`; não existe
operação `Identity` nem FLOP atribuído a ela. `collapse_affine` verifica a forma
equivalente `W*x+b`.

`experiments/run_v3.py`, `run_v3_all.py` e `plot_v3.py` preparam as nove runs,
seus artefatos e H3a–H3d. Os três smokes de duas épocas passaram:

- L1: 218 parâmetros e 26.107.501 FLOPs instrumentados por época;
- L2: 7.106 parâmetros e 840.291.601 FLOPs por época;
- L3: 11.266 parâmetros e 1.544.654.481 FLOPs por época.

Os checkpoints, a equivalência afim e os custos de inferência foram validados.
Os smokes não criaram `results.csv`, não carregaram o teste oficial e não são
evidência das hipóteses. A infraestrutura foi versionada no commit `2c15768`.

A ponte entre `F-RELU` da V1 e `L2-IDENTITY` confere dados, split,
arquitetura linear, inicialização, treinamento, instrumentação e software. Ela
foi aceita pelos controles, mas atravessa commits e kernels diferentes
(Linux 7.1.3 na V1 e 7.1.4 na V3); isso não é um contexto literal idêntico.

As nove runs científicas foram validadas em um único contexto da V3:

- validação média: L1 = 84,5414%, L2 = 84,6744% e L3 = 84,6796%;
- FLOPs instrumentados por run: 2,6107501, 84,0291601 e 154,4654481 GFLOPs;
- L2 e L3 superaram L1 por apenas 0,1331 e 0,1382 p.p.; H3a não foi
  contradita, sem alegação de equivalência;
- parâmetros e FLOPs cresceram estritamente; H3b foi sustentada;
- o retorno caiu na ordem L1 > L2 > L3; H3c foi sustentada;
- ReLU superou L2 em 0,3737 p.p. e nas três seeds, abaixo da margem de
  0,5 p.p.; H3d ficou inconclusiva;
- tabela, análise e três gráficos regeneráveis estão em `experiments/v3/`.

As nove runs não carregaram o teste oficial. O encerramento da V3 foi
versionado com autorização do estudante.

### Avaliação oficial

`experiments/evaluate_official_test.py` validou os 33 checkpoints primários
antes de carregar Adult test. A repetição `F-RELU-s0-r2` foi excluída por não
ser uma run primária. O avaliador:

- reproduziu acurácia e FLOPs de validação de cada checkpoint;
- congelou SHA-256 e blob Git do `adult.test`;
- no fluxo controlado do avaliador, carregou o teste uma vez e fez um forward
  por checkpoint;
- não treinou, não aceitou subconjuntos e não alterou os CSVs das variáveis;
- registrou manifesto, log e CSV em `experiments/final_evaluation/`.

A avaliação `OFFICIAL-185889b9b944304ba514` contém 33 resultados válidos para
16.281 amostras. As maiores médias descritivas no teste foram beta 2 =
85,6458%, Swish = 85,5967% e beta 5 = 85,5414%. `F-SOFTPLUS` e
`S-BETA-1` produziram predições idênticas nas três seeds.

### Análise conjunta

`experiments/plot_joint.py` verifica os resultados oficiais já salvos, une-os
por `source_run_id` às 33 runs primárias e não chama o loader do Adult test.
Ele gera `experiments/final_analysis/` com os pares brutos, tabela agregada,
retornos marginais, análise escrita e três gráficos.

Todas as decisões usam a validação da época 100 e FLOPs instrumentados de
treinamento. O teste permanece apenas descritivo:

- Pareto global: `L1-DIRECT`, `L2-IDENTITY`, `F-RELU` e `S-BETA-5`;
- melhor retorno por FLOP: `L1-DIRECT`, com 3,454657 p.p./GFLOP;
- melhor validação: `S-BETA-5`, com 85,1966%;
- escolha sob o orçamento de 84,2375505 GFLOPs da ReLU: `F-RELU`;
- V3 mudou muito o custo e pouco o desempenho; V2 alterou a acurácia sem mudar
  o custo instrumentado entre seus níveis;
- a maior média no teste foi `S-BETA-2`, mas isso não mudou a seleção.

Pelo limiar de 0,5 p.p., nenhum ganho entre vizinhos da Pareto foi relevante.
Os retornos marginais globais não decresceram monotonicamente; portanto não há
um único cotovelo suave. A queda é clara dentro de V3, especialmente de L2 para
L3. A suíte ampla de desenvolvimento alcançou 286 testes permitidos, mas não
integra a rota segura de reprodução pelos acessos e treinos descritos acima.
`test/test_model.py` continua excluído pelos placeholders fora do escopo.

### Resultado da Variável 1

- acurácia média: Swish 85,1351%, ReLU 85,0481%, Softplus 84,8843% e
  Sigmoid 84,7461%;
- melhor retorno por FLOP: ReLU; maior acurácia média: Swish;
- H1a e H1b inconclusivas; H1c sustentada;
- Pareto pelas médias: ReLU e Swish; Sigmoid e Softplus dominadas;
- gráficos e tabela regeneráveis com `python -m experiments.plot_v1`.

O desvio-padrão usa somente três seeds e não é intervalo de confiança. Como o
split é fixo, ele descreve variação de inicialização, não de amostragem. A
fronteira de Pareto usa as médias observadas e não prova significância.

## Fundamentos que o estudante deve conseguir explicar

Seja `g = dL/dy` o gradiente que chega à ativação.

### Sigmoid

```text
s(x)  = 1 / (1 + exp(-x))
s'(x) = s(x) * (1 - s(x))
dL/dx = g * s(x) * (1 - s(x))
```

A saída fica entre 0 e 1. Para entradas de grande magnitude, a derivada tende a
zero e pode produzir saturação. O forward deve ser numericamente estável para
valores positivos e negativos extremos.

### Swish / SiLU

```text
w(x)  = x * s(x)
w'(x) = s(x) + x * s(x) * (1 - s(x))
dL/dx = g * w'(x)
```

É suave, não limitada no lado positivo e permite pequenas saídas negativas. Sua
derivada pode preservar gradiente onde a ReLU zeraria a unidade.

### Softplus

```text
p(x)  = log(1 + exp(x))
p'(x) = s(x)
dL/dx = g * s(x)
```

É uma aproximação suave da ReLU, sempre positiva. A forma ingênua pode sofrer
overflow; uma implementação estável pode usar uma formulação NumPy equivalente,
como `logaddexp(0, x)`, mantendo a derivada documentada.

### Comparação com ReLU

```text
r(x)  = max(0, x)
r'(x) = 0 para x <= 0; 1 para x > 0
```

ReLU é barata e esparsa, mas zera valor e gradiente no semieixo negativo. As
ativações suaves fazem mais operações e alteram tanto a escala das representações
quanto o fluxo de gradientes. Esses mecanismos fundamentam hipóteses; seus
efeitos no Adult só podem ser chamados de resultados depois das medições.

## Requisitos para implementar q01

- manter `ExTensor` compatível com o `Tensor` do engine;
- construir cada saída com `self` como pai e um rótulo `_op` inequívoco;
- preservar `requires_grad`;
- acumular gradientes com `+=`, nunca sobrescrevê-los;
- não desprender o grafo usando `.data` para produzir saídas intermediárias;
- usar `.data` somente para calcular o valor NumPy da primitiva e sua derivada
  local dentro do fechamento de backward;
- garantir estabilidade numérica em entradas extremas;
- permitir aplicar os métodos de `ExTensor` a um `Tensor` produzido por
  `nn.Linear`, conforme o uso não vinculado já demonstrado na própria q01;
- chamar, por exemplo, `ExTensor.swish(z)` sobre o `Tensor` conectado ao grafo;
  não criar `ExTensor(z)`, pois essa reconstrução copiaria os dados e perderia os
  pais que ligam `z` à primeira camada linear;
- acrescentar contagem de FLOPs coerente e documentada para as três primitivas;
- criar gradient checks por diferenças finitas para valor, entrada e cadeias
  com operações anteriores/posteriores;
- testar dtype, forma, `requires_grad`, acumulação e finitude.

Não substitua a implementação pedagógica por uma biblioteca de deep learning.
Preserve NumPy/CPU.

## Variáveis confirmadas em 21/07/2026

O estudante escolheu as três variáveis abaixo, confirmou sua ordem e aprovou as
comparações e hipóteses em 21/07/2026. Elas são decisões pré-experimentais, não
resultados. Use `delta=0,5` ponto percentual e a regra confirmada neste contrato
para avaliá-las somente depois das medições.

### Variável 1 — família da ativação oculta

Manter a arquitetura `108 -> 64 -> 2` e alterar somente a transformação entre
as duas camadas lineares:

| ID | Configuração | Formulação de `h` |
|---|---|---|
| `F-RELU` | ReLU, referência | `max(0, z)` |
| `F-SIGMOID` | Sigmoid | `1 / (1 + exp(-z))` |
| `F-SWISH` | Swish / SiLU | `z * sigmoid(z)` |
| `F-SOFTPLUS` | Softplus padrão | `log(1 + exp(z))` |

Todas usam `z = fc1(x)`, `hidden=64` e 7.106 parâmetros escalares. O objetivo é
comparar diretamente ativações com esparsidade, saturação e suavidade distintas.

Hipóteses confirmadas: pelo menos uma função suave superará a ReLU por `delta`,
possivelmente por preservar gradientes no semieixo negativo; `F-SIGMOID` ficará
abaixo de `F-RELU` por `delta`, devido ao risco de saturação e à representação
não centrada; a ReLU terá o menor custo elementwise instrumentado.

Além de loss, acurácia e FLOPs, registre fora da janela medida estatísticas
comparáveis de `z`, de `h` e da derivada local: média, desvio, percentis, fração
de saídas próximas de zero e um indicador de saturação apropriado à função.

### Variável 2 — curvatura da Softplus

Fixar `hidden=64` e usar a família:

```text
softplus_beta(z) = log(1 + exp(beta * z)) / beta, beta > 0
```

com implementação numericamente estável equivalente a
`logaddexp(0, beta*z) / beta`.

| ID | `beta` | Interpretação |
|---|---:|---|
| `S-BETA-0.5` | 0,5 | transição mais suave |
| `S-BETA-1` | 1 | Softplus padrão e referência |
| `S-BETA-2` | 2 | aproximação mais marcada da ReLU |
| `S-BETA-5` | 5 | aproximação ainda mais próxima da ReLU |

`beta` é uma constante fixa da configuração, não um parâmetro aprendível. A
derivada local é `sigmoid(beta*z)`. O objetivo é verificar se o grau de suavidade
altera otimização e generalização sem mudar a arquitetura ou a quantidade de
parâmetros.

Hipótese confirmada: o melhor resultado médio entre `beta=1` e `beta=2`
superará por `delta` o melhor resultado médio entre os extremos `beta=0,5` e
`beta=5`. Escolha o vencedor de cada grupo pela média das três seeds e compare
depois os IDs vencedores também pelo sinal pareado nas seeds. Todos os níveis
executam a mesma forma algébrica e deverão ter o mesmo custo pela convenção
analítica escolhida. A aproximação à ReLU será tratada como verificação
mecânica, não como resultado incerto. Observe que `beta` também muda
`softplus_beta(0) = log(2)/beta`, portanto curvatura, deslocamento e escala não
ficam completamente separados.

`S-BETA-1` coincide matematicamente com `F-SOFTPLUS`. Para reutilizar a execução,
`F-SOFTPLUS` deve chamar literalmente o mesmo caminho Softplus-beta com
`beta=1`, inclusive multiplicação, divisão e contagem. Além disso, código,
commit/estado, seed, split, hiperparâmetros, instrumentação e diagnósticos devem
ser idênticos. Como Softplus-beta só deve ser implementada depois de fechar V1,
o padrão é executar e contar as duas runs separadamente; qualquer reutilização
exige prova de identidade completa.

### Variável 3 — profundidade linear sem ativação

Não aplicar função de ativação entre as camadas. O fator declarado é a
profundidade/fatorização linear; número de parâmetros, sobreparametrização e
geometria de otimização também mudam e não podem ser separados desse fator:

| ID | Arquitetura | Parâmetros escalares |
|---|---|---:|
| `L1-DIRECT` | `Linear(108,2)` | 218 |
| `L2-IDENTITY` | `Linear(108,64) -> Linear(64,2)` | 7.106 |
| `L3-IDENTITY` | `Linear(108,64) -> Linear(64,64) -> Linear(64,2)` | 11.266 |

O estudante confirmou 64 unidades nas duas camadas intermediárias de
`L3-IDENTITY`. A contagem de 11.266 parâmetros vale para essa forma.

“Sem ativação” significa passar `z` diretamente à próxima camada; não crie uma
operação `Identity` artificial nem atribua FLOPs a uma cópia inexistente. A
identidade é `g(z)=z`, mas a rede completa não é necessariamente `f(x)=x`.
Compondo camadas afins, cada configuração continua equivalente a uma única
função afim:

```text
W_n(...W_2(W_1*x + b_1) + b_2...) + b_n = W*x + b
```

Na arquitetura confirmada, as larguras intermediárias são pelo menos dois. Assim,
as três configurações têm a mesma classe de funções afins para duas saídas,
embora tenham parametrizações, inicializações e trajetórias de otimização
diferentes.

Hipóteses confirmadas: nem `L2-IDENTITY` nem `L3-IDENTITY` melhorará
`L1-DIRECT` por `delta`; parâmetros e FLOPs crescerão na ordem da profundidade;
o retorno seguirá `R(L1-DIRECT) > R(L2-IDENTITY) > R(L3-IDENTITY)`; e
`F-RELU` superará `L2-IDENTITY` por `delta`. Um resultado diferente pode
decorrer da parametrização/otimização implícita, não de aumento da classe
funcional. O contraste com `F-RELU` só é válido com os demais controles
idênticos.

Modelos com formas diferentes não podem compartilhar todos os pesos iniciais.
A seed garante repetibilidade, não equivalência tensor a tensor. Não conclua que
mais camadas “não servem” fora das redes lineares e deste protocolo Adult.
O professor pode interpretar profundidade como uma variável arquitetural
genérica, não como terceira variável de q01. O estudante decidiu prosseguir sem
consultar o professor sobre esse enquadramento. Preserve o risco e nunca afirme
que V3 foi aceita pelo professor ou que sua bonificação está garantida.

## Protocolo pré-experimental confirmado em 21/07/2026

- `L3-IDENTITY`: larguras intermediárias 64 e 64;
- treinamento: full-batch, Adam, `lr=1e-2`, 100 épocas e validação de 20%;
- split: gerar/materializar uma vez com seed 0, preservar índices e hash;
- inicializações: seeds de modelo `0`, `1` e `2`, aplicadas imediatamente antes
  de construir o modelo sem regenerar o split;
- métrica primária: média da acurácia de validação na época 100;
- checkpoint principal: época 100; melhor época apenas como análise secundária;
- margem relevante: `delta=0,5` ponto percentual;
- hipótese direcional sustentada: diferença média de pelo menos `delta` e mesmo
  sinal em pelo menos duas das três seeds; refutada quando o efeito inverso
  satisfizer a mesma regra; demais casos inconclusivos;
- desempate: menor FLOP total e, depois, menor quantidade de parâmetros;
- run inválida: somente erro, NaN/Inf, split/hash incorreto, instrumentação
  errada ou violação do protocolo; desempenho ruim mas finito continua válido;
- teste oficial: somente após congelar protocolo e seleção por validação; nessa
  fase, reportar todas as configurações válidas sem revisar decisões;
- pré-processamento: preservar o loader atual e documentar como limitação o
  encoder ajustado antes do hold-out;
- cross-entropy: reproduzir formalmente o teste de estresse e aplicar uma forma
  estável de maneira uniforme antes da baseline se a falha for confirmada;
- visualização: incluir os gráficos da q01 e registrar Matplotlib em
  `requirements.txt` quando a implementação começar;
- checkpoints: salvar pesos finais, configuração e hash de cada run;
- proveniência: o estudante autorizou o Passo 0 e o commit da implementação e
  smoke de V1. O rastreamento remoto local aponta para esse commit e seu reflog
  registra `update by push`, embora nenhum comando de push tenha sido executado
  pelo agente nesta etapa. Commits futuros e qualquer push exigem autorização;
- enquadramento de V3: a consulta ao professor não será realizada por decisão do
  estudante; o risco acadêmico permanece documentado.

Detalhes operacionais:

- reprodução de `F-RELU`: mesmos split/pesos iniciais, FLOPs exatos e
  pesos/métricas finais iguais com `rtol=1e-12`, `atol=1e-12` no mesmo ambiente;
- diagnósticos: antes do treino e após épocas `1`, `25`, `50`, `75` e `100`;
- próximo de zero: `abs(h) <= 1e-6`; para ReLU, registre também zero exato;
- saturação da Sigmoid: saída `<=0,05` ou `>=0,95`;
- estresse da cross-entropy: logits `(+1000,-1000)` com alvo da classe de menor
  logit e o caso simétrico; qualquer loss ou gradiente NaN/Inf aciona a
  estabilização uniforme antes da baseline;
- smoke: duas épocas sobre o split completo e fixo, sem uso científico.

### Convenção confirmada de FLOPs

Custos do forward por elemento:

| Operação | FLOPs instrumentados |
|---|---:|
| identidade/ausência de ativação | 0 |
| ReLU | 1 |
| Sigmoid estável | 4 |
| Swish | 5 |
| Softplus padrão da q01 | 3 |
| Softplus-beta, inclusive `beta=1` | 5 |

Cada operação escalar da formulação analítica vale uma unidade, inclusive
`exp` e `log`; isso não representa seu custo físico real. Comparações e ramos de
estabilidade não recebem peso adicional. Matmuls preservam
`2 * elementos_da_saida * dimensão_compartilhada`. Backward elementwise e Adam
permanecem fora do contador.

Em cada época, reinicie o contador antes do forward de treino e leia-o após a
loss de validação. A janela inclui forward/loss de treino, backward
instrumentado, passo do Adam sem custo registrado e forward/loss de validação.
Exclua diagnósticos, acurácia e teste. Meça inferência por amostra separadamente
com apenas um forward no conjunto de validação, dividindo pelo número de
amostras.

### Eficiência confirmada

- retorno por FLOP:
  `(acurácia_val em p.p. - acurácia_majoritária_val em p.p.) / GFLOPs_totais`;
- retorno marginal: `delta_acurácia / delta_GFLOPs` dentro de cada variável e
  entre vizinhos da fronteira global, somente para `delta_GFLOPs > 0`;
- custos iguais: não calcular retorno marginal; comparar diretamente acurácia;
- orçamento fixo: FLOPs totais de `F-RELU` em 100 épocas;
- retornos decrescentes: aplicar `delta=0,5` p.p. como limiar de ganho relevante;
- preservar sempre pares brutos e a fronteira de Pareto.

## Controles comuns confirmados

Salvo quando a própria variável selecionada alterar explicitamente um item,
preserve:

- Adult, arquivos brutos e loader atual; documente como limitação o encoder
  ajustado antes do hold-out;
- mesmos índices de treino e validação, gerados ou recarregados com seed 0 e
  identificados por hash;
- seed 0 obrigatória entre as seeds de modelo; para cada run, aplique a seed
  aprovada imediatamente antes de construir o modelo, sem regenerar o split;
- mesma inicialização das camadas lineares quando suas formas forem iguais;
- full-batch, Adam, `lr=1e-2`, 100 épocas e validação de 20%;
- 108 entradas, `hidden=64`, duas saídas e 7.106 parâmetros escalares;
- cross-entropy na forma uniforme definida antes da baseline, procedimento de
  avaliação e ordem dos dados;
- janela de FLOPs e formato de logs;
- durante desenvolvimento e seleção, uso apenas de treino e validação; consulte
  o teste somente na fase final confirmada e para todas as configurações válidas;
- ReLU como comportamento padrão da task fora de uma configuração explícita;
- diagnósticos fora da janela de FLOPs ou idênticos em todas as configurações.

Cada execução futura deve registrar ID, tarefa, variável/nível, seed, repetição,
commit/estado da árvore, comando, ambiente, hashes de dados/split/pesos iniciais,
hiperparâmetros, parâmetros, losses, acurácias, FLOPs, checkpoint/hash e
observações.

## Fluxo de trabalho obrigatório

1. auditar q01, Adult e a instrumentação — concluído em 21/07/2026;
2. apresentar seis candidatas e receber a escolha — concluído em 21/07/2026;
3. registrar as três variáveis e reescrever `Passo-a-passo.md` — concluído em
   21/07/2026;
4. confirmar o Portão 0, exceto a recomendação de consulta sobre V3 — concluído
   em 21/07/2026;
5. inaugurar os novos artefatos experimentais e registrar hipóteses formais —
   concluído em 21/07/2026;
6. implementar e validar Sigmoid, Swish e Softplus padrão — concluído em
   21/07/2026;
7. integrar e testar somente V1 na classificação Adult — concluído em
   21/07/2026;
8. fazer smoke de V1 e reproduzir `F-RELU` sem consultar o teste — concluído em
   21/07/2026;
9. preparar e validar logs, checkpoints e travas do executor — concluído
   e versionado em 21/07/2026;
10. executar e encerrar a análise da Variável 1 — concluído e versionado em
    21/07/2026;
11. implementar, testar, executar e encerrar V2 — concluído e versionado no
    commit de encerramento autorizado em 24/07/2026;
12. implementar, testar, executar e encerrar V3 — concluído e versionado com
    autorização em 24/07/2026;
13. realizar a avaliação de teste na fase aprovada — concluída e versionada em
    24/07/2026, sem novo treinamento;
14. realizar a análise conjunta de trade-offs — concluída e versionada em
    24/07/2026; o gerador não treina nem carrega o teste;
15. atualizar README, dependências, uso de IA, vídeo e reprodução — concluído
    e versionado com autorização em 24/07/2026; revisão do estudante e gravação
    pendentes.

Não implemente as três variáveis simultaneamente. Uma implementação fundamental
compartilhada da q01 pode ser concluída antes das varreduras, mas cada fator
experimental deve permanecer isolado.

## Protocolo experimental mínimo

- reproduza a baseline confirmada antes das variantes;
- varie uma variável por vez;
- registre hipóteses e critérios antes das execuções;
- mantenha os controles comuns, salvo o fator declarado;
- assegure split e inicialização comparáveis;
- nas execuções de desenvolvimento, registre treino e validação; depois de
  congelar a seleção e conforme a fase previamente definida, reporte também a
  acurácia de teste no escopo confirmado;
- nunca use o teste para escolher configuração ou alterar o protocolo;
- registre FLOPs por época, total bruto e GFLOP;
- preserve todas as configurações válidas, inclusive resultados negativos;
- não misture runs de commits, instrumentações ou splits diferentes;
- registre e explique falhas em vez de repeti-las silenciosamente.

## Análise obrigatória de trade-offs

Depois das três varreduras, reúna todas as configurações válidas numa tabela e
num gráfico de desempenho versus FLOPs. Preserve os pares brutos e responda, com
base nas medições:

1. qual configuração apresenta o melhor retorno por FLOP;
2. a partir de onde aparecem retornos decrescentes;
3. qual variável altera muito os FLOPs e pouco o desempenho, e se existe o
   comportamento contrário;
4. qual configuração escolher sob um orçamento fixo em FLOPs.

Identifique configurações dominadas e a fronteira de Pareto quando os dados
permitirem. Defina antes dos resultados a métrica, a fórmula de retorno, o
orçamento e o que constitui ganho relevante. Se a evidência não sustentar uma
resposta, classifique-a como inconclusiva.

## Artefatos da nova jornada

Não recrie os arquivos antigos. Depois da escolha das variáveis e do início
material do trabalho experimental, inaugure artefatos novos:

```text
experiments/
  checkpoints/
  configs/
  logs/
  results.csv
  hypotheses.md
  analysis.md
  ai_usage.md
  video_evidence.md
  plots/
PROJECT_STATUS.md
```

`PROJECT_STATUS.md` deve permanecer curto e registrar objetivo, decisões,
trabalho concluído, arquivos afetados, comandos/evidências, riscos, pendências,
próximo passo e progresso dos entregáveis. Não marque propostas como concluídas.
Se informar percentual, mostre o critério usado.

Tabelas e gráficos devem ser regeneráveis a partir dos dados brutos. Nunca
edite manualmente um resultado para fazê-lo coincidir com a hipótese.

## Evidências obrigatórias para o vídeo

Quando o trabalho experimental começar, mantenha
`experiments/video_evidence.md` como mapa vivo dos 12 requisitos explícitos:

1. uso de IA e verificação pelo estudante;
2. tarefa Adult, dados, entradas, saídas, objetivo e métrica;
3. baseline e justificativa;
4. variáveis e controles;
5. formulações e efeito no processamento;
6. arquitetura, treinamento e fundamentos;
7. configurações, repetições, seeds e métricas;
8. resultados completos;
9. desempenho versus FLOPs;
10. hipóteses versus resultados;
11. dificuldades, erros e mudanças;
12. reprodução.

Para cada item, ligue afirmações apresentáveis a caminhos, runs/commits,
limitações e uma explicação que o estudante consiga defender. Diferencie plano,
resultado medido e interpretação. Antes da gravação, transforme o mapa num
roteiro de até 20 minutos. Não afirme compreensão ou validação humana sem
confirmação do estudante.

## Validação mínima antes de qualquer treino completo

- testes de valor contra NumPy em domínio regular e extremo;
- gradient check por diferenças finitas de sigmoid, swish, softplus e
  Softplus-beta;
- teste de acumulação de gradiente e `requires_grad=False`;
- teste de estabilidade: nenhuma saída/derivada NaN ou Inf em casos definidos;
- comparação visual opcional das funções e derivadas;
- integração de cada ativação com `AdultMLP` e cross-entropy;
- teste de estresse e finitude da cross-entropy sob logits extremos;
- saída `(2, batch)` e contagem de parâmetros preservadas;
- um passo de Adam com gradientes finitos;
- teste da contagem de FLOPs por ativação;
- smoke test curto antes da baseline de 100 épocas;
- confirmação de que a baseline/default com ReLU continua reproduzível.

Rota segura para validar e regenerar os resultados publicados:

```bash
python -m experiments.evaluate_official_test --verify-only
pytest -q test/test_plot_joint.py
python -m exercises.q01_activations
python -m experiments.plot_v1
python -m experiments.plot_v2
python -m experiments.plot_v3
python -m experiments.plot_joint
python -m experiments.evaluate_official_test --verify-only
```

Essa sequência não treina e não carrega novamente o Adult test. O comando amplo
de pytest e `task_binary_classification` continuam úteis no desenvolvimento,
mas não pertencem à reprodução final segura.

## Fontes de verdade

Use, nesta ordem:

1. orientação mais recente do estudante ou professor;
2. `Passo-a-passo.md` depois de sua reformulação para q01;
3. `Estudo_Dirigido_RP___2026_1.pdf`;
4. código e testes para comportamento técnico;
5. este arquivo para o protocolo acordado;
6. `README.md` como entregável público.

O `Passo-a-passo.md` foi refeito em 21/07/2026 para Adult + q01. Menções à q04
devem limitar-se à nota histórica de que esse escopo foi abandonado antes de
implementação e treinamento experimental.

Exponha conflitos e diferencie requisito do professor, decisão do estudante,
observação do código e recomendação do agente.

## Convenções e limites

- preserve NumPy/CPU;
- justifique e registre toda dependência em `requirements.txt`;
- não altere datasets brutos;
- não duplique a classificação para cada configuração;
- faça mudanças pequenas, configuráveis e compatíveis com trabalho do estudante;
- não execute variantes antes da baseline;
- não use resultados para escolher hipóteses ou níveis retrospectivamente;
- não faça commit, push, publicação ou alteração externa sem solicitação
  explícita.

## IA e autoria

Quando os novos artefatos forem inaugurados, registre contribuições materiais da
IA em `experiments/ai_usage.md`: data, objetivo, sugestão ou código, arquivos
afetados, verificação executada, correções e decisão final do estudante.

Explique em português claro. Nunca invente execução, métrica, FLOPs, citação ou
validação humana. O agente pode estruturar, implementar e revisar, mas o
estudante deve conseguir explicar e assumir as decisões finais.

## Critério de conclusão futuro

- q01 implementada com forwards e backwards corretos e estáveis;
- gradient checks e integração com a MLP validados;
- baseline e variantes reproduzíveis;
- três variáveis estudadas isoladamente e em profundidade;
- hipóteses e controles preservados antes dos resultados;
- acurácias e FLOPs completos em logs, tabela e gráfico;
- retorno por FLOP, Pareto, orçamento e retornos decrescentes analisados;
- limitações e resultados negativos documentados;
- README e `requirements.txt` fiéis ao experimento;
- uso de IA e dificuldades registrados;
- 12 requisitos do vídeo ligados a evidências verificáveis;
- repositório e vídeo de até 20 minutos prontos para entrega.

Neste momento, a Variável 1 está concluída e versionada no commit `07243dc`,
sem teste oficial: 13 runs válidas, hipóteses avaliadas, tabela e três gráficos
reproduzíveis. A V2 possui 12 runs válidas, tabela, análise, três gráficos e H2
inconclusiva, tudo ainda sem teste oficial. Seus resultados são versionados pelo
commit `de000de`. A V3 possui protocolo, implementação, executor, análise
pré-registrada e infraestrutura versionada no commit `2c15768`. Suas nove runs,
tabela, análise e três gráficos estão concluídos localmente, sem teste oficial.
H3a não foi contradita, H3b/H3c foram sustentadas e H3d ficou inconclusiva.
O encerramento foi versionado com autorização. A consulta ao professor sobre V3
foi rejeitada pelo estudante e permanece como risco. A avaliação oficial dos
33 checkpoints primários foi concluída e auditada localmente, com um único load
controlado do teste e nenhum treinamento. A análise conjunta também foi
concluída localmente; seu gerador não acessou o teste: L1 teve o melhor retorno,
ReLU foi a escolha sob orçamento e a Pareto global terminou em beta 5. A suíte
ampla de desenvolvimento acessou depois o loader do teste para validar schema,
sem produzir métricas ou mudar conclusões. README, dependências, mapa e roteiro
do vídeo foram preparados localmente. O próximo passo é a revisão do estudante,
gravação e inclusão do link. Avaliador, resultados oficiais, análise conjunta e
materiais de entrega foram incluídos no commit de fechamento autorizado em
24/07/2026. Nenhum push foi autorizado.
