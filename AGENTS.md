# Orientações para agentes — Adult + ativações aprendíveis

## Função e escopo

Este arquivo é o contrato de trabalho para o estudo dirigido de Reconhecimento
de Padrões 2026.1. O escopo confirmado é investigar ativações aprendíveis
(`q04`) na classificação binária do conjunto Adult.

O objetivo é verificar se uma combinação aprendida de ReLU, GELU e SiLU melhora
a acurácia e se o resultado compensa o custo em FLOPs. O estudante deve conseguir
explicar a formulação, a implementação, o protocolo e os limites das conclusões.
Não desvie para outras tarefas, questões ou para a implementação do BERT.

Prazo informado no enunciado: **24 de julho de 2026**. A bonificação depende da
profundidade do estudo de uma a três variáveis: até 0,5, 1,0 e 1,5 ponto,
respectivamente.

## Decisões confirmadas

- Tarefa: classificação Adult em `exercises/task_binary_classification.py`.
- Questão: q04 em `exercises/q04_learnable_activations.py`.
- Motivo da tarefa: problema tangível e testável, com acurácia e FLOPs
  diretamente comparáveis.
- Motivo da questão: aproveita o conhecimento prévio sobre ativações e o amplia
  para parâmetros aprendidos durante o treinamento.
- Contexto teórico: a referência apresenta a ideia em modelos de linguagem; o
  estudo avaliará sua aplicação em classificação tabular. Não alegue ineditismo
  nem transfira conclusões do artigo sem evidência no Adult.
- Variáveis confirmadas, nesta ordem: formulação da ativação oculta,
  parametrização/restrição dos coeficientes e largura da camada oculta.
- As três variáveis serão estudadas separadamente; conclua implementação,
  validação, execução e análise de uma antes de iniciar a seguinte.

Na baseline:

```text
z = fc1(x)
h = ReLU(z)
logits = fc2(h)
```

Na q04, a ReLU é substituída por uma mistura global aprendível:

```text
h = alpha_relu * ReLU(z) + alpha_gelu * GELU(z) + alpha_silu * SiLU(z)
```

ou por sua forma normalizada:

```text
pi = softmax([beta_relu, beta_gelu, beta_silu])
h = pi_relu * ReLU(z) + pi_gelu * GELU(z) + pi_silu * SiLU(z)
```

Os coeficientes são globais e compartilhados entre amostras e unidades. Esta é
uma Learnable Activation independente da entrada, não uma mistura dinâmica
condicionada por amostra.

## Decisões pendentes

- Hipóteses registradas antes das execuções.
- Configurações experimentais e baseline executada de referência.
- Métrica primária para selecionar configurações.
- Repetições e sementes adicionais; `cpu.set_seed(0)` é obrigatório.
- Definição operacional de retorno por FLOP.
- Orçamento computacional de referência.

Não feche uma dessas decisões sem confirmação do estudante quando ela mudar o
rumo da investigação.

## Plano pré-experimental das variáveis

As seções abaixo registram ideias e hipóteses iniciais. **Nada foi implementado,
executado ou validado.** Não apresente efeitos esperados como resultados e não
combine níveis de variáveis diferentes durante estas três etapas.

### Controles comuns

Salvo quando for a própria variável investigada, preserve:

- Adult, pré-processamento e os mesmos índices de treino e validação;
- `cpu.set_seed(0)` antes de cada execução independente;
- full-batch, Adam, `lr=1e-2`, 100 épocas e validação de 20%;
- 108 entradas, duas saídas, loss e procedimento de avaliação;
- mesma inicialização das camadas lineares quando as formas forem iguais;
- mesma janela de medição dos FLOPs e mesmo formato de logs;
- seleção por validação e uso do teste apenas na avaliação final.

Registre um hash ou identificador do split. Não acrescente forwards de
diagnóstico dentro da região de FLOPs apenas em algumas configurações: calcule
estatísticas fora da janela medida ou aplique o mesmo diagnóstico a todas. Para
cada execução, registre ID, commit, seed, split, configuração, parâmetros,
loss, acurácias, FLOPs e observações numéricas.

### Variável 1 — formulação da ativação oculta

Manter `hidden=64` e alterar somente como `z = fc1(x)` produz `h`:

| ID | Configuração | Formulação |
|---|---|---|
| `A0` | ReLU fixa | `h = ReLU(z)` |
| `A1` | GELU fixa | `h = GELU(z)` |
| `A2` | SiLU fixa | `h = SiLU(z)` |
| `A3` | mistura uniforme fixa | `h = (ReLU(z) + GELU(z) + SiLU(z)) / 3` |
| `A4` | mistura normalizada aprendível | `pi = softmax(beta)` e `h = sum(pi_k * phi_k(z))`, com `beta=(0,0,0)` |

Objetivo: distinguir três efeitos — escolher outra ativação fixa, combinar três
ativações e aprender os pesos da combinação. `A0` é a baseline; `A1` e `A2` são
controles individuais. `A3` e `A4` começam com a mesma função e pesos efetivos
`(1/3, 1/3, 1/3)`, tornando `A3 x A4` a comparação central para o efeito do
aprendizado.

Hipótese inicial: `A4` poderá igualar ou superar a melhor ativação fixa e `A3`
em acurácia de validação. `A3` e `A4` deverão ter custos semelhantes, mas não
idênticos; ambos calculam três ativações, enquanto `A4` também calcula o softmax
e atualiza três parâmetros.

Implementação futura exigida:

- tornar a ativação da mesma `AdultMLP` configurável, sem duplicar a task;
- preservar ReLU como comportamento padrão;
- implementar a mistura fixa com constantes, sem `Parameter`;
- integrar `NormalizedLearnableActivation` como submódulo e confirmar que os
  três `beta` aparecem em `model.parameters()`;
- testar numericamente que `A3` e `A4` produzem a mesma saída inicial;
- registrar, em `A4`, `beta`, pesos efetivos `pi` e contribuição média absoluta
  de cada termo `pi_k * phi_k(z)`.

Cuidados: `A0`–`A3` têm 7.106 parâmetros e `A4` tem 7.109. GELU e SiLU podem
ser muito correlacionadas no domínio visitado, deixando os coeficientes pouco
identificáveis. Um `pi` maior não prova maior importância sem considerar a
escala de `phi_k(z)`.

### Variável 2 — parametrização e restrição dos coeficientes

Manter `hidden=64`, ReLU, GELU e SiLU e comparar duas formas aprendíveis que
representam a mesma função inicial:

| ID | Configuração | Inicialização efetiva |
|---|---|---|
| `C0` | normalizada por softmax | `beta=(0,0,0)`, logo `pi=(1/3,1/3,1/3)` |
| `C1` | livre, sem normalização | `alpha=(1/3,1/3,1/3)` |

Objetivo: comparar a formulação normalizada, com pesos positivos somando 1, à
formulação livre, que permite pesos negativos e escala global variável. A
comparação envolve tanto a restrição quanto a geometria de parametrização do
softmax; não atribua o efeito somente à restrição.

Hipótese inicial: `C0` poderá preservar melhor a escala e produzir pesos mais
interpretáveis; `C1` terá maior flexibilidade e poderá reduzir mais a loss de
treino, mas também poderá gerar coeficientes negativos, crescimento de escala
ou pior generalização. Os FLOPs devem ser próximos, não presumidos idênticos.

Implementação futura exigida:

- permitir escolher as duas classes da q04 na mesma MLP;
- preservar o default pedagógico `alpha=(1,1,1)` da classe existente e expor
  uma inicialização configurável de `1/3` para o experimento Adult;
- confirmar que `C0` e `C1` produzem a mesma saída inicial para a mesma entrada;
- confirmar que os três coeficientes são atualizados pelo Adam;
- registrar `beta` e `pi` em `C0`; em `C1`, registrar `alpha`, sua soma, sinais,
  norma e a escala da saída;
- registrar as contribuições efetivas `peso_k * phi_k(z)`, não somente os
  coeficientes finais.

`C0` coincide com `A4`. Uma execução só pode ser reaproveitada se commit, split,
seed, hiperparâmetros, instrumentação e código forem idênticos. Existe a
possibilidade de o professor considerar esta variável um desdobramento da
primeira; sinalize essa ambiguidade antes de associá-la à bonificação.

### Variável 3 — largura da camada oculta

Fixar antecipadamente `C0` — mistura normalizada, `beta=(0,0,0)` — e alterar
somente `hidden`:

| ID | `hidden` | Parâmetros estimados |
|---|---:|---:|
| `H32` | 32 | 3.557 |
| `H64` | 64 | 7.109 |
| `H128` | 128 | 14.213 |

Para 108 entradas, duas saídas e três `beta`, use `P(h) = 111h + 5`; recalcule
se a arquitetura mudar. `H64` é a referência. Não escolha a ativação desta
etapa retrospectivamente a partir do melhor resultado anterior.

`H64` coincide com `C0` apenas quando todos os metadados e o código são
idênticos; nesse caso, a execução pode ser reutilizada sem duplicação.

Objetivo: quantificar capacidade versus custo e procurar retornos decrescentes.
Hipótese inicial: `H32` será mais econômico e poderá perder capacidade; `H128`
elevará parâmetros e FLOPs aproximadamente em proporção à largura, mas poderá
ter ganho marginal menor que `H32 -> H64` e maior diferença entre treino e
validação.

Implementação futura exigida:

- expor `hidden` como configuração da mesma MLP;
- preservar a mistura normalizada e todos os outros controles;
- testar dimensões, contagem de parâmetros e saída `(2, batch)` em cada largura;
- registrar loss/acurácia de treino e validação, parâmetros, FLOPs por época e
  total, além dos ganhos marginais entre larguras;
- analisar os pares brutos `(FLOPs, acurácia)` e as curvas por época.

Modelos com larguras diferentes não podem iniciar com pesos idênticos; a seed
garante repetibilidade, não equivalência tensor a tensor. Manter 100 épocas
iguala o número de passos, não o orçamento em FLOPs. Não altere épocas para
compensar isso dentro desta variável.

### Avaliação exploratória posterior

Somente depois de concluir e preservar as três análises isoladas, poderá ser
avaliada uma configuração que reúna níveis selecionados das variáveis. Essa
execução não substitui os estudos unifatoriais, não conta como nova variável e
não permite atribuir causalmente o resultado a um único fator. Se realizada,
selecione os níveis pela validação, rotule a configuração como exploratória e
consulte o teste apenas na avaliação final. Por decisão do estudante, esta etapa
não deve constar no `Passo-a-passo.md`.

## Fontes de verdade

Use, nesta ordem:

1. orientação mais recente do estudante ou professor;
2. `Passo-a-passo.md`, para decisões e trajetória;
3. `Estudo_Dirigido_RP___2026_1.pdf`, para requisitos acadêmicos;
4. código e testes, para comportamento técnico;
5. este arquivo, para o protocolo acordado;
6. `README.md`, que será atualizado como entregável.

Exponha conflitos e diferencie requisito do professor, decisão do estudante e
recomendação do agente.

## Memória contínua

Mantenha um `PROJECT_STATUS.md` curto quando o trabalho experimental começar.
Ao final de cada interação que alterar materialmente o projeto, registre:

- objetivo e decisões atuais;
- trabalho concluído e arquivos afetados;
- comandos e evidências de validação;
- riscos, pendências e próximo passo;
- progresso dos entregáveis.

Não marque como concluído algo apenas proposto. Se informar percentual, mostre o
critério, como `itens concluídos / total de itens`.

## Evidências obrigatórias para o vídeo — seção 4 do enunciado

Os 12 pontos abaixo são requisitos explícitos da apresentação, não sugestões.
Quando o trabalho experimental começar, mantenha
`experiments/video_evidence.md` como um mapa vivo entre cada requisito e suas
evidências no repositório. Atualize-o ao concluir cada mudança ou execução
relevante; não deixe a reconstrução dessas informações apenas para o final.

O mapa de evidências deve preservar todos os itens:

1. **Uso de IA:** o que foi delegado à IA e como código, explicações e sugestões
   foram verificados, corrigidos e validados pelo estudante.
2. **Tarefa de aprendizado:** Adult, conjunto de dados, entradas, saídas,
   objetivo do treinamento e métrica usada para avaliar o desempenho.
3. **Baseline:** configuração de referência empregada e justificativa de sua
   escolha.
4. **Variáveis e controles:** variável modificada em cada experimento e
   elementos mantidos constantes para garantir comparação controlada.
5. **Formulação investigada:** descrição matemática ou computacional das
   modificações e explicação de como elas alteram o processamento da rede.
6. **Arquitetura e treinamento:** funcionamento do modelo e do processo de
   treinamento, relacionando implementação e conceitos teóricos da disciplina.
7. **Protocolo experimental:** quantidade de configurações, repetições, sementes
   aleatórias e métricas de desempenho escolhidas.
8. **Resultados completos:** resultados de cada configuração por meio de
   tabelas, gráficos ou registros de treinamento.
9. **Desempenho versus FLOPs:** relação entre a métrica de desempenho e os
   FLOPs, melhor retorno computacional e possíveis retornos decrescentes.
10. **Hipóteses versus resultados:** quais hipóteses foram sustentadas,
    refutadas ou permaneceram inconclusivas.
11. **Dificuldades:** erros, instabilidades, resultados inesperados e alterações
    realizadas durante a implementação e o desenvolvimento.
12. **Reprodução:** dependências, comandos, configurações e organização dos
    arquivos necessários para reproduzir os experimentos.

Para cada item, registre afirmações que poderão ser apresentadas, caminhos dos
artefatos que as sustentam, IDs de execução ou commits pertinentes, limitações e
uma explicação em linguagem que o estudante consiga defender. Preserve também
erros e tentativas descartadas quando forem relevantes ao item 11. Diferencie
sempre plano, resultado medido e interpretação.

Antes da gravação, converta esse mapa em um roteiro de até 20 minutos que cubra
os 12 itens. Não basta mostrar o código ou as métricas finais: a apresentação
deve demonstrar compreensão da modificação, dos fundamentos teóricos, dos
efeitos sobre o modelo e dos limites das conclusões. O agente pode estruturar e
revisar o roteiro, mas não deve afirmar que o estudante compreendeu ou validou
uma explicação sem confirmação dele.

## Protocolo obrigatório

- Reproduza a baseline antes das variantes e registre commit, ambiente,
  dependências, comando, configuração, métricas e FLOPs.
- Antes de executar, registre cada hipótese: variável e níveis, controles,
  efeitos esperados em acurácia e FLOPs, justificativa e critério para
  sustentá-la, refutá-la ou considerá-la inconclusiva.
- Varie uma variável por vez e mantenha as três varreduras independentes.
- Mantenha dataset, pré-processamento, split, ordem, épocas, otimizador, learning
  rate e arquitetura constantes, salvo o fator declarado.
- Reinicialize cada execução independente com `cpu.set_seed(0)` e assegure o
  mesmo split e inicialização comparável.
- Registre acurácia de treino, validação e teste. Por padrão, selecione pela
  validação e reserve o teste para a avaliação final; qualquer alternativa deve
  ser definida antes dos resultados.
- Registre FLOPs por época e total, em valor bruto e GFLOP. O contador mede
  operações instrumentadas, não tempo, energia ou memória.
- Relate todas as configurações válidas, inclusive resultados negativos ou
  inesperados.
- Execute integralmente a análise obrigatória de trade-offs definida na seção
  seguinte.

## Análise obrigatória de trade-offs — Passo 4 do enunciado

Depois de concluir as investigações das três variáveis, reúna todas as
configurações válidas em uma tabela e em um gráfico de desempenho versus FLOPs.
Preserve os pares brutos `(FLOPs, desempenho)`, defina a métrica de desempenho e
a fórmula de “retorno por FLOP” antes de observar os resultados e não substitua
FLOPs por tempo, energia ou memória.

Responda explicitamente, com base nos resultados medidos, a todas as perguntas:

1. Qual configuração apresenta o melhor retorno por FLOP, isto é, o maior
   desempenho em relação ao custo computacional?
2. A partir de qual configuração começam a ocorrer retornos decrescentes, de
   modo que o aumento do custo deixa de produzir ganhos relevantes?
3. Qual variável provoca grande variação em FLOPs, mas pouca alteração no
   desempenho? Existe alguma variável com comportamento contrário?
4. Com um orçamento computacional fixo em FLOPs, qual configuração escolher?
   Justifique a escolha com base nos resultados.

Além das respostas, identifique configurações dominadas e a fronteira de Pareto
quando os dados permitirem. Declare o orçamento fixo e o critério operacional de
“ganho relevante” antes de aplicá-los. Se as medições não sustentarem uma
resposta, registre-a como inconclusiva em vez de completar a lacuna por hipótese.
Esta seção é um requisito futuro: nenhuma resposta ou conclusão experimental foi
produzida ainda.

## Fluxo de trabalho

1. definir as hipóteses formais, métrica primária e regra de repetição;
2. reproduzir e registrar `A0`, a baseline;
3. implementar, validar, executar e analisar `A0`–`A4`;
4. implementar, validar, executar e analisar `C0`–`C1`;
5. implementar, validar, executar e analisar `H32`–`H128`;
6. executar a análise obrigatória de trade-offs e responder às quatro perguntas;
7. gerar a análise geral e atualizar reprodução, IA e apresentação.

Não implemente todas as variáveis simultaneamente. Feche os artefatos e a
interpretação de cada etapa antes de iniciar a seguinte.

## Artefatos experimentais

Quando necessários, use uma estrutura simples e versionável:

```text
experiments/
  configs/
  logs/
  results.csv
  hypotheses.md
  analysis.md
  ai_usage.md
  video_evidence.md
  plots/
```

Cada execução deve registrar tarefa, variável/nível, seed, repetição,
hiperparâmetros, commit, comando, status, acurácias, FLOPs e observações.
Tabelas e gráficos devem ser regeneráveis a partir dos dados brutos.

## Validação

Antes de considerar uma mudança pronta:

- explique a equação e sua tradução para `forward` e gradientes;
- execute o gradient check da q04;
- teste a integração com a MLP;
- faça um smoke test antes do treinamento completo;
- confirme que a baseline continua reproduzível;
- atualize dependências e comandos realmente usados.

Comandos relevantes:

```bash
pytest -q --ignore=test/test_model.py
python -m exercises.q04_learnable_activations
python -m exercises.task_binary_classification
```

Estado observado em 20/07/2026: 62 testes passam; quatro placeholders do
Transformer falham deliberadamente com `NotImplementedError`. Não afirme que a
suíte completa passa nem amplie o escopo para corrigi-los.

## Convenções e limites

- Preserve NumPy/CPU e justifique qualquer nova dependência em
  `requirements.txt`.
- A classificação usa tensores `(features, amostras)`.
- A configuração atual usa 108 entradas, `hidden=64`, `epochs=100`, `lr=1e-2`,
  Adam e validação de 20%.
- A baseline tem 7.106 parâmetros; a q04 adiciona três coeficientes, totalizando
  7.109, sem contar configurações que alterem outras dimensões.
- Não altere datasets brutos nem duplique a tarefa para cada configuração.
- Faça mudanças pequenas, configuráveis e compatíveis com alterações do
  estudante.
- Não faça commit, push ou publicação sem solicitação explícita.

## IA e autoria

Registre contribuições materiais da IA em `experiments/ai_usage.md`: data,
objetivo, sugestão ou código, arquivos afetados, verificação realizada,
correções e decisão final do estudante.

Explique em português claro e diferencie observação, inferência e recomendação.
Nunca invente execução, métrica, FLOPs, citação ou validação humana. Só descreva
uma saída de IA como compreendida e validada após verificação real.

## Critério de conclusão

- código e gradient checks validados;
- baseline e variantes reproduzíveis;
- hipóteses e configurações preservadas;
- acurácia e FLOPs completos em logs, tabela e gráfico;
- trade-off, retornos decrescentes e limitações analisados;
- `README.md` e `requirements.txt` fiéis ao experimento;
- dificuldades e uso de IA documentados;
- os 12 requisitos do vídeo ligados a evidências verificáveis no repositório;
- repositório GitHub e vídeo de até 20 minutos prontos para entrega.

Permanecem ambíguos no enunciado o número de repetições, a métrica primária entre
as acurácias, a fórmula de retorno por FLOP e o critério exato de profundidade de
uma variável. Defina essas escolhas antes dos experimentos e recomende consulta
ao professor se houver impacto relevante na avaliação.
