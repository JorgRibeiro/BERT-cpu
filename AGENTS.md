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

- Variáveis investigadas e seus níveis.
- Hipóteses registradas antes das execuções.
- Configurações experimentais e baseline executada de referência.
- Métrica primária para selecionar configurações.
- Repetições e sementes adicionais; `cpu.set_seed(0)` é obrigatório.
- Definição operacional de retorno por FLOP.
- Orçamento computacional de referência.

Não feche uma dessas decisões sem confirmação do estudante quando ela mudar o
rumo da investigação.

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

## Protocolo obrigatório

- Reproduza a baseline antes das variantes e registre commit, ambiente,
  dependências, comando, configuração, métricas e FLOPs.
- Antes de executar, registre cada hipótese: variável e níveis, controles,
  efeitos esperados em acurácia e FLOPs, justificativa e critério para
  sustentá-la, refutá-la ou considerá-la inconclusiva.
- Varie uma variável por vez. Faça varreduras separadas para variáveis distintas;
  combinações posteriores são exploratórias.
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
- Produza tabela e gráfico de acurácia versus FLOPs. Analise dominância/Pareto,
  retorno computacional, retornos decrescentes e escolha sob orçamento fixo.
- Defina “retorno por FLOP” antes de observar os resultados e preserve sempre os
  pares brutos `(FLOPs, acurácia)`.

## Cuidados específicos da q04

- A mistura calcula todas as ativações; coeficientes pequenos não eliminam seus
  FLOPs.
- Três coeficientes acrescentam poucos parâmetros, mas as operações elementwise
  aumentam o custo.
- Um coeficiente maior não implica, sozinho, maior importância: considere também
  a escala de saída de cada ativação.
- A mistura livre inicializada com três coeficientes iguais a 1 pode produzir
  escala maior que a baseline. Controle ou documente esse efeito ao comparar com
  a versão normalizada.
- Não confunda ganho da mistura com ganho do aprendizado dos coeficientes; use
  controles adequados quando essa distinção fizer parte da hipótese.
- Não conclua que uma ativação é universalmente superior a partir de um único
  dataset e protocolo.

## Fluxo de trabalho

1. escolher variáveis, níveis, controles e hipóteses;
2. reproduzir e registrar a baseline;
3. implementar e validar a integração da q04;
4. executar as configurações controladas;
5. gerar tabela, gráfico e análise;
6. atualizar reprodução, registro de IA e apresentação.

Priorize uma variável bem investigada e a entrega completa. Só amplie o escopo
quando baseline, instrumentação e reprodução estiverem confiáveis.

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
- repositório GitHub e vídeo de até 20 minutos prontos para entrega.

Permanecem ambíguos no enunciado o número de repetições, a métrica primária entre
as acurácias, a fórmula de retorno por FLOP e o critério exato de profundidade de
uma variável. Defina essas escolhas antes dos experimentos e recomende consulta
ao professor se houver impacto relevante na avaliação.
