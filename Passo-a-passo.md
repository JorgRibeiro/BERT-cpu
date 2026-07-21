# Passo a passo

### 1 - Criar um AGENTS.md com o Codex - 5.6 Sol (Ultra)
- A propósito por tras desse agente é me ajudar a chegar no caminho, até a entrega do projeto.
- O mesmo, acada prompt ira salvar em um arquivo de memória oque estamos fazendo, qual o objetivo e nossa porcetagem de progresso.
- O intuito dessa organizacão é documentar toda a jornada, com  intuito de na apresentacao, ter nocao de como foi feito, por que e qual foi o proximo passo.
- O AGENTS.md será modificado constantemente.

### 2 - Definir a tarefa e a questão de investigação
- A tarefa escolhida foi a **classificação binária no conjunto Adult**, executada por `python -m exercises.task_binary_classification`.
- Escolhi essa tarefa porque ela me pareceu mais tangível e testável, com resultados comparáveis por acurácia e FLOPs.
- A questão escolhida foi a **q04 — ativações aprendíveis**, que combina ReLU, GELU e SiLU por meio de coeficientes aprendidos durante o treinamento.
- Escolhi a q04 porque ela se encaixa melhor no conhecimento que construí até agora sobre funções de ativação e permite aprofundá-lo com parâmetros aprendíveis.
- O artigo de referência investiga a proposta principalmente no contexto de LLMs. Neste projeto, avaliaremos a ideia em um contexto diferente: classificação binária tabular no Adult.

### 3 - Investigar a formulação da ativação oculta
- **Ideia inicial:** comparar cinco configurações para a ativação da camada oculta: ReLU fixa, GELU fixa, SiLU fixa, mistura fixa uniforme `(ReLU + GELU + SiLU) / 3` e mistura normalizada aprendível iniciada com pesos iguais a `1/3`.
- **Objetivo:** descobrir se combinar ativações melhora a classificação e separar o efeito de apenas misturá-las do efeito de aprender os coeficientes. A mistura fixa será o controle direto da mistura aprendível, pois ambas calcularão as mesmas três funções.
- **Controles iniciais:** manter `hidden=64`, 100 épocas, Adam, `lr=1e-2`, split, seed e demais configurações constantes.
- **Efeito final esperado:** a mistura normalizada aprendível poderá alcançar acurácia de validação igual ou superior à melhor ativação fixa e à mistura uniforme. Espera-se mais FLOPs que uma ativação única, mas custo próximo ao da mistura fixa.
- **Observação:** esta é somente a ideia e a hipótese iniciais. Nenhuma configuração, integração ou execução experimental deste passo foi implementada.

### 4 - Investigar a parametrização e a restrição dos coeficientes aprendíveis
- **Ideia inicial:** comparar a mistura normalizada por softmax, iniciada com `beta=(0,0,0)` e pesos `pi=(1/3,1/3,1/3)`, com a mistura livre iniciada em `alpha=(1/3,1/3,1/3)`.
- **Objetivo:** comparar o efeito conjunto da parametrização e da restrição dos coeficientes, garantindo que as duas versões comecem representando a mesma função e com a mesma escala.
- **Controles iniciais:** usar ReLU, GELU e SiLU nas duas versões e manter largura 64, dados, split, seed, treinamento e avaliação constantes.
- **Efeito final esperado:** a versão normalizada poderá produzir escala e coeficientes mais estáveis e interpretáveis; a versão livre terá maior flexibilidade e poderá reduzir mais a loss de treino, mas com maior risco de coeficientes negativos, aumento de escala ou pior generalização. Os FLOPs deverão ser próximos, com pequena diferença causada pelo softmax.
- **Observação:** esta é somente a ideia e a hipótese iniciais. Nenhuma alteração de inicialização, camada ou treinamento deste passo foi implementada.

### 5 - Investigar a largura da camada oculta
- **Ideia inicial:** avaliar a q04 normalizada com larguras `hidden=32`, `hidden=64` e `hidden=128`, usando 64 como referência.
- **Objetivo:** analisar como a capacidade da MLP afeta acurácia, parâmetros e FLOPs, procurando o ponto em que aumentar a largura deixa de produzir ganho relevante.
- **Controles iniciais:** manter a mistura normalizada, as três ativações, 100 épocas, Adam, `lr=1e-2`, dados, split, seed e avaliação constantes.
- **Efeito final esperado:** aumentar a largura deverá elevar parâmetros e FLOPs aproximadamente de forma linear. A acurácia poderá melhorar de 32 para 64, enquanto o ganho de 64 para 128 poderá ser menor, indicando retornos decrescentes ou maior overfitting.
- **Observação:** esta é somente a ideia e a hipótese iniciais. Nenhuma largura alternativa ou execução experimental deste passo foi implementada.

### 6 - Realizar o Passo 4 do enunciado: análise de trade-offs
- **Ideia inicial:** depois de concluir as três investigações, reunir os pares brutos de desempenho e FLOPs de todas as configurações válidas em uma tabela e em um gráfico de desempenho versus FLOPs. A comparação deverá considerar eficiência computacional, dominância de Pareto e ganhos marginais, sem usar tempo de execução como substituto de FLOPs.
- **Objetivo:** avaliar conjuntamente qualidade preditiva e custo computacional para transformar os resultados experimentais em uma decisão justificável, em vez de escolher uma configuração apenas pela maior acurácia.
- **Perguntas obrigatórias do enunciado:**

  1. Qual configuração apresenta o melhor retorno por FLOP, isto é, o maior desempenho em relação ao custo computacional?
  2. A partir de qual configuração começam a ocorrer retornos decrescentes, de modo que o aumento do custo deixa de produzir ganhos relevantes?
  3. Qual variável provoca grande variação em FLOPs, mas pouca alteração no desempenho? Existe alguma variável com comportamento contrário?
  4. Com um orçamento computacional fixo em FLOPs, qual configuração escolher? Justifique a escolha com base nos resultados.

- **Efeito final esperado:** obter uma análise apoiada nos valores medidos que identifique configurações eficientes, dominadas ou pertencentes à fronteira de Pareto, além de explicitar os critérios usados para retorno por FLOP, retornos decrescentes e orçamento fixo.
- **Observação:** este passo registra somente o plano e as perguntas que deverão ser respondidas. A fórmula de retorno por FLOP, o orçamento de referência, os gráficos, as respostas e as conclusões ainda não foram definidos ou produzidos.

## Registro das conversas

### Conversa de planejamento — encerrada em 21/07/2026
- **Encerramento:** esta conversa foi encerrada após a definição do escopo e do plano pré-experimental. As etapas 3 a 6 permanecem planejadas, mas ainda não foram implementadas ou executadas.
- **Referência da conversa:** conversa de planejamento encerrada em 21/07/2026 e identificada pelo resumo abaixo. O conteúdo relevante foi preservado nos arquivos versionados, portanto não é necessário manter um link externo para o chat.
- **Resumo extremo:** escolhemos Adult + q04, definimos três variáveis isoladas, estabelecemos o protocolo reprodutível e registramos a futura análise de trade-offs.
- **Continuidade:** uma nova conversa iniciará exclusivamente a preparação e a execução da Variável 1 — formulação da ativação oculta.
