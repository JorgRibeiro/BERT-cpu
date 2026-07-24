# Evidências para o vídeo

## Highlights — preparação técnica da V1

Antes da investigação de 100 épocas, foi necessário:

- implementar Sigmoid, Swish e Softplus estáveis, incluindo seus backwards;
- usar uma única MLP configurável, mudando somente a ativação oculta;
- garantir o mesmo split e os mesmos pesos iniciais entre ativações;
- estabilizar a cross-entropy para impedir NaN/Inf em logits extremos;
- instrumentar custos diferentes de FLOPs para cada ativação;
- impedir acesso automático ao teste oficial;
- validar tudo com 123 testes, gráficos, smoke e repetição exata da ReLU;
- preparar logs, checkpoints e hashes antes da primeira run científica.

Mensagem para a apresentação: esta etapa não escolheu a melhor ativação. Ela
removeu fontes de erro e garantiu uma comparação controlada para a V1.

## Highlights — resultado da V1

- Swish teve a maior validação média: 85,1351%.
- ReLU teve o melhor retorno por FLOP e o menor custo.
- H1a e H1b foram inconclusivas; H1c foi sustentada.
- ReLU e Swish ficaram na Pareto; Sigmoid e Softplus foram dominadas.
- As curvas, seeds, dispersão e Pareto são regeneráveis por
  `python -m experiments.plot_v1`.

## Highlights — preparação da V2

- `beta` é uma constante fixa, não um parâmetro aprendido.
- O forward usa `logaddexp(0, beta*z) / beta` e o backward usa
  `sigmoid(beta*z)`.
- A Softplus antiga permanece em 3 FLOPs por elemento; Softplus-beta usa 5 em
  todos os níveis.
- `S-BETA-1` foi executada novamente: igualdade matemática não torna commits,
  caminhos e instrumentações experimentais idênticos.
- Quatro smokes validaram o executor e os custos; eles não testam H2.
- As três seeds de `S-BETA-1` são resultado parcial; H2 exige os quatro betas.

## Resultado da V2

- Validação média observada: beta 0,5 = 84,7461%; beta 1 = 84,8843%;
  beta 2 = 85,0379%; beta 5 = 85,1966%.
- As 12 runs custaram 85,0711121 GFLOPs instrumentados cada.
- H2 ficou inconclusiva: beta 2 menos beta 5 = -0,1587 p.p., abaixo do limiar.
- Beta 1 reproduziu exatamente a Softplus da V1 em métricas e pesos, mas teve
  custo instrumentado maior.

## Highlights — preparação da V3

- As três redes não usam ativação: as camadas `Linear` são ligadas diretamente.
- Não foi criada uma camada Identity; portanto seu custo é zero.
- Mesmo com mais camadas, a composição continua sendo uma função afim `W*x+b`.
- O que muda é a parametrização, a otimização, os parâmetros e os FLOPs.
- Três smokes confirmaram 218, 7.106 e 11.266 parâmetros e custos crescentes.
- `L2-IDENTITY` começa com os mesmos pesos lineares da ReLU em cada seed.
- A comparação com ReLU verifica os controles, mas atravessa commits e kernel;
  essa limitação deve ser mostrada.
- 269 testes passaram antes das runs; o teste oficial permaneceu reservado.

## Resultado da V3

- Validação média: L1 = 84,5414%; L2 = 84,6744%; L3 = 84,6796%.
- Profundidade ganhou só `0,1331` a `0,1382` p.p. sobre L1, abaixo do limiar.
- O custo cresceu de 2,6107501 para 84,0291601 e 154,4654481 GFLOPs/run.
- H3a não foi contradita; H3b e H3c foram sustentadas.
- ReLU venceu L2 nas três seeds por `0,3737` p.p. em média, mas H3d ficou
  inconclusiva porque não atingiu `0,5` p.p.
- Mensagem principal: sem ativação, mais camadas mantiveram a mesma classe afim
  e trouxeram ganho pequeno com forte aumento de custo.
- Evidência: `experiments/v3/analysis.md`, `summary.csv` e três gráficos.

| Item | Estado | Evidência atual |
|---:|---|---|
| 1. Uso de IA | Parcial | `experiments/ai_usage.md` |
| 2. Tarefa Adult | Parcial | `AGENTS.md` e configuração de V1 |
| 3. Baseline | Concluído | ReLU: três seeds e repetição exata da seed 0; teste reservado |
| 4. Variáveis e controles | Concluído | `AGENTS.md`, `hypotheses.md` e protocolos V1/V2/V3 |
| 5. Formulação | Concluído | ativações q01 e colapso afim da V3 |
| 6. Arquitetura e treino | Concluído | `AdultMLP`, `AdultLinearClassifier` e testes |
| 7. Protocolo | Concluído | executores e configurações de V1, V2 e V3 |
| 8. Resultados | Parcial | V1, V2 e V3 completas em validação; teste final pendente |
| 9. Desempenho/FLOPs | Parcial | resultados por variável prontos; síntese conjunta pendente |
| 10. Hipóteses/resultados | Concluído | análises de V1, V2 e V3 |
| 11. Dificuldades | Parcial | estabilidade da loss e correção vetor `@` vetor em `experiments/ai_usage.md` |
| 12. Reprodução | Parcial | executores, logs, tabelas e gráficos das três variáveis |
