# Hipóteses experimentais

## Variável 1 — família da ativação

Registradas em 21/07/2026, antes de implementação e resultados.

## Controles

Mesma MLP `108 -> 64 -> 2`, split, pesos iniciais por seed, Adam, `lr=1e-2`,
100 épocas e avaliação. Muda apenas a ativação oculta.

## Hipóteses

### H1a — melhor função suave

A melhor entre Sigmoid, Swish e Softplus superará `F-RELU` em pelo menos 0,5
ponto percentual na média de validação, com o mesmo sinal em duas das três
seeds.

Escolher primeiro a função suave pela maior média e depois compará-la com ReLU.

### H1b — Sigmoid

`F-SIGMOID` ficará pelo menos 0,5 ponto percentual abaixo de `F-RELU`, com o
mesmo sinal em duas das três seeds, por possível saturação e saída não centrada.

### H1c — custo

`F-RELU` terá menos FLOPs instrumentados que cada função suave, pois seu custo
por elemento é menor.

## Regras de conclusão

- H1a/H1b: usar a margem de 0,5 ponto e a concordância entre seeds; efeito
  inverso refuta e os demais casos são inconclusivos.
- H1c: sustentada se `F-RELU` tiver FLOPs estritamente menores que cada função
  suave na mesma janela; refutada caso contrário.

O teste oficial não foi consultado nesta etapa. Este arquivo preserva o registro
pré-experimental; os resultados medidos estão em `experiments/analysis.md`.

## Variável 2 — curvatura da Softplus

Registrada em 23/07/2026, antes dos smokes e das runs científicas da V2.

### Controles

Mesma MLP `108 -> 64 -> 2`, split, pesos iniciais por seed, Adam, `lr=1e-2`,
100 épocas e avaliação. Muda somente o `beta` fixo de
`logaddexp(0, beta*z) / beta`, com níveis `0,5`, `1`, `2` e `5`.

`S-BETA-1` será executada novamente: ela coincide em valor com a Softplus da V1,
mas usa o caminho Softplus-beta de 5 FLOPs por elemento, enquanto a V1 usou
3 FLOPs por elemento.

### H2 — valores centrais versus extremos

Escolher pela maior média de validação na época 100:

- o vencedor central entre `S-BETA-1` e `S-BETA-2`;
- o vencedor extremo entre `S-BETA-0.5` e `S-BETA-5`.

H2 será sustentada se o vencedor central superar o extremo por pelo menos
0,5 p.p. na média e tiver diferença positiva nas seeds pareadas em pelo menos
duas de três. O efeito inverso, com a mesma margem e concordância, refuta H2.
Os demais casos são inconclusivos.

Em empate exato dentro de um grupo, tratar os níveis como co-vencedores. Só
classificar H2 se todas as comparações entre co-vencedores produzirem a mesma
classificação; caso contrário, considerá-la inconclusiva.

### Custo e verificações mecânicas

Todos os níveis devem ter exatamente os mesmos FLOPs instrumentados, pois usam
a mesma forma algébrica. Também verificar, sem tratar como hipótese, que
`beta=1` coincide numericamente com a Softplus padrão e que aumentar `beta`
aproxima a função da ReLU na grade definida pelos testes.

O teste oficial não será consultado durante implementação, seleção ou análise
da V2.
