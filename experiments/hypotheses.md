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

## Variável 3 — profundidade linear sem ativação

Registrada em 24/07/2026, antes dos smokes e das runs científicas da V3.

### Controles

Adult, split, seeds, Adam, `lr=1e-2`, 100 épocas e avaliação permanecem fixos.
Muda a quantidade de camadas lineares, sem ativação entre elas:
`L1-DIRECT`, `L2-IDENTITY` e `L3-IDENTITY`.

As arquiteturas têm parametrizações e custos diferentes, mas todas representam
uma única função afim `W*x + b`. A seed torna cada modelo repetível; ela não
iguala pesos de formatos diferentes.

### H3a — ganho com a profundidade linear

Nem `L2-IDENTITY` nem `L3-IDENTITY` melhorará `L1-DIRECT` por pelo menos
`0,5` p.p. Para cada comparação, registrar a diferença pareada nas três seeds.
Um ganho do modelo mais profundo de pelo menos `0,5` p.p., positivo em duas
seeds, contradiz a previsão correspondente. Ausência desse ganho não prova
equivalência estatística; será descrita como “não contradita”.

H3a global será “contradita” se qualquer uma das duas comparações contradizer
a previsão; caso contrário, será “não contradita”.

### H3b — custo

Parâmetros e FLOPs instrumentados crescerão estritamente:

```text
L1-DIRECT < L2-IDENTITY < L3-IDENTITY
```

### H3c — retorno por FLOP

O retorno seguirá `R(L1-DIRECT) > R(L2-IDENTITY) > R(L3-IDENTITY)`, usando a
fórmula já congelada no protocolo geral.

### H3d — efeito da não linearidade

`F-RELU` superará `L2-IDENTITY` por pelo menos `0,5` p.p. na média e terá
diferença positiva nas seeds pareadas em pelo menos duas de três. O efeito
inverso refuta H3d; demais casos são inconclusivos.

As runs ReLU vêm da V1. Antes da comparação, conferir dados, split,
arquitetura linear, hiperparâmetros, procedimento, instrumentação, software e
hashes iniciais. Commit e plataforma/kernel diferentes serão registrados como
limitação: a ponte é controlada, mas não é um contexto literal idêntico. Não
serão criadas três runs ReLU extras fora do plano confirmado.

### Verificação mecânica

Para cada profundidade, comparar a saída do modelo com a matriz e o viés
colapsados, usando `rtol=1e-12` e `atol=1e-12`. Isso verifica a equivalência
afim, mas não é resultado incerto nem hipótese de desempenho.

O teste oficial continuará indisponível. O risco de enquadramento acadêmico de
V3 permanece registrado por decisão do estudante.
