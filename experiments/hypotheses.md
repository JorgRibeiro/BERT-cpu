# Hipóteses — Variável 1

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
