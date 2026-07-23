# Protocolo — Variável 2

## Objetivo

Medir o efeito da curvatura da Softplus na classificação Adult, alterando
somente o `beta` fixo:

```text
softplus_beta(z) = logaddexp(0, beta*z) / beta
```

Configurações: `S-BETA-0.5`, `S-BETA-1`, `S-BETA-2` e `S-BETA-5`.

## Execução

- três seeds por configuração: `0`, `1` e `2`;
- 100 épocas, full-batch Adam, `lr=1e-2`;
- mesmo split, pesos iniciais, arquitetura e pré-processamento da V1;
- executar `S-BETA-1` nas três seeds antes das demais configurações;
- não reutilizar `F-SOFTPLUS` e não consultar o teste oficial.

Cada run deve registrar as 100 épocas, checkpoint final, hashes, losses,
acurácias e FLOPs. Diagnósticos de `z`, `h` e da derivada local serão medidos
fora da janela de FLOPs nas épocas `0`, `1`, `25`, `50`, `75` e `100`.

## Análise

A métrica principal é a média da acurácia de validação na época 100. A melhor
época é apenas secundária. Aplicar H2 exatamente como congelada em
`experiments/hypotheses.md`, incluindo a regra para empates.

Todos os níveis devem custar `850.711.121` FLOPs instrumentados por época,
`85.071.112.100` por run e `94.632.384` na inferência completa da validação.
Os custos por época e inferência foram confirmados nos quatro smokes de duas
épocas. Smoke é validação técnica, não resultado de H2.

Os artefatos ficam isolados em `experiments/v2/`; a V1 permanece imutável.

Comandos:

```bash
python -m experiments.run_v2 --config-id S-BETA-1 --seed 0 --smoke
python -m experiments.run_v2_all --dry-run
```
