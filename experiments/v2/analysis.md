# Análise — Variável 2

## Resultado principal

A métrica principal é a acurácia de validação na época 100, usando as três
seeds. O teste oficial não foi avaliado.

| beta | Loss treino | Loss validação | Acurácia treino | Acurácia validação | DP |
|---:|---:|---:|---:|---:|---:|
| 0,5 | 0,312865 | 0,326604 | 85,5196% | 84,7461% | 0,0177 p.p. |
| 1 | 0,308014 | 0,322490 | 85,7115% | 84,8843% | 0,1022 p.p. |
| 2 | 0,299514 | 0,316856 | 86,1927% | 85,0379% | 0,1546 p.p. |
| 5 | 0,289481 | 0,315672 | 86,7544% | 85,1966% | 0,1199 p.p. |

As médias cresceram com `beta`, mas o ganho entre os extremos foi de apenas
0,4505 p.p., abaixo da margem relevante de 0,5 p.p.

## H2 — valores centrais versus extremos

- vencedor central: `S-BETA-2`, com 85,0379%;
- vencedor extremo: `S-BETA-5`, com 85,1966%;
- diferença central menos extremo: `-0,1587` p.p.;
- diferenças pareadas nas seeds: `-0,2611`, `-0,2764` e `+0,0614` p.p.;
- sinais: uma seed positiva e duas negativas.

**H2 é inconclusiva.** O efeito observado ficou no sentido inverso ao previsto,
mas não alcançou 0,5 p.p.; portanto também não satisfaz a regra de refutação.

## Convergência

Na época 25, a validação média já seguia a ordem `0,5 < 1 < 2 < 5`. As losses
de treino e validação diminuíram quando `beta` aumentou. A acurácia de treino
cresceu mais do que a de validação, ampliando o intervalo treino-validação de
aproximadamente 0,77 p.p. em `beta=0,5` para 1,56 p.p. em `beta=5`.

`beta=0,5` começou mais lentamente. Em `beta=5`, a curva média atingiu 85,2733%
na época 78 e terminou apenas 0,0768 p.p. abaixo, indicando um leve platô
tardio, não uma falha da run.

As melhores épocas por seed, usadas somente como análise secundária, foram:

- `beta=0,5`: épocas 60, 87 e 75; média dos melhores valores 84,8434%;
- `beta=1`: épocas 91, 68 e 76; média 84,9253%;
- `beta=2`: épocas 71, 97 e 87; média 85,0788%;
- `beta=5`: épocas 78, 96 e 92; média 85,3501%.

O checkpoint principal permanece sendo a época 100, conforme definido antes
das execuções.

## Diagnósticos da ativação

Os valores iniciais de `z` foram idênticos entre os níveis, confirmando os
mesmos pesos iniciais por seed. Como `softplus_beta(0)=log(2)/beta`, a média
inicial de `h` caiu de 1,3888 em `beta=0,5` para 0,2036 em `beta=5`.

Na época 100, aumentar `beta` tornou a transição mais marcada. Os valores abaixo
são médias das estatísticas das três seeds, não um novo pooling das ativações:

| beta | Média de `h` | Derivada baixa | Derivada alta | `h` próximo de zero |
|---:|---:|---:|---:|---:|
| 0,5 | 1,3264 | 0,0869% | 0,0537% | 0,0000% |
| 1 | 0,6913 | 0,2092% | 0,1876% | 0,0000% |
| 2 | 0,3957 | 2,2315% | 1,8640% | 0,0501% |
| 5 | 0,2631 | 24,0437% | 15,3481% | 0,3038% |

Isso confirma mecanicamente a aproximação à ReLU. A associação com a melhora
observada não prova causalidade. A Softplus permaneceu estritamente positiva:
“próximo de zero” não significa zero exato. Além disso, somente a época 0 isola
a formulação; após o treino, os diagnósticos também refletem trajetórias de
pesos diferentes.

## Verificação de beta igual a 1

`S-BETA-1` reproduziu exatamente, nas três seeds, todos os valores de loss e
acurácia das 100 épocas e os pesos finais de `F-SOFTPLUS`. O custo mudou de
84,6543313 para 85,0711121 GFLOPs por run porque a V2 contabiliza também a
multiplicação e a divisão por `beta`.

## FLOPs e retorno

Todos os níveis usaram 85,0711121 GFLOPs instrumentados por run e 94.632.384
FLOPs na inferência completa da validação. As 12 runs somaram 1.020,8533452
GFLOPs instrumentados.

Como os custos são iguais, não há retorno marginal por FLOP dentro da V2.
`beta=5` teve o maior retorno observado, 0,1137219 ponto percentual acima da
classe majoritária por GFLOP. Pelas médias observadas, ele domina os outros
níveis de V2; isso não constitui teste de significância.

## Limitações

- três seeds e um split fixo medem variação de inicialização, não de amostragem;
- o desvio-padrão não é intervalo de confiança;
- o encoder foi ajustado antes do hold-out, sem usar rótulos da validação;
- FLOPs são instrumentados e não representam tempo, energia ou custo completo;
- a época 100 foi mantida mesmo quando uma época anterior teve valor melhor;
- nenhuma configuração foi escolhida com o teste oficial.

## Reprodução

```bash
python -m experiments.run_v2_all --dry-run
python -m experiments.plot_v2
pytest -q --ignore=test/test_model.py
```

Dados brutos: `experiments/v2/results.csv`, `logs/` e `checkpoints/`. Tabela:
`experiments/v2/summary.csv`. Gráficos: `experiments/v2/plots/`.
