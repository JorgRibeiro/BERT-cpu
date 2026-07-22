# Análise experimental

As quatro configurações passaram no smoke de duas épocas com seed 0, valores
finitos e sem consulta ao teste. O smoke não entra em `results.csv` e não testa
as hipóteses; os resultados científicos começam nas seções abaixo.

O executor persistente também passou em smoke isolado: salvou log e checkpoint,
não carregou o teste e manteve `results.csv` sem runs científicas.

## Baseline ReLU — primeira run (`F-RELU-s0-r1`)

Na época 100: treino 87,2202%, validação 84,8741% e 84,2375505 GFLOPs
instrumentados. A melhor validação secundária foi 85,3808% na época 64. A
fração de saídas ReLU exatamente zeradas foi de 52,08% antes do treino para
58,21% na época 100. Isso descreve uma seed; não permite concluir hipóteses nem
comparar ativações. O teste oficial não foi consultado.

## Reprodução determinística — `F-RELU-s0-r2`

A repetição reproduziu exatamente os 100 registros de época, os seis
diagnósticos, os pesos finais e os 84,2375505 GFLOPs. Isso valida o determinismo
da seed 0 no ambiente atual; não substitui as seeds 1 e 2.

## Baseline ReLU — três seeds

| Seed | Loss treino | Loss validação | Treino (%) | Validação (%) | Melhor validação secundária |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,279128 | 0,321657 | 87,2202 | 84,8741 | 85,3808% na época 64 |
| 1 | 0,278274 | 0,318769 | 87,4007 | 85,1044 | 85,2733% na época 70 |
| 2 | 0,282462 | 0,316709 | 87,1818 | 85,1658 | 85,3194% na época 32 |

A loss média final foi 0,279955 no treino e 0,319045 na validação. A média
primária de acurácia foi 85,0481%, com desvio-padrão amostral de 0,1538 p.p.
Cada run custou 84,2375505 GFLOPs. A baseline superou a classe majoritária em
média por 9,5260 p.p., equivalente provisoriamente a 0,1131 p.p./GFLOP. Todas
as seeds aumentaram a fração de saídas ReLU zeradas durante o treino. Esses
dados estabelecem a referência; ainda não comparam ativações.

## Regra de análise aplicada às ativações

Cada ativação será executada como um bloco com seeds 0, 1 e 2. A análise sempre
mostrará, por seed e em resumo, loss de treino/validação, acurácia de
treino/validação, FLOPs e melhor validação secundária. Não será feita conclusão
com base em uma única seed.

## Sigmoid — três seeds

| Seed | Loss treino | Loss validação | Treino (%) | Validação (%) | Melhor validação secundária |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,314341 | 0,326947 | 85,3737 | 84,7359 | 84,7359% na época 100 |
| 1 | 0,312031 | 0,324892 | 85,5426 | 84,6744 | 84,8280% na época 88 |
| 2 | 0,311857 | 0,324752 | 85,5388 | 84,8280 | 84,8741% na época 97 |

A loss média final foi 0,312743 no treino e 0,325530 na validação. A acurácia
média de validação foi 84,7461%, com desvio-padrão amostral de 0,0773 p.p. Cada
run custou 84,8627217 GFLOPs. A saturação média passou de 0,0051% antes do
treino para 0,5444% na época 100, permanecendo baixa segundo o limiar definido.

### Comparação Sigmoid versus ReLU

| Métrica na época 100 | ReLU | Sigmoid | Diferença Sigmoid − ReLU |
|---|---:|---:|---:|
| Loss média de treino | 0,279955 | 0,312743 | +0,032788 |
| Loss média de validação | 0,319045 | 0,325530 | +0,006485 |
| Acurácia média de treino | 87,2676% | 85,4850% | −1,7825 p.p. |
| Acurácia média de validação | 85,0481% | 84,7461% | −0,3020 p.p. |
| GFLOPs por run | 84,2375505 | 84,8627217 | +0,6251712 |

A Sigmoid ficou abaixo da ReLU nas três seeds, mas não alcançou a diferença
pré-definida de 0,5 p.p. Por isso, H1b é **inconclusiva**, e não sustentada. A
ReLU foi mais barata neste par, evidência parcial para H1c. O retorno provisório
da Sigmoid foi 0,1087 p.p./GFLOP, contra 0,1131 da ReLU. O teste oficial não foi
consultado.

## Swish — três seeds

| Seed | Loss treino | Loss validação | Treino (%) | Validação (%) | Melhor validação secundária |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,293120 | 0,316576 | 86,5830 | 85,1198 | 85,1198% na época 99 |
| 1 | 0,295391 | 0,314311 | 86,4525 | 85,1658 | 85,1658% na época 97 |
| 2 | 0,295951 | 0,313728 | 86,4102 | 85,1198 | 85,1966% na época 72 |

A loss média final foi 0,294821 no treino e 0,314872 na validação. A acurácia
média de validação foi 85,1351%, com desvio-padrão amostral de 0,0266 p.p. Cada
run custou 85,0711121 GFLOPs. A fração média de derivadas locais com módulo até
0,05 passou de 0,4531% para 5,2662%; a derivada final mínima foi cerca de
−0,09984 e a saída mínima, −0,27846, comportamento permitido pela Swish.

### Comparação Swish versus ReLU

| Métrica na época 100 | ReLU | Swish | Diferença Swish − ReLU |
|---|---:|---:|---:|
| Loss média de treino | 0,279955 | 0,294821 | +0,014866 |
| Loss média de validação | 0,319045 | 0,314872 | −0,004173 |
| Acurácia média de treino | 87,2676% | 86,4819% | −0,7857 p.p. |
| Acurácia média de validação | 85,0481% | 85,1351% | +0,0870 p.p. |
| GFLOPs por run | 84,2375505 | 85,0711121 | +0,8335616 |

A Swish superou a ReLU nas seeds 0 e 1 e ficou 0,0461 p.p. abaixo na seed 2.
Ela terminou como a melhor função suave, mas o ganho médio de 0,0870 p.p. não
atingiu o limiar de 0,5 p.p. A ReLU foi mais barata também neste par. O retorno
da Swish foi 0,1130 p.p./GFLOP, praticamente igual aos 0,1131 da ReLU. O teste
oficial não foi consultado.

## Softplus — três seeds

| Seed | Loss treino | Loss validação | Treino (%) | Validação (%) | Melhor validação secundária |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,310301 | 0,324651 | 85,5810 | 84,8587 | 84,8741% na época 91 |
| 1 | 0,307902 | 0,322646 | 85,7077 | 84,7973 | 84,8587% na época 68 |
| 2 | 0,305840 | 0,320172 | 85,8459 | 84,9969 | 85,0430% na época 76 |

A loss média final foi 0,308014 no treino e 0,322490 na validação. A acurácia
média de validação foi 84,8843%, com desvio-padrão amostral de 0,1022 p.p. Cada
run custou 84,6543313 GFLOPs. A fração média de derivadas locais com módulo até
0,05 passou de 0,0039% para 0,2092%, e nenhuma saída ficou próxima de zero pelo
limiar definido.

### Comparação Softplus versus ReLU

| Métrica na época 100 | ReLU | Softplus | Diferença Softplus − ReLU |
|---|---:|---:|---:|
| Loss média de treino | 0,279955 | 0,308014 | +0,028059 |
| Loss média de validação | 0,319045 | 0,322490 | +0,003445 |
| Acurácia média de treino | 87,2676% | 85,7115% | −1,5560 p.p. |
| Acurácia média de validação | 85,0481% | 84,8843% | −0,1638 p.p. |
| GFLOPs por run | 84,2375505 | 84,6543313 | +0,4167808 |

A Softplus ficou abaixo da ReLU nas três seeds e também teve custo maior. Seu
retorno foi 0,1106 p.p./GFLOP, contra 0,1131 da ReLU.

## Síntese da Variável 1

| Configuração | Loss treino | Loss validação | Validação média | Desvio | GFLOPs/run | Retorno p.p./GFLOP |
|---|---:|---:|---:|---:|---:|---:|
| Swish | 0,294821 | 0,314872 | 85,1351% | 0,0266 p.p. | 85,0711121 | 0,1130 |
| ReLU | 0,279955 | 0,319045 | 85,0481% | 0,1538 p.p. | 84,2375505 | 0,1131 |
| Softplus | 0,308014 | 0,322490 | 84,8843% | 0,1022 p.p. | 84,6543313 | 0,1106 |
| Sigmoid | 0,312743 | 0,325530 | 84,7461% | 0,0773 p.p. | 84,8627217 | 0,1087 |

As 12 runs primárias consumiram 1.016,4771468 GFLOPs instrumentados. Incluindo
a repetição determinística da ReLU, a V1 executou 1.100,7146973 GFLOPs.

- **H1a — inconclusiva:** Swish foi a melhor suave e superou ReLU em duas
  seeds, mas o ganho médio de 0,0870 p.p. ficou abaixo de 0,5 p.p.
- **H1b — inconclusiva:** Sigmoid ficou abaixo nas três seeds, mas a diferença
  média de 0,3020 p.p. não atingiu 0,5 p.p.
- **H1c — sustentada:** ReLU teve FLOPs estritamente menores que todas as
  funções suaves na mesma janela.

Na análise pelas médias, ReLU e Swish formam a fronteira de Pareto. Sigmoid e
Softplus são dominadas pela ReLU, que apresentou maior acurácia e menor custo.
ReLU teve o melhor retorno por FLOP; Swish teve a maior acurácia, mas seu ganho
de 0,0870 p.p. custou 0,8335616 GFLOP adicional e não atingiu a margem de ganho
relevante. Essas conclusões usam apenas treino e validação; o teste oficial não
foi consultado.

## Reprodução dos gráficos

Execute:

```bash
python -m experiments.plot_v1
```

O comando valida os 13 artefatos, exclui a repetição ReLU das médias e gera:

- `experiments/v1_summary.csv`;
- `experiments/plots/v1_learning_curves.png`;
- `experiments/plots/v1_final_metrics_by_seed.png`;
- `experiments/plots/v1_accuracy_vs_flops.png`.

As curvas não são suavizadas. Os pontos mostram as seeds e as faixas/barras
mostram um desvio-padrão amostral. Com apenas três seeds, isso não é intervalo
de confiança nem teste de significância. O split é fixo, portanto a dispersão
mede somente variação da inicialização. A Pareto usa as médias observadas.
