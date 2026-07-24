# Análise conjunta — V1, V2 e V3

A avaliação de origem é `OFFICIAL-185889b9b944304ba514`. A tabela usa 33 pares brutos:
11 configurações, seeds 0, 1 e 2.

## Regra de leitura

- Decisões, retorno, orçamento e Pareto usam somente a média de validação
  da época 100 e os FLOPs instrumentados de treinamento.
- Retorno: `(validação − classe majoritária) / GFLOPs de treinamento`;
  maioria da validação = 75,5221%.
- Ganho relevante: `0,5` p.p.; orçamento:
  `84,2375505` GFLOPs de `F-RELU`.
- O teste oficial aparece somente como descrição pós-congelamento; ele
  não muda hipóteses, seleção, Pareto ou novas execuções.

## Tabela conjunta

| Rank val | Configuração | Variável | Parâmetros | Validação ± DP | Teste ± DP* | GFLOPs | Retorno | Estado |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `S-BETA-5` | V2 | 7.106 | 85,1966% ± 0,1199 | 85,5414% ± 0,1383 | 85,0711121 | 0,113722 | pareto |
| 2 | `F-SWISH` | V1 | 7.106 | 85,1351% ± 0,0266 | 85,5967% ± 0,1611 | 85,0711121 | 0,113000 | dominated |
| 3 | `F-RELU` | V1 | 7.106 | 85,0481% ± 0,1538 | 85,4780% ± 0,1636 | 84,2375505 | 0,113085 | pareto |
| 4 | `S-BETA-2` | V2 | 7.106 | 85,0379% ± 0,1546 | 85,6458% ± 0,0798 | 85,0711121 | 0,111857 | dominated |
| 5 | `F-SOFTPLUS` | V1 | 7.106 | 84,8843% ± 0,1022 | 85,3735% ± 0,0606 | 84,6543313 | 0,110593 | dominated |
| 6 | `S-BETA-1` | V2 | 7.106 | 84,8843% ± 0,1022 | 85,3735% ± 0,0606 | 85,0711121 | 0,110051 | dominated |
| 7 | `F-SIGMOID` | V1 | 7.106 | 84,7461% ± 0,0773 | 85,2527% ± 0,0369 | 84,8627217 | 0,108693 | dominated |
| 8 | `S-BETA-0.5` | V2 | 7.106 | 84,7461% ± 0,0177 | 85,2814% ± 0,0188 | 85,0711121 | 0,108427 | dominated |
| 9 | `L3-IDENTITY` | V3 | 11.266 | 84,6796% ± 0,0177 | 85,2343% ± 0,0123 | 154,4654481 | 0,059285 | dominated |
| 10 | `L2-IDENTITY` | V3 | 7.106 | 84,6744% ± 0,0154 | 85,2466% ± 0,0163 | 84,0291601 | 0,108919 | pareto |
| 11 | `L1-DIRECT` | V3 | 218 | 84,5414% ± 0,0177 | 85,1135% ± 0,0872 | 2,6107501 | 3,454657 | pareto |

\* Teste apenas descritivo.

## Pareto global por validação

`L1-DIRECT` → `L2-IDENTITY` → `F-RELU` → `S-BETA-5`

Retornos marginais entre vizinhos:

| Transição | Δ validação | Δ GFLOPs | Retorno marginal |
|---|---:|---:|---:|
| `L1-DIRECT` → `L2-IDENTITY` | 0,133088 p.p. | 81,418410 | 0,001635 p.p./GFLOP |
| `L2-IDENTITY` → `F-RELU` | 0,373669 p.p. | 0,208390 | 1,793121 p.p./GFLOP |
| `F-RELU` → `S-BETA-5` | 0,148444 p.p. | 0,833562 | 0,178084 p.p./GFLOP |

Dominadas, com ao menos uma testemunha registrada no CSV:

- `F-SIGMOID` por `F-RELU`.
- `F-SWISH` por `S-BETA-5`.
- `F-SOFTPLUS` por `F-RELU`.
- `S-BETA-0.5` por `F-RELU`.
- `S-BETA-1` por `F-RELU`.
- `S-BETA-2` por `F-RELU`.
- `L3-IDENTITY` por `F-RELU`.

## Quatro respostas obrigatórias

1. **Melhor retorno por FLOP:** `L1-DIRECT`, com
   `3,454657` p.p./GFLOP, cerca de
   `30,4` vezes o segundo melhor retorno. Ele é
   barato, mas não tem a maior acurácia.
2. **Retornos decrescentes:** pelo limiar congelado, eles aparecem já
   em `L1 → L2`: o ganho é de apenas
   `0,133088`
   p.p. para
   `81,418410`
   GFLOPs. Em V3, `L2 → L3` cai para
   `0,0000727`
   p.p./GFLOP. A fronteira global não tem inclinações monotonicamente
   decrescentes; portanto não há um único cotovelo suave.
3. **Maior mudança de custo com pouco desempenho:** V3. De L1 para L3,
   o custo cresce `151,854698` GFLOPs
   (`59,17x`) para somente
   `0,138206` p.p. Em sentido contrário, V2 muda
   `0,450450` p.p. entre betas com o mesmo
   custo instrumentado; ainda assim, o efeito fica abaixo de 0,5 p.p.
4. **Escolha sob orçamento fixo:** `F-RELU`. Entre L1, L2 e ReLU,
   ela tem a maior validação (`85,0481%`)
   e ganha `0,506757` p.p. sobre L1, atingindo o limiar
   relevante. Se o objetivo fosse eficiência absoluta, L1 seria a
   escolha; sob o orçamento e priorizando acurácia, é ReLU.

## Leitura do teste oficial

- Maior validação: `S-BETA-5` = 85,1966%.
- Maior teste descritivo: `S-BETA-2` = 85,6458%.
- A troca de ordem é uma observação pós-congelamento, não motivo para
  escolher outra configuração ou repetir treinamento.

## Limitações

- Três seeds no mesmo split medem variação de inicialização, não de
  amostragem; DP não é intervalo de confiança.
- FLOPs são instrumentados e não medem tempo, energia ou custo completo.
- O encoder foi ajustado antes do hold-out, sem rótulos da validação.
- `F-SOFTPLUS` e `S-BETA-1` permanecem IDs distintos; têm métricas e
  predições iguais, mas instrumentações de custo diferentes.
- V3 mantém o risco acadêmico de ser interpretada como variável
  arquitetural, não como terceira variável de q01.
- A suíte ampla de desenvolvimento foi executada depois da avaliação
  oficial e inclui um teste do loader que acessa `adult.test`, além de
  treinos curtos. Esse acesso não executou os checkpoints oficiais,
  não gerou métricas de modelos e não alterou decisões ou resultados.

## Reprodução

```bash
python -m experiments.evaluate_official_test --verify-only
pytest -q test/test_plot_joint.py
python -m experiments.plot_joint
python -m experiments.evaluate_official_test --verify-only
```

Resultado esperado do teste focado: `6 passed`. O gerador apenas
verifica e lê a avaliação oficial já salva; não carrega o Adult test e
não executa treinamento. O comando amplo de pytest não faz parte desta
rota segura.
