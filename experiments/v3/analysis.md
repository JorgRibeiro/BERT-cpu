# Análise — Variável 3

## Ideia

Comparar uma, duas e três camadas `Linear` sem ativação. Todas continuam
equivalentes a uma única função afim `W*x+b`; mudam a parametrização, a
otimização, a quantidade de parâmetros e os FLOPs.

Foram validadas nove runs: três arquiteturas nas seeds `0`, `1` e `2`, com 100
épocas. O teste oficial não foi consultado.

## Resultados na época 100

| Configuração | Loss treino | Loss validação | Acurácia treino | Acurácia validação ± DP | Parâmetros | GFLOPs/run | Retorno p.p./GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L1-DIRECT` | 0,319824 | 0,331773 | 85,2522% | 84,5414% ± 0,0177 | 218 | 2,6107501 | 3,4547 |
| `L2-IDENTITY` | 0,313991 | 0,327721 | 85,4492% | 84,6744% ± 0,0154 | 7.106 | 84,0291601 | 0,1089 |
| `L3-IDENTITY` | 0,313938 | 0,327302 | 85,4543% | 84,6796% ± 0,0177 | 11.266 | 154,4654481 | 0,0593 |

O desvio-padrão é amostral entre três seeds. O retorno usa a acurácia
majoritária de validação, 75,5221%.

## Hipóteses

### H3a — profundidade linear

- `L2 - L1` por seed: `+0,1382`, `+0,1382` e `+0,1229` p.p.; média
  `+0,1331` p.p.
- `L3 - L1` por seed: `+0,1382`, `+0,1382` e `+0,1382` p.p.; média
  `+0,1382` p.p.

**Não contradita.** Os ganhos foram positivos, mas ficaram muito abaixo do
limiar pré-definido de `0,5` p.p. Isso não prova equivalência entre os modelos.

### H3b — custo

**Sustentada.** Parâmetros e FLOPs cresceram estritamente:

```text
L1-DIRECT < L2-IDENTITY < L3-IDENTITY
```

Em relação a L1, L2 usou cerca de 32,19 vezes mais FLOPs e L3, 59,17 vezes
mais.

### H3c — retorno por FLOP

**Sustentada.** A ordem medida foi:

```text
R(L1)=3,4547 > R(L2)=0,1089 > R(L3)=0,0593 p.p./GFLOP
```

As três médias pertencem à Pareto calculada somente dentro da V3, pois cada
ganho mínimo de acurácia veio acompanhado de maior custo. Isso não significa
que as três pertençam à fronteira conjunta do projeto.

### H3d — efeito da ReLU

`F-RELU - L2-IDENTITY` por seed foi `+0,1843`, `+0,4453` e `+0,4914` p.p.;
média `+0,3737` p.p.

**Inconclusiva.** A ReLU venceu nas três seeds, mas a média não atingiu o
limiar de `0,5` p.p.

A ponte confirmou dados, split, pesos iniciais, arquitetura linear,
hiperparâmetros, procedimento, instrumentação, Python, NumPy e dependências.
Ainda assim, a ReLU veio de outro commit e do kernel Linux 7.1.3, enquanto V3
usou 7.1.4; portanto, não é um contexto literal idêntico.

## Convergência

As redes profundas chegaram antes aos melhores checkpoints secundários:

- L1: épocas `91`, `100` e `92`;
- L2: épocas `43`, `53` e `45`;
- L3: épocas `30`, `57` e `24`.

A decisão principal continua baseada na época 100, como definido antes das
runs. Os checkpoints anteriores não foram usados para escolher resultados.

## Conclusão

No Adult e neste protocolo, fatorar a função afim em mais camadas melhorou a
acurácia final em apenas cerca de `0,13 p.p.`, abaixo da margem relevante, e
aumentou fortemente parâmetros e FLOPs. L1 foi a escolha mais eficiente da V3.
Esse resultado trata redes lineares sem ativação e não permite concluir que
profundidade seja inútil em redes não lineares.

Limitações: três seeds sobre um split fixo, FLOPs apenas instrumentados,
pré-processamento ajustado antes do hold-out, teste oficial ainda reservado e
risco de V3 ser interpretada como variável arquitetural em vez de variável de
q01.

## Reprodução

```bash
python -m experiments.plot_v3
python -m experiments.run_v3_all --dry-run --quiet
pytest -q --ignore=test/test_model.py
```

Dados brutos: `experiments/v3/results.csv` e `experiments/v3/logs/`.
