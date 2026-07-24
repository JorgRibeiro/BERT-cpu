# Protocolo — Variável 3

## Pergunta

Sem ativação, aumentar a profundidade muda desempenho e custo mesmo sem ampliar
a classe de funções, que continua afim?

## Configurações

- `L1-DIRECT`: `108 -> 2`;
- `L2-IDENTITY`: `108 -> 64 -> 2`;
- `L3-IDENTITY`: `108 -> 64 -> 64 -> 2`.

Não existe camada `Identity`: a saída de cada `Linear` entra diretamente na
próxima. As três composições podem ser reduzidas mecanicamente a `W*x + b`.

## Execução

- 100 épocas, Adam full-batch e `lr=1e-2`;
- mesmo split; seeds de modelo `0`, `1` e `2`;
- registrar losses, acurácias, parâmetros, FLOPs, hashes e checkpoint;
- diagnósticos afins nas épocas `0`, `1`, `25`, `50`, `75` e `100`;
- não consultar o teste oficial.

Ordem fixa:

```text
L1-DIRECT:   seeds 0, 1, 2
L2-IDENTITY: seeds 0, 1, 2
L3-IDENTITY: seeds 0, 1, 2
```

| ID | Parâmetros | FLOPs/época | GFLOPs/100 épocas |
|---|---:|---:|---:|
| `L1-DIRECT` | 218 | 26.107.501 | 2,6107501 |
| `L2-IDENTITY` | 7.106 | 840.291.601 | 84,0291601 |
| `L3-IDENTITY` | 11.266 | 1.544.654.481 | 154,4654481 |

Smokes de duas épocas validam apenas código e custos. As nove runs científicas
só poderão ocorrer depois de versionar a infraestrutura com autorização.

## Análise

Usar a validação da época 100 e a margem de `0,5` p.p. Também comparar
`L2-IDENTITY` com a `F-RELU` já registrada, porque têm as mesmas duas camadas
lineares e diferem pela presença da ReLU.

Regras:

- H3a é contradita se L2 ou L3 ganhar `0,5` p.p. com sinal positivo em duas
  seeds; fora disso, fica “não contradita”, sem alegar equivalência;
- H3b exige crescimento estrito de parâmetros e FLOPs;
- H3c exige `R(L1) > R(L2) > R(L3)`; reversão estrita refuta e empate ou ordem
  mista deixa inconclusiva;
- H3d aplica a regra direcional comum a `F-RELU - L2-IDENTITY`.

`F-RELU` vem da V1, em outro commit. A ponte só será aceita após conferir dados,
split, arquitetura linear, treinamento, instrumentação, software e hashes
iniciais. O kernel/plataforma também será comparado e qualquer diferença ficará
explícita. Mesmo assim, não é o mesmo contexto literal de execução. Exigir o
mesmo commit criaria três runs ReLU extras, fora do plano confirmado de nove.

O risco acadêmico permanece: V3 pode ser interpretada como variável
arquitetural, e não como uma terceira variável de q01.
