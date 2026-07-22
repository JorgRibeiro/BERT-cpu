# Passo a passo — Adult + q01

Este é o roteiro operacional do projeto. As justificativas e regras completas
ficam em `AGENTS.md`.

## Estado atual

- Branch: `q01-ativacoes-adult`, criada a partir da `main`.
- q04 foi abandonada antes de implementação ou experimento.
- Adult, q01, baseline, variáveis e protocolo estão confirmados.
- Artefatos novos, configuração e hipóteses de V1 estão registrados.
- q01 e a integração Adult de V1 passaram nos testes e no smoke de duas épocas.
- Nenhuma run de 100 épocas ou conclusão experimental foi produzida.
- Não fazer commit ou push sem autorização.

## O que vamos testar

### Variável 1 — função de ativação

Mesma MLP `108 -> 64 -> 2`, mudando apenas a ativação:

- `F-RELU` — baseline;
- `F-SIGMOID`;
- `F-SWISH`;
- `F-SOFTPLUS`.

### Variável 2 — curvatura da Softplus

Usar `softplus_beta(z) = log(1 + exp(beta*z)) / beta` com:

- `S-BETA-0.5`;
- `S-BETA-1`;
- `S-BETA-2`;
- `S-BETA-5`.

### Variável 3 — profundidade sem ativação

- `L1-DIRECT`: `Linear(108,2)` — 218 parâmetros;
- `L2-IDENTITY`: `Linear(108,64) -> Linear(64,2)` — 7.106 parâmetros;
- `L3-IDENTITY`: `Linear(108,64) -> Linear(64,64) -> Linear(64,2)` — 11.266 parâmetros.

V3 seguirá sem consulta ao professor. Manter registrado o risco de ela ser
considerada uma variável arquitetural, e não uma terceira variável de q01.

## Protocolo fixo

- Full-batch, Adam, `lr=1e-2`, 100 épocas e validação de 20%.
- Um único split gerado com seed 0.
- Seeds de modelo: `0`, `1` e `2`.
- Métrica principal: média da acurácia de validação na época 100.
- Diferença relevante: 0,5 ponto percentual, com mesmo sinal em pelo menos duas
  das três seeds.
- Teste oficial somente no final, para todas as configurações válidas.
- Salvar configuração, log, métricas, FLOPs e checkpoint de cada run.
- Preservar o loader atual e registrar sua limitação de pré-processamento.
- Custos por elemento: Identity 0, ReLU 1, Sigmoid 4, Swish 5, Softplus 3 e
  Softplus-beta 5 FLOPs instrumentados.
- Retorno por FLOP:
  `(acurácia_val - acurácia_majoritária_val) / GFLOPs_totais`.

Planejamento padrão: 11 configurações, três seeds e 33 treinamentos, mais uma
reexecução de `F-RELU` para conferir determinismo.

## Execução

### 1. Preparar os registros — concluído em 21/07/2026

- [x] Criar `experiments/` e `PROJECT_STATUS.md` novos.
- [x] Registrar configurações e hipóteses antes dos resultados.
- [x] Registrar branch e commit base; diff e hashes serão congelados antes das
  runs completas.

### 2. Implementar a q01 básica — concluído em 21/07/2026

- [x] Implementar e testar Sigmoid, Swish e Softplus estáveis.
- [x] Instrumentar os FLOPs confirmados.
- [x] Gerar o gráfico da q01 e registrar Matplotlib.
- [x] Estabilizar a cross-entropy após reproduzir a falha extrema.

Validar:

```bash
python -m exercises.q01_activations
pytest -q --ignore=test/test_model.py
```

### 3. Executar a Variável 1

- [x] Tornar a mesma `AdultMLP` configurável para as quatro ativações.
- [x] Separar o split da seed de inicialização.
- [x] Impedir consulta automática ao teste.
- [x] Fazer smoke de duas épocas e repetir ReLU com resultado idêntico.
- Executar e reproduzir `F-RELU` primeiro.
- Executar as quatro configurações com seeds `0`, `1` e `2`.
- Salvar resultados e checkpoints.
- Analisar as hipóteses e encerrar V1 antes de continuar.

### 4. Executar a Variável 2

Somente depois de fechar V1:

- implementar Softplus-beta;
- testar valores, derivadas e FLOPs;
- fazer smoke test;
- executar os quatro valores de `beta` nas três seeds;
- analisar e encerrar V2.

### 5. Executar a Variável 3

Somente depois de fechar V2:

- implementar as três arquiteturas sem ativação;
- verificar contagem de parâmetros e equivalência com uma função afim única;
- fazer smoke test;
- executar as três profundidades nas três seeds;
- comparar também `L2-IDENTITY` com `F-RELU`;
- analisar e encerrar V3.

### 6. Finalizar a análise

- Avaliar no teste todos os checkpoints válidos, sem mudar decisões.
- Consolidar `results.csv` e gerar os gráficos.
- Comparar acurácia, parâmetros e FLOPs.
- Identificar configurações dominadas e fronteira de Pareto.
- Responder às quatro perguntas obrigatórias sobre retorno por FLOP, retornos
  decrescentes, variável de maior custo e escolha sob orçamento fixo.

### 7. Preparar a entrega

- Atualizar `README.md` e `requirements.txt`.
- Finalizar registros de IA, dificuldades e reprodução.
- Ligar os 12 requisitos do vídeo às evidências do repositório.
- Preparar vídeo de até 20 minutos, link do GitHub e envio no Classroom.

## Próximo passo

Preparar logs/checkpoints e executar a baseline definitiva antes das variantes.
Não implementar V2 ou V3.
