# Passo a passo — Adult + q01

Este é o roteiro operacional do projeto. As justificativas e regras completas
ficam em `AGENTS.md`.

## Estado atual

- Branch: `q01-ativacoes-adult`, criada a partir da `main`.
- q04 foi abandonada antes de implementação ou experimento.
- Adult, q01, baseline, variáveis e protocolo estão confirmados.
- A V1 tem 12 runs primárias e uma repetição determinística válidas.
- H1a/H1b ficaram inconclusivas e H1c foi sustentada.
- Tabela e três gráficos da V1 são regeneráveis.
- O teste oficial não foi consultado e os artefatos foram versionados no commit
  de fechamento da V1.
- A V2 tem 12 runs válidas, tabela, três gráficos e H2 inconclusiva.
- A V2 foi encerrada no commit `de000de`.
- A infraestrutura da V3 foi versionada no commit `2c15768`.
- A V3 tem nove runs válidas, tabela, três gráficos e análise de H3a–H3d.
- H3a não foi contradita, H3b/H3c foram sustentadas e H3d ficou inconclusiva.
- Os 33 checkpoints primários foram avaliados no teste oficial, sem treinar.
- A avaliação final está em `experiments/final_evaluation/`.
- A análise conjunta está em `experiments/final_analysis/`.
- Pareto global: L1, L2, ReLU e beta 5; melhor retorno: L1; escolha sob o
  orçamento da ReLU: ReLU.
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
pytest -q test/test_q01_activations.py
```

### 3. Executar a Variável 1 — concluído localmente em 21/07/2026

- [x] Tornar a mesma `AdultMLP` configurável para as quatro ativações.
- [x] Separar o split da seed de inicialização.
- [x] Impedir consulta automática ao teste.
- [x] Fazer smoke de duas épocas e repetir ReLU com resultado idêntico.
- [x] Preparar logs JSONL, checkpoints NPZ, hashes e travas contra duplicação.
- [x] Validar o executor com testes e smoke isolado.
- [x] Versionar o executor após autorização.
- [x] Executar e reproduzir `F-RELU` primeiro.
- [x] Executar as quatro configurações com seeds `0`, `1` e `2`.
- [x] Salvar resultados e checkpoints.
- [x] Analisar H1a, H1b e H1c.
- [x] Gerar tabela e gráficos reproduzíveis.
- [x] Versionar o fechamento após autorização.

### 4. Executar a Variável 2 — concluída localmente em 24/07/2026

Somente depois de fechar V1:

- [x] registrar protocolo, níveis e hipótese antes das runs;
- [x] implementar Softplus-beta sem alterar a Softplus da V1;
- [x] testar valores, derivadas, integração e FLOPs;
- [x] preparar e validar executor unitário, lote e análise;
- [x] fazer smoke das quatro configurações;
- [x] versionar a infraestrutura após autorização;
- [x] executar os quatro valores de `beta` nas três seeds;
- [x] gerar tabela, três gráficos e avaliar H2;
- [x] revalidar os 12 artefatos e os 196 testes permitidos;
- [x] versionar o encerramento após autorização.

### 5. Executar a Variável 3 — análise concluída localmente em 24/07/2026

Somente depois de fechar V2:

- [x] registrar protocolo e H3a–H3d antes das runs;
- [x] implementar as três arquiteturas sem ativação;
- [x] validar parâmetros, FLOPs e equivalência afim;
- [x] preparar executor, lote, análise e checkpoints;
- [x] fazer três smokes de duas épocas;
- [x] validar os 269 testes permitidos;
- [x] versionar a infraestrutura após autorização;
- [x] executar as três profundidades nas três seeds;
- [x] comparar `L2-IDENTITY` com `F-RELU`;
- [x] gerar tabela, três gráficos e analisar H3a–H3d;
- [x] versionar o encerramento após autorização.

### 6. Finalizar a análise — concluída e versionada em 24/07/2026

- [x] Avaliar no teste todos os checkpoints primários, sem mudar decisões.
- [x] Consolidar os resultados e gerar os gráficos conjuntos.
- [x] Comparar acurácia, parâmetros e FLOPs.
- [x] Identificar configurações dominadas e fronteira de Pareto.
- [x] Responder às quatro perguntas obrigatórias sobre retorno por FLOP, retornos
  decrescentes, variável de maior custo e escolha sob orçamento fixo.

### 7. Preparar a entrega — base versionada em 24/07/2026

- [x] Atualizar `README.md` e dependências.
- [x] Finalizar registros de IA, dificuldades e reprodução segura.
- [x] Ligar os 12 requisitos do vídeo às evidências do repositório.
- [x] Preparar o roteiro de até 20 minutos.
- [x] Versionar o fechamento após autorização.
- [ ] Estudante revisar, gravar o vídeo e inserir o link.
- [ ] Enviar no Classroom.

## Próximo passo

O estudante revisar o roteiro, gravar o vídeo e inserir seu link. Depois,
preparar o envio no Classroom e decidir separadamente sobre o push.
