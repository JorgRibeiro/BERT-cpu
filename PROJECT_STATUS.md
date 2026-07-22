# Estado do projeto

Atualizado em 21/07/2026.

## Agora

- Objetivo: comparar funções de ativação na classificação Adult.
- Fase: Variável 1 fechada e versionada.
- Branch: `q01-ativacoes-adult`.
- Commit base: `14e65c20c2312e66ba76e54431b396f60ce65e10`.
- Passo 0: `b334d37` (`docs: establish q01 experiment step 0`).
- Commit do executor: `5ae67ab` (`feat: add reproducible v1 experiment runner`).
- Fechamento da V1: este commit.
- `origin/q01-ativacoes-adult` aponta para `5ae67ab`; após este fechamento, a
  branch local fica um commit à frente. Nenhum push foi executado nesta etapa.
- Regra atual: executar e analisar cada ativação sempre com seeds 0, 1 e 2,
  incluindo loss, acurácia e FLOPs.

## Concluído

- Escopo Adult + q01 definido.
- Protocolo pré-experimental confirmado.
- Artefatos novos inaugurados.
- Configurações e hipóteses de V1 registradas antes dos resultados.
- Sigmoid, Swish e Softplus implementadas com backward e FLOPs testados.
- Cross-entropy extrema estabilizada antes da baseline.
- Adult configurável para V1, com seeds separadas e teste somente opt-in.
- Smoke de duas épocas concluído nas quatro ativações; ReLU repetida exatamente.
- Executor persistente preparado com logs JSONL, checkpoint NPZ, hashes,
  diagnósticos e proteção contra runs duplicadas.
- Smoke isolado do executor concluído sem teste e sem linha em `results.csv`.
- `F-RELU-s0-r1` concluiu 100 épocas e teve seus artefatos verificados.
- `F-RELU-s0-r2` reproduziu exatamente métricas, diagnósticos, pesos e FLOPs.
- `F-RELU-s1-r1` e `F-RELU-s2-r1` concluíram 100 épocas com artefatos válidos.
- Portão da baseline completo: seed 0 reproduzida e seeds 1 e 2 registradas.
- `F-SIGMOID` concluiu as seeds 0, 1 e 2 com artefatos válidos.
- `F-SWISH` concluiu as seeds 0, 1 e 2 com artefatos válidos.
- `F-SOFTPLUS` concluiu as seeds 0, 1 e 2 com artefatos válidos.
- As quatro ativações foram consolidadas e H1a, H1b e H1c avaliadas.
- Tabela agregada e três gráficos foram gerados por script e inspecionados.
- Uma auditoria independente confirmou dados, médias, hipóteses e Pareto.

## Ainda não feito

- V2 não foi iniciada.

## Riscos

- V3 pode ser considerada arquitetural, não uma terceira variável de q01.
- O smoke não pode ser interpretado como resultado experimental.
- H1a e H1b permaneceram inconclusivas pelo limiar pré-definido.
- A melhor acurácia intermediária ocorreu antes da época 100; ela permanece
  apenas como análise secundária, conforme o protocolo.
- Três seeds não formam intervalo de confiança; com split fixo, a dispersão
  representa apenas a inicialização.

## Entregáveis

- README: pendente.
- Reprodução: 13 runs, tabela e três gráficos regeneráveis.
- Vídeo: mapa, implementação e evidências científicas parciais.

## Evidência

- `pytest -q --ignore=test/test_model.py`: 123 testes passaram; 14 avisos de
  depreciação internos de Matplotlib/PyParsing.
- Testes focados do executor e da Adult: 33 passaram.
- Smoke isolado de `F-RELU`: duas épocas, checkpoint verificado, teste oficial
  não carregado e zero resultados científicos registrados.
- As três equações da q01 e o gráfico foram validados.
- Estresse `+/-1000`: losses 0/2000 e gradientes finitos.
- Split SHA-256 (forma, dtype e bytes dos índices):
  `118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0`.
- Pesos iniciais seed 0 SHA-256:
  `9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596`.
- `F-RELU-s0-r1`: validação 84,8741%, treino 87,2202% e 84,2375505 GFLOPs
  instrumentados na época 100.
- Melhor validação secundária: 85,3808% na época 64.
- Log: `experiments/logs/F-RELU-s0-r1.jsonl`.
- Checkpoint: `experiments/checkpoints/F-RELU-s0-r1.npz`, SHA-256
  `ee4e94545ecc7bb90d2c8b0521b34f3d1961bc213500e97fecc6917fa0e48704`.
- `F-RELU-s0-r2`: os 100 registros de época, seis diagnósticos e pesos finais
  são idênticos aos de r1; checkpoint SHA-256
  `4947377540f8d6104a3476b47f89c9f807275c74b4b1c8e4257bff16bafcb611`.
- Validação na época 100: seed 0 = 84,8741%, seed 1 = 85,1044% e seed 2 =
  85,1658%.
- Média de validação: 85,0481%; desvio-padrão amostral: 0,1538 p.p.
- Cada run custou 84,2375505 GFLOPs; as três seeds primárias consumiram
  252,7126515 GFLOPs e a repetição elevou o executado a 336,9502020 GFLOPs.
- Sigmoid: loss média de treino 0,312743 e de validação 0,325530; acurácia
  média de validação 84,7461% com desvio-padrão amostral 0,0773 p.p.
- Sigmoid ficou 0,3020 p.p. abaixo da ReLU em média e nas três seeds, diferença
  menor que o limiar de 0,5 p.p.; H1b permanece inconclusiva.
- Sigmoid custou 84,8627217 GFLOPs/run, 0,6251712 GFLOP ou 0,7422% acima da
  ReLU. A saturação média foi 0,5444% na época 100.
- Swish: loss média de treino 0,294821 e de validação 0,314872; acurácia média
  de validação 85,1351% com desvio-padrão amostral 0,0266 p.p.
- Swish ficou 0,0870 p.p. acima da ReLU em média, sem atingir os 0,5 p.p. de
  H1a, e custou 85,0711121 GFLOPs/run, 0,9895% acima da ReLU.
- Softplus: loss média de treino 0,308014 e de validação 0,322490; acurácia
  média de validação 84,8843% com desvio-padrão amostral 0,1022 p.p.
- Softplus ficou 0,1638 p.p. abaixo da ReLU e custou 84,6543313 GFLOPs/run,
  0,4948% acima da ReLU.
- Ranking por validação: Swish 85,1351%, ReLU 85,0481%, Softplus 84,8843% e
  Sigmoid 84,7461%.
- H1a inconclusiva; H1b inconclusiva; H1c sustentada.
- Fronteira de Pareto parcial: ReLU e Swish. Sigmoid e Softplus são dominadas.
- Custo executado da V1: 1.100,7146973 GFLOPs, incluindo a repetição ReLU.
- Teste oficial não carregado; `results.csv` contém 13 runs científicas.
- Regeneração: `python -m experiments.plot_v1`.
- Tabela: `experiments/v1_summary.csv`.
- Gráficos: `experiments/plots/v1_learning_curves.png`,
  `v1_final_metrics_by_seed.png` e `v1_accuracy_vs_flops.png`.

## Próximo passo

Revisar as evidências da V1 com o estudante. Iniciar a preparação da V2 apenas
mediante nova solicitação; não iniciar V3 nem consultar o teste oficial.
