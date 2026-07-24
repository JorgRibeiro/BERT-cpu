# Uso de IA

## 21/07/2026 — preparação da Variável 1

- Objetivo: criar os registros novos e formalizar configurações e hipóteses.
- Contribuição da IA: estrutura e texto inicial dos artefatos.
- Arquivos: `PROJECT_STATUS.md` e `experiments/`.
- Verificação: JSON válido, CSV com 32 colunas e `git diff --check` sem erros.
- Decisão do estudante: criação autorizada; revisão final do conteúdo ainda é
  responsabilidade do estudante.
- Proveniência: Passo 0 versionado com autorização.

## 21/07/2026 — implementação e smoke da Variável 1

- Objetivo: implementar q01 e ligá-la à classificação Adult sem tocar V2/V3.
- Contribuição da IA: forwards/backwards estáveis, FLOPs, cross-entropy estável,
  integração configurável, seeds separadas, CLI e testes automatizados.
- Arquivos principais: `bert_cpu/engine.py`, `bert_cpu/loss.py`,
  `exercises/q01_activations.py`, `exercises/task_binary_classification.py` e
  testes correspondentes.
- Verificação: 98 testes, três equações da q01, gráfico, estresse `+/-1000`,
  quatro smokes de duas épocas e repetição exata da ReLU.
- Correções durante a validação: o exemplo g3 revelou um defeito preexistente no
  backward de vetor `@` vetor; foi aplicada uma correção mínima com regressão.
- Decisão do estudante: início dos testes e commit da implementação de V1
  autorizados. O smoke não foi validado como resultado científico.

## 21/07/2026 — executor persistente da Variável 1

- Objetivo: tornar cada run rastreável antes do treino de 100 épocas.
- Contribuição da IA: runner unitário, logs JSONL, checkpoints NPZ, hashes,
  diagnósticos e travas contra duplicação ou execução fora do protocolo.
- Arquivos: `experiments/run_v1.py`, configuração, CSV, task Adult e testes.
- Verificação: 33 testes focados, 120 testes permitidos e smoke ReLU isolado;
  o teste oficial não foi carregado e `results.csv` permaneceu sem runs.
- Decisão do estudante: commit do executor autorizado; baseline ainda não
  iniciada.
- Proveniência remota: `origin/q01-ativacoes-adult` aponta para `eeed1f2` e o
  reflog registra `update by push`; nenhum comando de push foi executado pelo
  agente nesta etapa.

## 21/07/2026 — primeira run científica da baseline

- Objetivo: executar `F-RELU-s0-r1` por 100 épocas, sem acessar o teste.
- Contribuição da IA: execução, checagem do log, checkpoint, hashes, FLOPs e
  resumo descritivo da curva e dos diagnósticos.
- Verificação: status `completed_valid`, 100 épocas, checkpoint íntegro, seis
  diagnósticos e uma única linha em `results.csv`.
- Decisão do estudante: primeira run autorizada. Interpretação ainda parcial;
  repetição determinística e demais seeds pendentes.

## 21/07/2026 — reprodução da baseline

- Objetivo: repetir `F-RELU` com seed 0 sem mudar o contexto experimental.
- Contribuição da IA: execução e comparação independente das épocas,
  diagnósticos, pesos finais, métricas, hashes e FLOPs.
- Verificação: r1 e r2 foram exatamente iguais; ambos os artefatos estão
  íntegros e o teste oficial não foi carregado.
- Decisão do estudante: repetição autorizada; seeds 1 e 2 ainda pendentes.

## 21/07/2026 — conclusão da baseline ReLU

- Objetivo: executar as seeds 1 e 2 e consolidar as três seeds primárias.
- Contribuição da IA: execuções paralelas, validação dos quatro artefatos e
  cálculo das métricas agregadas da baseline.
- Verificação: média de validação 85,0481%, desvio-padrão amostral 0,1538 p.p.,
  FLOPs idênticos e nenhuma consulta ao teste oficial.
- Decisão do estudante: seeds 1 e 2 autorizadas conjuntamente; variantes ainda
  não executadas. As próximas ativações devem ser sempre analisadas com as três
  seeds e incluir losses de treino e validação.

## 21/07/2026 — variante Sigmoid

- Objetivo: executar `F-SIGMOID` nas seeds 0, 1 e 2 e comparar com ReLU.
- Contribuição da IA: execução do bloco, validação dos artefatos, análise de
  losses, acurácias, saturação, FLOPs e aplicação do critério de H1b.
- Verificação: sete runs totais válidas, Sigmoid 0,3020 p.p. abaixo da ReLU em
  média, saturação final média de 0,5444% e teste oficial não carregado.
- Interpretação: H1b inconclusiva porque a diferença não atingiu 0,5 p.p.; H1c
  tem apenas evidência parcial até concluir Swish e Softplus.

## 21/07/2026 — variante Swish

- Objetivo: executar `F-SWISH` nas seeds 0, 1 e 2 e comparar com as funções já
  concluídas.
- Contribuição da IA: execução do bloco, validação dos artefatos e análise de
  losses, acurácias, derivadas locais, FLOPs e critérios de H1a/H1c.
- Verificação: dez runs totais válidas, Swish 0,0870 p.p. acima da ReLU em
  média, 0,9895% mais FLOPs e teste oficial não carregado.
- Interpretação: Swish é a melhor suave parcial, mas H1a permanece pendente até
  executar Softplus; H1c segue com evidência parcial.

## 21/07/2026 — variante Softplus e síntese da V1

- Objetivo: executar `F-SOFTPLUS` nas três seeds e consolidar a Variável 1.
- Contribuição da IA: execução, validação dos 13 artefatos, análise de losses,
  acurácias, derivadas, FLOPs, retorno, hipóteses e Pareto.
- Verificação: Softplus 0,1638 p.p. abaixo da ReLU e 0,4948% mais cara; todas
  as runs válidas e teste oficial não carregado.
- Interpretação: H1a e H1b inconclusivas; H1c sustentada. ReLU teve o melhor
  retorno por FLOP e Swish a maior acurácia média.

## 21/07/2026 — gráficos e revisão final da V1

- Objetivo: tornar tabela e gráficos regeneráveis a partir dos dados brutos.
- Contribuição da IA: gerador, três gráficos, tabela agregada, testes e revisão
  independente dos 13 logs e das conclusões.
- Verificação: inspeção visual corrigiu sobreposição de título/legenda; três
  testes novos e 123 testes permitidos passaram.
- Limites preservados: repetição ReLU excluída das médias, três seeds não
  tratadas como intervalo de confiança e teste oficial não acessado.

## 23/07/2026 — preparação da Variável 2

- Objetivo: implementar e preparar o estudo da curvatura da Softplus antes das
  novas runs.
- Contribuição da IA: protocolo e H2, Softplus-beta estável, backward, FLOPs,
  integração na AdultMLP, executor unitário, lote sequencial, análise e testes.
- Arquivos principais: `bert_cpu/engine.py`, `exercises/q01_activations.py`,
  `exercises/task_binary_classification.py`, configuração/protocolo da V2 e
  testes correspondentes.
- Verificação: 196 testes permitidos e 73 testes focados passaram. Os quatro
  smokes concluíram duas épocas, custos e checkpoints esperados; o dry-run
  planejou 12 runs. Revisões independentes encontraram e ajudaram a fechar as
  travas de retomada e artefatos temporários.
- Decisão do estudante: início da V2 autorizado. Nenhuma run científica,
  avaliação no teste oficial, commit ou push foi realizado nesta etapa.

## 24/07/2026 — referência da Variável 2

- Objetivo: versionar a infraestrutura e executar `S-BETA-1` nas seeds 0, 1 e
  2 antes das demais curvas.
- Contribuição da IA: commit pré-runs autorizado, três execuções sequenciais e
  validação de logs, checkpoints, métricas, hashes e FLOPs.
- Verificação: três runs `completed_valid`; média de validação 84,8843%,
  desvio-padrão amostral 0,1022 p.p. e 85,0711121 GFLOPs por run.
- Limites: resultado ainda parcial, sem decisão de H2 e sem avaliação no teste
  oficial. Nenhum push ou commit dos resultados foi realizado.

## 24/07/2026 — conclusão das execuções da Variável 2

- Objetivo: executar `beta=0,5`, `2` e `5`, sempre nas três seeds, e validar o
  grid completo.
- Contribuição da IA: lote sequencial retomável, acompanhamento das nove runs e
  revalidação conjunta dos 12 logs e checkpoints.
- Verificação: 12 runs `completed_valid`, um único contexto experimental,
  85,0711121 GFLOPs por run e `test_accuracy` vazio em todas.
- Limite: tabela, gráficos, diagnósticos e H2 ainda não foram analisados.
  Nenhum commit ou push dos resultados foi realizado.

## 24/07/2026 — análise da Variável 2

- Objetivo: consolidar as 12 runs, analisar convergência e diagnósticos e
  aplicar H2 sem alterar o protocolo.
- Contribuição da IA: tabela agregada, três gráficos, análise escrita,
  verificação visual e recálculo independente de H2.
- Verificação: beta 2 venceu o grupo central, beta 5 venceu o extremo e a
  diferença foi -0,1587 p.p.; H2 ficou inconclusiva. Beta 1 reproduziu
  exatamente pesos e métricas da Softplus da V1. Os 12 artefatos foram
  revalidados e 196 testes permitidos passaram.
- Limites: três seeds, split fixo, FLOPs instrumentados e teste oficial não
  avaliado. O estudante proibiu commit durante a análise e depois autorizou
  explicitamente o commit de encerramento. Nenhum push foi autorizado.

## 24/07/2026 — preparação da Variável 3

- Objetivo: preparar o estudo da profundidade linear sem ativação.
- Contribuição da IA: classe V3 isolada, colapso afim, protocolo, H3a–H3d,
  configuração congelada, executores, análise, checkpoints e testes.
- Decisão técnica: encadear `Linear` diretamente, sem criar função ou camada
  `Identity`; preservar integralmente os caminhos de V1/V2.
- Verificação: 269 testes permitidos e três smokes de duas épocas passaram.
  Parâmetros e FLOPs por época foram 218/26.107.501, 7.106/840.291.601 e
  11.266/1.544.654.481. Checkpoints e equivalência `W*x+b` foram validados.
- Limites: smoke não testa hipóteses; nenhuma run científica ou consulta ao
  teste oficial ocorreu. O estudante autorizou depois o commit pré-runs
  `2c15768`; nenhum push foi realizado.

## 24/07/2026 — execução e análise da Variável 3

- Objetivo: executar as nove runs e aplicar H3a–H3d sem consultar o teste.
- Contribuição da IA: execuções sequenciais, validação de logs/checkpoints,
  tabela, três gráficos, análise escrita e revisão independente.
- Verificação: nove runs em um único contexto, seeds `0`, `1` e `2`, artefatos
  íntegros e ponte controlada entre `F-RELU` e `L2-IDENTITY`.
- Resultado: H3a não contradita, H3b/H3c sustentadas e H3d inconclusiva.
- Limites: três seeds, split fixo, FLOPs instrumentados e kernels diferentes na
  ponte com ReLU. O teste oficial não foi consultado. O estudante autorizou o
  commit de encerramento; nenhum push foi autorizado.

## 24/07/2026 — avaliação oficial

- Objetivo: avaliar os 33 checkpoints primários no Adult test, sem treinar.
- Contribuição da IA: inventário fechado, preflight, avaliador idempotente,
  testes, manifesto, log e CSV separado dos resultados científicos.
- Verificação: 280 testes; reprodução exata da validação antes do acesso; um
  load do teste, 33 forwards, zero falhas e duas auditorias independentes.
- Resultado descritivo: beta 2 teve 85,6458% e Swish 85,5967% de média.
- Limites: o teste não será usado para mudar decisões. A análise conjunta,
  commit e push permanecem pendentes.

## 24/07/2026 — análise conjunta

- Objetivo: reunir V1, V2 e V3 e responder às quatro perguntas de eficiência.
- Contribuição da IA: junção verificável das 33 runs, tabela, Pareto, retornos
  marginais, orçamento fixo, análise escrita, três gráficos e testes.
- Verificação: três auditorias somente leitura, seis testes focados e 286 testes
  permitidos. O gerador bloqueia reevaluar ou carregar o Adult test e preserva
  os quatro artefatos oficiais.
- Resultado: L1 teve o melhor retorno; ReLU foi escolhida sob seu orçamento; a
  Pareto global foi L1, L2, ReLU e beta 5.
- Limites: teste apenas descritivo, FLOPs instrumentados, três seeds no mesmo
  split e risco acadêmico de V3. Nenhum commit ou push foi realizado.

## 24/07/2026 — preparação da entrega e auditoria de reprodução

- Objetivo: tornar README, dependências, reprodução e roteiro fiéis ao estado
  final do estudo.
- Contribuição da IA: reescrita do README, versões de ambiente, rota segura de
  reprodução, mapa dos 12 requisitos e roteiro cronometrado de 19min20s.
- Correção importante: a auditoria constatou que a suíte ampla de 286 testes
  inclui treinos curtos e um teste de loader que acessa `adult.test`. Esse
  acesso posterior não executou checkpoints, não gerou métricas e não alterou
  hipótese, seleção, Pareto ou resultados.
- Decisão aplicada: a reprodução final usa `--verify-only`, seis testes focados
  e os geradores. O comando amplo ficou registrado apenas como evidência
  histórica de desenvolvimento.
- Verificação: comandos conferidos contra os parsers, seis testes focados
  aprovados, gráficos regenerados e os quatro hashes da avaliação oficial
  preservados. As versões instaladas coincidem com `requirements.txt`.
- Ambiente: `pip check` do ambiente global apontou conflitos externos entre
  FastAPI/Starlette e Pyppeteer/WebSockets; eles não são dependências deste
  projeto. O README orienta criar um ambiente virtual limpo.
- Teste adicional: antes do commit, quatro preflights recusaram corretamente o
  `requirements.txt` ainda não versionado. Depois do commit, sem alterar a
  trava, avaliador e análise conjunta terminaram com `17 passed`;
  `--verify-only` continuou aprovado.
- Revisão e gravação do vídeo continuam sendo responsabilidade do estudante.
- Limites: o commit de fechamento foi autorizado pelo estudante; nenhum link de
  vídeo ou push foi criado nesta etapa.
