# Estado do projeto

Atualizado em 21/07/2026.

## Agora

- Objetivo: comparar funções de ativação na classificação Adult.
- Fase: executor da Variável 1 validado e autorizado para commit.
- Branch: `q01-ativacoes-adult`.
- Commit base: `14e65c20c2312e66ba76e54431b396f60ce65e10`.
- Passo 0: `b334d37` (`docs: establish q01 experiment step 0`).
- Commit anterior: `eeed1f2` (`feat: implement and validate q01 activation family`).
- `origin/q01-ativacoes-adult` também aponta para `eeed1f2`; o reflog local
  registra `update by push`, embora o agente não tenha executado esse comando.
- O executor é versionado no commit que contém este registro.

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

## Ainda não feito

- Nenhuma run definitiva de 100 épocas foi executada.
- Nenhum log, checkpoint ou linha científica de `results.csv` foi produzido.
- Nenhuma hipótese foi sustentada, refutada ou julgada inconclusiva.

## Riscos

- V3 pode ser considerada arquitetural, não uma terceira variável de q01.
- O smoke não pode ser interpretado como resultado experimental.
- O executor rejeita corretamente uma run científica enquanto estiver sem
  commit; isso preserva a proveniência dos resultados.

## Entregáveis

- README: pendente.
- Reprodução: executor e configuração validados; runs definitivas pendentes.
- Vídeo: mapa e evidências de implementação parciais; resultados pendentes.

## Evidência

- `pytest -q --ignore=test/test_model.py`: 120 testes passaram.
- Testes focados do executor e da Adult: 33 passaram.
- Smoke isolado de `F-RELU`: duas épocas, checkpoint verificado, teste oficial
  não carregado e zero resultados científicos registrados.
- As três equações da q01 e o gráfico foram validados.
- Estresse `+/-1000`: losses 0/2000 e gradientes finitos.
- Split SHA-256 (forma, dtype e bytes dos índices):
  `118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0`.
- Pesos iniciais seed 0 SHA-256:
  `9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596`.

## Próximo passo

Executar e reproduzir a `F-RELU` definitiva antes das variantes. Não iniciar
V2/V3.
