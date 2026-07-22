# Estado do projeto

Atualizado em 21/07/2026.

## Agora

- Objetivo: comparar funções de ativação na classificação Adult.
- Fase: implementação e smoke da Variável 1 concluídos.
- Branch: `q01-ativacoes-adult`.
- Commit base: `14e65c20c2312e66ba76e54431b396f60ce65e10`.
- Passo 0: `b334d37` (`docs: establish q01 experiment step 0`).
- Etapa versionada no commit atual: `feat: implement and validate q01 activation family`.
- Nenhum push foi realizado.

## Concluído

- Escopo Adult + q01 definido.
- Protocolo pré-experimental confirmado.
- Artefatos novos inaugurados.
- Configurações e hipóteses de V1 registradas antes dos resultados.
- Sigmoid, Swish e Softplus implementadas com backward e FLOPs testados.
- Cross-entropy extrema estabilizada antes da baseline.
- Adult configurável para V1, com seeds separadas e teste somente opt-in.
- Smoke de duas épocas concluído nas quatro ativações; ReLU repetida exatamente.

## Ainda não feito

- Nenhuma run definitiva de 100 épocas foi executada.
- Logs, checkpoints e linhas de `results.csv` ainda não foram produzidos.
- Nenhuma hipótese foi sustentada, refutada ou julgada inconclusiva.

## Riscos

- V3 pode ser considerada arquitetural, não uma terceira variável de q01.
- O smoke não pode ser interpretado como resultado experimental.
- Antes das runs completas, ainda falta preparar a persistência dos resultados.

## Entregáveis

- README: pendente.
- Reprodução: configuração e código validados; runs definitivas pendentes.
- Vídeo: mapa e evidências de implementação parciais; resultados pendentes.

## Evidência

- `pytest -q --ignore=test/test_model.py`: 98 testes passaram.
- As três equações da q01 e o gráfico foram validados.
- Estresse `+/-1000`: losses 0/2000 e gradientes finitos.
- Split SHA-256 (forma, dtype e bytes dos índices):
  `118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0`.
- Pesos iniciais seed 0 SHA-256:
  `9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596`.

## Próximo passo

Preparar logs/checkpoints e executar `F-RELU` definitiva antes das variantes.
Não iniciar V2/V3.
