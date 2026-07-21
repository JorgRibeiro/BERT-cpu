# Estado do projeto

Atualizado em 21/07/2026.

## Agora

- Objetivo: comparar funções de ativação na classificação Adult.
- Fase: preparação da Variável 1.
- Branch: `q01-ativacoes-adult`.
- Commit base: `14e65c20c2312e66ba76e54431b396f60ce65e10`.
- Passo 0: versionado no commit `docs: establish q01 experiment step 0`.
- Estado esperado após este registro: árvore limpa e nenhum push realizado.

## Concluído

- Escopo Adult + q01 definido.
- Protocolo pré-experimental confirmado.
- Artefatos novos inaugurados.
- Configurações e hipóteses de V1 registradas antes dos resultados.

## Ainda não feito

- q01 não implementada.
- Nenhum smoke test ou treinamento executado.
- Nenhuma métrica ou conclusão produzida.

## Riscos

- V3 pode ser considerada arquitetural, não uma terceira variável de q01.
- q01, FLOPs e estabilidade da cross-entropy ainda precisam ser validados.

## Entregáveis

- README: pendente.
- Reprodução: configuração inicial pronta; código e runs pendentes.
- Vídeo: mapa criado; evidências de implementação e resultados pendentes.

## Evidência

- Arquivos: `experiments/`, `PROJECT_STATUS.md`, `AGENTS.md` e
  `Passo-a-passo.md`.
- Validação: configuração JSON válida, CSV com 32 colunas e
  `git diff --check` sem erros.

## Próximo passo

Implementar e testar Sigmoid, Swish e Softplus. Não iniciar V2 ou V3.
