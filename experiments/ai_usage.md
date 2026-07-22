# Uso de IA

## 21/07/2026 — preparação da Variável 1

- Objetivo: criar os registros novos e formalizar configurações e hipóteses.
- Contribuição da IA: estrutura e texto inicial dos artefatos.
- Arquivos: `PROJECT_STATUS.md` e `experiments/`.
- Verificação: JSON válido, CSV com 32 colunas e `git diff --check` sem erros.
- Decisão do estudante: criação autorizada; revisão final do conteúdo ainda é
  responsabilidade do estudante.
- Proveniência: Passo 0 versionado com autorização; nenhum push realizado.

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
  autorizados. O smoke não foi validado como resultado científico e nenhum push
  foi feito.
