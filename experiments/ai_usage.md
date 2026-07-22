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
