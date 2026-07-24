# Estado do projeto

Atualizado em 24/07/2026.

## Agora

- Branch: `q01-ativacoes-adult`.
- Fase: fechamento experimental e materiais de entrega versionados.
- Nenhum push foi executado pelo agente.

## Evidência atual

- ID: `OFFICIAL-185889b9b944304ba514`.
- 33 checkpoints primários, 11 configurações e três seeds.
- No avaliador oficial: um load controlado do teste, um forward por checkpoint
  e nenhum treinamento.
- O gerador da análise conjunta não carrega o teste nem treina.
- Pareto: L1, L2, ReLU e beta 5.
- Melhor retorno: L1; sob orçamento da ReLU: ReLU.
- Melhor validação: beta 5; melhor teste descritivo: beta 2.
- Rota segura validada com seis testes focados e verificação dos artefatos.
- Após o commit, avaliador e análise conjunta somaram 17 testes focados
  aprovados.
- A suíte ampla alcançou 286 testes, mas acessa o loader do teste para conferir
  schema e executa treinos curtos; não gerou métricas oficiais nem mudou decisões.

## Pendências

1. O estudante revisar o README e o roteiro.
2. Gravar o vídeo e inserir seu link.
3. Preparar o envio no Classroom e autorizar eventual push separadamente.

## Limites

- Smoke não é resultado experimental.
- O split é fixo; três seeds medem variação de inicialização.
- A ponte ReLU–L2 atravessa commits e kernels diferentes.
- O teste não pode ser usado para alterar hipóteses ou selecionar novas runs.
- O risco acadêmico de V3 permanece registrado.

Detalhes: `experiments/final_evaluation/` e `experiments/final_analysis/`.
