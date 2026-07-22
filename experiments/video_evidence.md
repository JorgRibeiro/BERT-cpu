# Evidências para o vídeo

## Highlights — preparação técnica da V1

Antes da investigação de 100 épocas, foi necessário:

- implementar Sigmoid, Swish e Softplus estáveis, incluindo seus backwards;
- usar uma única MLP configurável, mudando somente a ativação oculta;
- garantir o mesmo split e os mesmos pesos iniciais entre ativações;
- estabilizar a cross-entropy para impedir NaN/Inf em logits extremos;
- instrumentar custos diferentes de FLOPs para cada ativação;
- impedir acesso automático ao teste oficial;
- validar tudo com 123 testes, gráficos, smoke e repetição exata da ReLU;
- preparar logs, checkpoints e hashes antes da primeira run científica.

Mensagem para a apresentação: esta etapa não escolheu a melhor ativação. Ela
removeu fontes de erro e garantiu uma comparação controlada para a V1.

## Highlights — resultado da V1

- Swish teve a maior validação média: 85,1351%.
- ReLU teve o melhor retorno por FLOP e o menor custo.
- H1a e H1b foram inconclusivas; H1c foi sustentada.
- ReLU e Swish ficaram na Pareto; Sigmoid e Softplus foram dominadas.
- As curvas, seeds, dispersão e Pareto são regeneráveis por
  `python -m experiments.plot_v1`.

| Item | Estado | Evidência atual |
|---:|---|---|
| 1. Uso de IA | Parcial | `experiments/ai_usage.md` |
| 2. Tarefa Adult | Parcial | `AGENTS.md` e configuração de V1 |
| 3. Baseline | Concluído | ReLU: três seeds e repetição exata da seed 0; teste reservado |
| 4. Variáveis e controles | Parcial | `AGENTS.md` e `experiments/hypotheses.md` |
| 5. Formulação | Parcial | `exercises/q01_activations.py`, testes e gráfico da q01 |
| 6. Arquitetura e treino | Parcial | `exercises/task_binary_classification.py` e testes de integração |
| 7. Protocolo | Parcial | `AGENTS.md`, configuração e `experiments/run_v1.py` |
| 8. Resultados | Parcial | V1: tabela, curvas e métricas finais de 13 runs |
| 9. Desempenho/FLOPs | Parcial | `v1_accuracy_vs_flops.png`: ReLU/Swish na Pareto |
| 10. Hipóteses/resultados | Parcial | `analysis.md`: H1a/H1b inconclusivas; H1c sustentada |
| 11. Dificuldades | Parcial | estabilidade da loss e correção vetor `@` vetor em `experiments/ai_usage.md` |
| 12. Reprodução | Parcial | `plot_v1.py`, `v1_summary.csv`, 123 testes e fechamento da V1 versionado |
