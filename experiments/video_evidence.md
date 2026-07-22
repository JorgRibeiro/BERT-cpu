# Evidências para o vídeo

## Highlights — preparação técnica da V1

Antes da investigação de 100 épocas, foi necessário:

- implementar Sigmoid, Swish e Softplus estáveis, incluindo seus backwards;
- usar uma única MLP configurável, mudando somente a ativação oculta;
- garantir o mesmo split e os mesmos pesos iniciais entre ativações;
- estabilizar a cross-entropy para impedir NaN/Inf em logits extremos;
- instrumentar custos diferentes de FLOPs para cada ativação;
- impedir acesso automático ao teste oficial;
- validar tudo com 98 testes, gráfico, smoke e repetição exata da ReLU.

Mensagem para a apresentação: esta etapa não escolheu a melhor ativação. Ela
removeu fontes de erro e garantiu uma comparação controlada para a V1.

| Item | Estado | Evidência atual |
|---:|---|---|
| 1. Uso de IA | Parcial | `experiments/ai_usage.md` |
| 2. Tarefa Adult | Parcial | `AGENTS.md` e configuração de V1 |
| 3. Baseline | Parcial | `F-RELU` configurada e validada em smoke; run definitiva pendente |
| 4. Variáveis e controles | Parcial | `AGENTS.md` e `experiments/hypotheses.md` |
| 5. Formulação | Parcial | `exercises/q01_activations.py`, testes e gráfico da q01 |
| 6. Arquitetura e treino | Parcial | `exercises/task_binary_classification.py` e testes de integração |
| 7. Protocolo | Parcial | `AGENTS.md` e configuração de V1 |
| 8. Resultados | Pendente | nenhuma run experimental de 100 épocas executada |
| 9. Desempenho/FLOPs | Pendente | nenhuma run executada |
| 10. Hipóteses/resultados | Parcial | hipóteses registradas |
| 11. Dificuldades | Parcial | estabilidade da loss e correção vetor `@` vetor em `experiments/ai_usage.md` |
| 12. Reprodução | Parcial | configuração, hashes, comandos, smoke e commit da implementação de V1 |
