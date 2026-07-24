# Roteiro do vídeo — Adult + q01

Duração planejada: **19min20s**. Fale pelos tópicos; não leia os arquivos
inteiros.

## 0:00–0:40 — abertura

Mostrar: `README.md`, título e gráfico conjunto.

Falar:

- “Investiguei se a ativação oculta pode melhorar uma classificação Adult.”
- “Comparei 11 configurações, três seeds e 33 treinamentos em NumPy/CPU.”
- “O resultado não foi um vencedor universal: beta 5 teve a melhor validação,
  L1 o melhor retorno por FLOP e ReLU venceu sob o orçamento da baseline.”

## 0:40–1:35 — uso de IA

Mostrar: `experiments/ai_usage.md`.

Falar:

- A IA ajudou no protocolo, código, testes, executores e documentação.
- Eu decidi trocar q04 por q01, escolhi as variáveis, confirmei três seeds,
  cancelei o extra de 1000 épocas e autorizei cada commit.
- Eu verifiquei a IA pedindo explicações etapa a etapa, exigindo as três seeds
  e as losses, conferindo resultados antes de avançar e rejeitando o extra que
  fugia do protocolo original.
- Eu ainda preciso revisar este roteiro e assumir a explicação final.

## 1:35–2:45 — tarefa e dados

Mostrar: `exercises/task_binary_classification.py` e a seção “Tarefa” do README.

Falar:

- Adult prevê renda `<=50K` ou `>50K`.
- Os 14 atributos viram 108 features; a saída são dois logits.
- O treino oficial tem 32.561 amostras: 26.049 treino e 6.512 validação.
- A métrica principal é a acurácia de validação na época 100.
- O objetivo de treinamento é minimizar a cross-entropy.
- Limitação: o encoder é ajustado antes do hold-out, usando features da futura
  validação, mas não seus rótulos.

## 2:45–3:35 — baseline

Mostrar: `AdultMLP` e `experiments/configs/v1_activation_family.json`.

Falar:

- Baseline: `108 -> 64 -> ReLU -> 2`, com 7.106 parâmetros.
- ReLU foi escolhida porque já era o comportamento padrão da tarefa original.
- ReLU faz `max(0,z)`: é barata, mas zera valores e gradientes negativos.
- Ela foi reproduzida antes das variantes, inclusive com uma repetição
  determinística da seed 0.

## 3:35–5:10 — variáveis e controles

Mostrar: tabelas de variáveis no README.

Falar:

- V1 muda apenas a família: ReLU, Sigmoid, Swish e Softplus.
- V2 mantém a arquitetura e muda `beta` da Softplus: 0,5, 1, 2 e 5.
- V3 remove a ativação e compara uma, duas e três camadas lineares.
- Controles: full-batch, Adam, `lr=0,01`, 100 épocas, split seed 0 e seeds de
  modelo 0, 1 e 2.
- Ganho relevante: 0,5 ponto percentual e mesmo sinal em duas de três seeds.
- O teste ficou reservado até hipóteses, runs e seleção estarem congeladas.
- V3 pode ser interpretada como variável arquitetural; o professor não
  confirmou seu enquadramento em q01.

## 5:10–7:25 — formulações e efeito no processamento

Mostrar: `exercises/q01_activations.py` e
`test/test_q01_activations.py`.

Falar:

- Sigmoid: `1/(1+exp(-z))`; pode saturar e reduzir o gradiente.
- Swish: `z*sigmoid(z)`; é suave e preserva pequenas saídas negativas.
- Softplus: `log(1+exp(z))`; aproxima ReLU de forma suave.
- Softplus-beta: `log(1+exp(beta*z))/beta`; beta maior marca a transição.
- As formas estáveis evitam overflow, e os backwards foram comparados por
  diferenças finitas.
- O fluxo é `x -> Linear -> ativação -> Linear -> logits -> cross-entropy`;
  depois, `backward` calcula os gradientes e Adam atualiza os pesos.
- A cross-entropy original falhava em logits extremos; log-sum-exp tornou loss
  e gradientes finitos para todas as configurações.
- Os FLOPs são instrumentados: servem para a comparação deste protocolo, não
  medem tempo, energia, memória ou custo completo.

## 7:25–8:25 — desenho experimental

Mostrar: `experiments/hypotheses.md`, um log e um checkpoint.

Falar:

- Cada run salva configuração, hashes, histórico, diagnósticos e pesos.
- A regra e os limites da janela de FLOPs são iguais entre configurações; as
  contagens mudam conforme as operações e arquiteturas.
- Smokes de duas épocas validaram o caminho, mas não testaram hipóteses.
- Média e desvio usam só três inicializações no mesmo split; o desvio não é
  intervalo de confiança.

## 8:25–9:35 — resultado da V1

Mostrar: `experiments/plots/v1_accuracy_vs_flops.png` e
`experiments/analysis.md`.

Falar:

- Swish: 85,1351%; ReLU: 85,0481%; Softplus: 84,8843%; Sigmoid: 84,7461%.
- Nenhuma suave superou ReLU por 0,5 p.p.; H1a ficou inconclusiva.
- Sigmoid não ficou 0,5 p.p. abaixo; H1b também ficou inconclusiva.
- ReLU teve o menor custo e melhor retorno da V1; H1c foi sustentada.

## 9:35–10:35 — resultado da V2

Mostrar: `experiments/v2/plots/validation_vs_beta.png` e
`experiments/v2/analysis.md`.

Falar:

- As médias cresceram de 84,7461% em beta 0,5 até 85,1966% em beta 5.
- Todos os betas custaram 85,0711121 GFLOPs instrumentados por run.
- A hipótese favorecia o melhor beta central contra o melhor extremo.
- Beta 2 perdeu para beta 5 por 0,1587 p.p.; H2 ficou inconclusiva.
- Beta 1 e Softplus da V1 deram métricas e pesos iguais, mas a convenção de
  instrumentação da V2 tem custo maior.

## 10:35–11:50 — resultado da V3

Mostrar: `experiments/v3/plots/accuracy_vs_flops.png` e
`experiments/v3/analysis.md`.

Falar:

- Sem ativação, a composição continua sendo uma função afim `W*x+b`.
- L1, L2 e L3 tiveram 84,5414%, 84,6744% e 84,6796%.
- O ganho máximo foi só 0,1382 p.p., enquanto o custo foi de 2,6107501 para
  154,4654481 GFLOPs.
- H3a não foi contradita; crescimento de custo e ordem do retorno sustentaram
  H3b e H3c.
- ReLU venceu L2 nas três seeds, mas por 0,3737 p.p.; H3d ficou inconclusiva.

## 11:50–12:40 — avaliação oficial

Mostrar: `experiments/final_evaluation/results.csv` e manifesto.

Falar:

- Só depois do congelamento, 33 checkpoints primários foram avaliados.
- No avaliador oficial houve um load controlado do teste, 33 forwards e nenhum
  treinamento.
- Beta 2 liderou o teste com 85,6458%, mas beta 5 liderava a validação.
- O teste é descritivo: não mudou seleção, hipóteses ou novas runs.
- A tabela conjunta mostrada em seguida preserva as médias das 11 configurações,
  não apenas as quatro maiores.

## 12:40–15:30 — análise conjunta

Mostrar, nesta ordem:

1. `experiments/final_analysis/plots/accuracy_vs_training_flops.png`;
2. `experiments/final_analysis/plots/return_per_flop.png`;
3. `experiments/final_analysis/plots/validation_vs_test.png`;
4. `experiments/final_analysis/analysis.md`.

Falar:

- Pareto global: L1, L2, ReLU e beta 5.
- Melhor retorno: L1, com 3,454657 p.p./GFLOP.
- Sob 84,2375505 GFLOPs, a melhor validação é ReLU, com 85,0481%.
- V3 é o caso de muito custo e pouco ganho: +151,854698 GFLOPs para +0,138206
  p.p. de L1 a L3.
- V2 mudou 0,450450 p.p. sem mudar o custo entre betas, ainda abaixo do limiar.
- Já há ganho irrelevante em L1 para L2. A Pareto global não apresenta um único
  cotovelo monotônico.
- Conclusão: “melhor” depende do objetivo — acurácia, orçamento ou eficiência.

## 15:30–16:55 — dificuldades e limitações

Mostrar: `experiments/ai_usage.md` e limitações do README.

Falar:

- Foi necessário estabilizar ativações e cross-entropy.
- Um caso de vetor `@` vetor exigiu correção durante a preparação técnica da V1.
- Três seeds e um split não permitem alegar significância estatística.
- A ponte ReLU–L2 atravessa commits e kernels diferentes.
- A suíte ampla de desenvolvimento, executada depois da avaliação oficial,
  chamou o loader do teste em uma verificação de schema e fez treinos curtos.
  Isso não avaliou checkpoints, não gerou métricas e não alterou conclusões.
- Por isso ela não é apresentada como rota segura de reprodução.

## 16:55–18:25 — reprodução

Mostrar: seção “Reprodução segura” do README e executar:

```bash
python -m experiments.evaluate_official_test --verify-only
pytest -q test/test_plot_joint.py
python -m experiments.plot_joint
python -m experiments.evaluate_official_test --verify-only
```

Falar:

- `--verify-only` confere manifestos, CSV, log e hashes sem carregar o teste.
- Os seis testes bloqueiam loader, treino, forward e reavaliação na análise.
- O gerador reconstrói tabela e gráficos a partir dos pares salvos.
- As duas verificações confirmam que a avaliação oficial permaneceu íntegra.

## 18:25–19:20 — conclusão

Mostrar: resultado em uma frase no README.

Falar:

- Ativações mudaram custo, representação e resultado, mas nenhuma suave venceu
  ReLU pelo limiar pré-definido.
- Beta 5 teve a maior validação; L1 foi muito mais eficiente; ReLU foi a escolha
  sob o orçamento da baseline.
- Resultados inconclusivos foram preservados, sem escolher hipótese depois dos
  dados.

## Não afirmar no vídeo

- que existe significância estatística ou um vencedor universal;
- que FLOPs instrumentados equivalem a tempo ou custo físico;
- que o teste foi usado para escolher configuração;
- que a suíte completa passa ou que a suíte ampla não acessa o teste;
- que V3 foi aprovada pelo professor;
- que mais camadas nunca ajudam fora desta rede linear e deste protocolo;
- que houve revisão humana final antes de você realmente revisar.
