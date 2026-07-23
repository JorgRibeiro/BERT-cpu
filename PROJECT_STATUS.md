# Estado do projeto

Atualizado em 23/07/2026.

## Agora

- Branch: `q01-ativacoes-adult`.
- `HEAD` e remoto local: `07243dc`, fechamento da V1.
- Fase: infraestrutura pré-experimental da V2 validada localmente.
- Nenhuma run científica da V2 foi executada.
- Nenhum commit ou push da V2 foi feito.

## V2 pronta para versionar

- Níveis: `beta=0,5`, `1`, `2` e `5`; seeds `0`, `1` e `2`.
- Hipótese e protocolo registrados antes dos smokes.
- Softplus-beta estável, backward, integração Adult e 5 FLOPs por elemento.
- Executor unitário, lote sequencial de 12 runs e análise reproduzível.
- 196 testes permitidos passaram; 73 deles focados na V2.
- Quatro smokes passaram sem `results.csv` e sem avaliação no teste oficial.
- Cada smoke confirmou `850.711.121` FLOPs por época e `94.632.384` na
  inferência da validação.

## Pendências

1. Receber autorização para o commit pré-runs.
2. Executar 12 runs científicas de 100 épocas.
3. Gerar tabela, gráficos e decisão de H2.
4. Encerrar V2 antes de iniciar V3.

## Limites

- Smoke não é resultado experimental.
- O split é fixo; três seeds medem variação de inicialização.
- O teste oficial continua reservado para a fase final.
- O risco acadêmico de V3 permanece registrado.

Detalhes da V1: `experiments/analysis.md`. Protocolo da V2:
`experiments/v2/protocol.md`.
