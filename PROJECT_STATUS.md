# Estado do projeto

Atualizado em 24/07/2026.

## Agora

- Branch: `q01-ativacoes-adult`.
- Infraestrutura da V3: commit `2c15768`.
- Fase: V3 encerrada e versionada com autorização.
- Nenhum push foi executado pelo agente.

## Evidência da V3

- Nove runs válidas: L1, L2 e L3 nas seeds 0, 1 e 2.
- Validação média: 84,5414%, 84,6744% e 84,6796%.
- Custo: 2,6107501, 84,0291601 e 154,4654481 GFLOPs/run.
- H3a não contradita; H3b e H3c sustentadas; H3d inconclusiva.
- Tabela, análise e três gráficos estão em `experiments/v3/`.
- Checkpoints, hashes, equivalência afim e ausência de Identity foram validados.
- As nove runs passaram no dry-run estrito e 269 testes permitidos passaram.
- O teste oficial não foi avaliado.

## Pendências

1. Confirmar o início da avaliação final no teste.
2. Fazer a análise conjunta.
3. Preparar a entrega.

## Limites

- Smoke não é resultado experimental.
- O split é fixo; três seeds medem variação de inicialização.
- A ponte ReLU–L2 atravessa commits e kernels diferentes.
- O teste oficial continua reservado para a fase final.
- O risco acadêmico de V3 permanece registrado.

Detalhes: `experiments/v3/protocol.md` e `experiments/v3/analysis.md`.
