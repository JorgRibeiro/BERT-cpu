# Estado do projeto

Atualizado em 24/07/2026.

## Agora

- Branch: `q01-ativacoes-adult`.
- Base pré-runs: `26e4473`.
- Fase: execução, análise e versionamento da V2 concluídos.
- Nenhum push foi executado pelo agente.

## Evidência da V2

- Validação média: beta 0,5 = 84,7461%; beta 1 = 84,8843%;
  beta 2 = 85,0379%; beta 5 = 85,1966%.
- Cada uma das 12 runs: `85,0711121` GFLOPs instrumentados.
- Todas têm 100 épocas, seis diagnósticos e checkpoints validados.
- As 12 compartilham commit, configuração, dados e split.
- H2: inconclusiva; beta 2 menos beta 5 = -0,1587 p.p., abaixo de 0,5 p.p.
- Tabela, análise e três gráficos foram gerados e inspecionados.
- Os 12 artefatos foram revalidados; 196 testes permitidos passaram.
- O teste oficial não foi avaliado.

## Pendências

1. Revisar as evidências da V2 com o estudante.
2. Aguardar nova solicitação antes de iniciar V3 ou avaliar o teste oficial.

## Limites

- Smoke não é resultado experimental.
- O split é fixo; três seeds medem variação de inicialização.
- O teste oficial continua reservado para a fase final.
- O risco acadêmico de V3 permanece registrado.

Detalhes: `experiments/analysis.md` e `experiments/v2/analysis.md`.
