# Estado do projeto

Atualizado em 24/07/2026.

## Agora

- Branch: `q01-ativacoes-adult`.
- Base da V3: fechamento V2 `de000de`.
- Fase: infraestrutura e smokes da V3 concluídos localmente.
- Nenhum push foi executado pelo agente.

## Evidência da V3

- Três modelos afins: 218, 7.106 e 11.266 parâmetros.
- Sem operação Identity; cada composição equivale a `W*x+b`.
- Três smokes de duas épocas passaram com custos exatos.
- Checkpoints, hashes, equivalência afim e inferência foram validados.
- 269 testes permitidos passaram.
- O dry-run planeja nove runs na ordem L1, L2 e L3, sempre seeds 0, 1 e 2.
- Smoke não entrou em `results.csv`.
- O teste oficial não foi avaliado.

## Pendências

1. Autorizar e realizar o commit pré-runs da V3.
2. Executar as nove runs científicas.
3. Gerar tabela, gráficos e avaliar H3a–H3d.

## Limites

- Smoke não é resultado experimental.
- O split é fixo; três seeds medem variação de inicialização.
- O teste oficial continua reservado para a fase final.
- O risco acadêmico de V3 permanece registrado.

Detalhes: `experiments/v3/protocol.md` e `experiments/hypotheses.md`.
