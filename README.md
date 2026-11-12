# Analisador de Avaliações com IA

Classificação de avaliações de clientes de e-commerce brasileiro (base Olist) com TF-IDF, BERTimbau e LLM, e agente que gera relatórios para vendedores.

🚧 Em construção.

## Como reproduzir

Requer Python 3.12 e uma conta no Kaggle.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

kaggle auth login              # uma vez, abre o navegador
python -m reviews.download     # baixa a base para data/raw/
python -m reviews.prepare      # gera data/processed/reviews.parquet
pytest                         # roda os testes
```

Depois abra `notebooks/01_eda.ipynb` para a análise exploratória.

### Anotação dos tipos de problema

```bash
python -m reviews.annotation fila      # sorteia as 1.500 avaliações a anotar (uma vez)
python -m reviews.annotate_app         # abre o app de anotação no navegador
python -m reviews.annotation status    # mostra o progresso
```

As regras estão em [`docs/guia_anotacao.md`](docs/guia_anotacao.md).

### Baseline (Fase 3)

```bash
python -m reviews.baseline sentimento --teste   # TF-IDF + Regressão Logística
python -m reviews.baseline problemas --teste    # multirrótulo (8 categorias)
python -m reviews.baseline curva                # curva de aprendizado
```

Sem `--teste`, mede só na validação. O valor de C é escolhido na **validação**;
o teste é medido uma única vez, ao fechar a fase.

### BERTimbau (Fase 4)

O ajuste fino precisa de GPU e roda no Google Colab:
[`notebooks/02_bertimbau_colab.ipynb`](notebooks/02_bertimbau_colab.ipynb).
O notebook baixa este repositório, recebe a base processada, treina as duas tarefas,
publica os modelos no Hugging Face Hub e devolve os resultados para versionar aqui.

```bash
python -m reviews.bert sentimento --teste
python -m reviews.bert problemas --teste
python -m reviews.bert publicar --usuario SEU_USUARIO_HF   # token na variável HF_TOKEN
```

A validação escolhe a melhor época (parada antecipada); o teste é medido uma vez.

### LLM zero-shot (Fase 5)

O Claude Haiku 5.5 classifica as avaliações **sem treino nenhum**: recebe só a definição
da tarefa e, nos tipos de problema, as categorias e regras do guia de anotação, sem
nenhum exemplo rotulado. As avaliações vão em lotes de 25 por chamada, com saída em
JSON validada por esquema.

```bash
# uma vez: copie .env.example para .env e preencha ANTHROPIC_API_KEY (o .env não vai para o Git)
python -m reviews.llm verificar                         # testa a chave (custo ~US$ 0,0001)
python -m reviews.llm problemas --simular               # mostra o plano, sem chamar a API
python -m reviews.llm problemas --esforco low           # validação
python -m reviews.llm problemas --esforco medium        # validação, outro esforço
python -m reviews.llm problemas --esforco low --teste   # teste, com o esforço escolhido
```

- O **esforço de raciocínio** (`low`, `medium`, `high`) é escolhido na validação; toda
  execução fica registrada em `reports/llm_execucoes.csv`, inclusive as descartadas.
- Depois que o teste é medido, medi-lo de novo com outra configuração é bloqueado.
- Cada resposta da API vai para `data/llm_cache/` (fora do Git): se a execução cair,
  rodar de novo continua de onde parou, sem pagar outra vez.
- O custo é calculado a partir dos tokens que a API informa como cobrados, e há uma trava
  de gasto por execução (`--limite-usd`, padrão US$ 2).

### Resultados

Os números ficam em [`reports/resultados.md`](reports/resultados.md), acumulados por
tarefa e conjunto em `reports/resultados.json`: os três modelos (baseline, BERTimbau
e LLM) entram na mesma tabela, medidos pelas mesmas funções de `evaluate.py`.

Teste (medido uma vez por modelo):

| Tarefa | Modelo | F1 macro | F1 micro | Latência (ms/aval.) | Custo (US$/1.000) |
|---|---|---:|---:|---:|---:|
| Sentimento (binário, 3.712 aval.) | TF-IDF + Regressão Logística | 0,930 | – | 0,02 | 0 |
| | BERTimbau ajustado | **0,947** | – | 94 (CPU) | 0 |
| | Claude Haiku 5.5 zero-shot | 0,935 | – | 85 (API, lote de 25) | 0,014 |
| Tipo de problema (multirrótulo, 500 aval.) | TF-IDF + Regressão Logística | 0,678 | 0,756 | 0,02 | 0 |
| | BERTimbau ajustado | 0,696 | 0,827 | 125 (CPU) | 0 |
| | Claude Haiku 5.5 zero-shot | **0,881** | **0,902** | 353 (API, lote de 25) | 0,049 |

Modelos publicados no Hugging Face Hub:
[`gmarconems/bertimbau-olist-sentimento`](https://huggingface.co/gmarconems/bertimbau-olist-sentimento) e
[`gmarconems/bertimbau-olist-problemas`](https://huggingface.co/gmarconems/bertimbau-olist-problemas).

**Tipo de problema: o LLM sem treino vence com folga.** O Claude Haiku 5.5, só com as regras
do guia de anotação e nenhum exemplo, chega a F1 macro de 0,881 (intervalo de 95% por
bootstrap: 0,844 a 0,907), contra 0,696 do BERTimbau e 0,678 do baseline, e acerta o
conjunto exato de rótulos em 78% das avaliações (BERTimbau 67%, baseline 50%). É melhor
em todas as categorias, inclusive em **"sem reclamação"** (0,84), que o BERTimbau nunca
previu por ter só 14 exemplos no treino: o LLM só precisa da definição. "Outro problema"
segue a categoria mais difícil para todos (0,65), por ser a mais subjetiva. Com 750
avaliações de treino, regras explícitas valem mais que exemplos.

**Sentimento: o BERTimbau continua na frente.** O LLM (0,935) fica um pouco acima do
baseline (0,930) e abaixo do BERTimbau (0,947). Ele acerta 97% das negativas, mas marca
como negativas 178 avaliações que tinham nota 4 ou 5 ("Só chegou um item", "Embalagem
muito arranhada"): o texto reclama, a nota não. O rótulo aqui vem das estrelas, e esse
padrão (quem reclama mas dá nota alta) o BERTimbau aprende nos dados, o LLM não tem como
saber. Em compensação, o BERTimbau precisou de quase 30 mil exemplos rotulados; o LLM,
de nenhum.

**Custo e latência.** O LLM custou US$ 0,23 no total, somando as escolhas de esforço na
validação e o teste (menos de US$ 0,05 por mil avaliações). A latência por avaliação vem
de lotes de 25; uma chamada isolada leva cerca de 2 segundos (`reviews.llm verificar`). O TF-IDF responde em
microssegundos sem custo nenhum, e o BERTimbau roda em CPU sem depender de API.

**Escolha do esforço de raciocínio (validação).** Regra fixada antes de ver os números:
fica o maior F1, mas uma diferença menor que 0,01 conta como empate e vence o mais barato.

| Tarefa | low | medium | high | Escolhido |
|---|---:|---:|---:|---|
| Tipo de problema | 0,865 | 0,899 | 0,903 | medium |
| Sentimento | 0,936 | 0,935 | – | low |

**Cuidado na leitura.** Durante a anotação, cerca de 35 rótulos da **validação** foram
ajustados com ajuda de um assistente Claude aplicando o guia, o que pode favorecer um
pouco o Haiku na validação (0,899 lá, 0,881 no teste). No teste, os ajustes foram só
mecânicos, pelas regras escritas, por isso o número que vale é o do teste.

**Próximos passos registrados.** Dar peso maior às classes raras no BERTimbau (escolhido
na validação) e testar o LLM com alguns exemplos (few-shot) — ambos exigiriam uma nova
medição no teste, então ficam como variações declaradas, não como substitutos destes
resultados.

A curva de aprendizado do baseline
([`reports/figures/curva_aprendizado.png`](reports/figures/curva_aprendizado.png))
ainda sobe com 750 exemplos: mais anotação ajudaria os modelos treinados.

## Dados

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), publicado pela Olist no Kaggle sob a licença **CC BY-NC-SA 4.0** (uso não comercial). Os dados não ficam neste repositório: são baixados pelo script. Os modelos treinados com eles seguem a mesma licença.

## Decisões de dados

| Decisão | Motivo |
|---|---|
| Só avaliações com comentário | Sem texto não há o que classificar. |
| Sentimento pela nota: 1–2 negativa, 4–5 positiva, 3 neutra | A nota 3 mistura elogio e reclamação; fica fora da tarefa binária. |
| Métrica principal: F1 macro | As positivas dominam; acurácia premiaria chutar sempre "positiva". |
| Texto limpo de forma conservadora (mantém maiúsculas, acentos e pontuação) | O BERTimbau é *cased* e aproveita essas pistas. |
| Um registro por `review_id` | Na base original o mesmo `review_id` aparece em pedidos diferentes. |
| Divisão 80/10/10 estratificada pela nota e **agrupada por texto** | Textos repetidos ("Recomendo") ficam no mesmo conjunto; senão o teste mediria memorização. |
| Divisão versionada em `splits/olist_splits_v1.csv` | Os três modelos são avaliados exatamente no mesmo teste. |
| Tipo de problema **multirrótulo** (7 categorias + "sem reclamação") | Uma reclamação costuma ter mais de um problema ("atrasou e veio quebrado"); escolher um "principal" seria arbitrário. |
| Anotação feita a partir do **texto**, não dos dados de entrega | O modelo só vê o texto; os dados de entrega ficam para validar depois. |
| Teste 100% anotado à mão, sem pré-rótulo de LLM | Senão a comparação com o LLM zero-shot ficaria enviesada a favor dele. |
| Piloto de 100 avaliações tirado do treino | Ajustes no guia não contaminam o teste. |
| Concordância medida com kappa de Cohen numa reanotação de 100 avaliações | Mostra quais categorias têm regras claras e quais são ambíguas. |
| Hiperparâmetros escolhidos na validação; teste medido uma vez por modelo | O teste nunca orienta ajustes, senão a comparação final ficaria viciada. |
| Multirrótulo: limiar 0,5; sem rótulo acima dele, fica o mais provável; "sem reclamação" nunca acompanha outro | Segue as regras do guia de anotação; toda avaliação tem ao menos um rótulo. |
| BERTimbau com hiperparâmetros fixos; a validação só escolhe a melhor época | Sem busca de hiperparâmetros, a validação não vira um segundo treino. |
| LLM zero-shot com as regras do guia, mas sem nenhum exemplo rotulado | É o mesmo critério que o anotador seguiu; exemplos tornariam o teste "few-shot". |
| LLM classifica em lotes de 25 com saída em JSON validada | 25 vezes menos chamadas, e as instruções são pagas uma vez por lote; resposta fora do formato é refeita, e o que falhar fica contado. |
| Latência dos modelos locais medida em CPU | É o hardware da demo; a latência em GPU do BERTimbau fica registrada à parte. |

## Estrutura

```
src/reviews/      código do projeto (dados, anotação, avaliação, baseline, BERTimbau, LLM)
docs/             guia de anotação
annotations/      fila e rótulos da anotação (só IDs, sem texto)
notebooks/        análise exploratória (01) e treino do BERTimbau no Colab (02)
splits/           IDs da divisão treino/validação/teste
reports/          resultados por modelo (`resultados.md`), figuras e relatórios
models/           modelos treinados (fora do Git)
tests/            testes automatizados
```
