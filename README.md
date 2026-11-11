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

### Resultados

Os números ficam em [`reports/resultados.md`](reports/resultados.md), acumulados por
tarefa e conjunto em `reports/resultados.json`: os três modelos (baseline, BERTimbau
e LLM) entram na mesma tabela, medidos pelas mesmas funções de `evaluate.py`.

Teste (medido uma vez por modelo):

| Tarefa | Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) |
|---|---|---:|---:|---:|
| Sentimento (binário, 3.712 aval.) | TF-IDF + Regressão Logística | 0,930 | 0,02 | 0,00 |
| Tipo de problema (multirrótulo, 500 aval.) | TF-IDF + Regressão Logística | 0,678 | 0,02 | 0,00 |

No tipo de problema, o baseline vai bem nas categorias de entrega (não recebido 0,86;
atraso 0,81; pedido incompleto 0,79) e mal nas que dependem de interpretação ou têm
poucos exemplos (outro 0,40; sem reclamação 0,38). A curva de aprendizado
([`reports/figures/curva_aprendizado.png`](reports/figures/curva_aprendizado.png))
ainda sobe com 750 exemplos: mais anotação ajudaria, mas o próximo ganho deve vir
de um modelo que entenda contexto.

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
| Latência dos modelos locais medida em CPU | É o hardware da demo; a latência em GPU do BERTimbau fica registrada à parte. |

## Estrutura

```
src/reviews/      código do projeto (dados, anotação, avaliação, baseline, BERTimbau)
docs/             guia de anotação
annotations/      fila e rótulos da anotação (só IDs, sem texto)
notebooks/        análise exploratória (01) e treino do BERTimbau no Colab (02)
splits/           IDs da divisão treino/validação/teste
reports/          resultados por modelo (`resultados.md`), figuras e relatórios
models/           modelos treinados (fora do Git)
tests/            testes automatizados
```
