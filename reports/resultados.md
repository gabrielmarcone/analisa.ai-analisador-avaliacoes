# Resultados por modelo

Gerado automaticamente a partir de `reports/resultados.json`;
para atualizar, rode as avaliações de cada modelo.

## problemas · validacao

| Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) | n |
|---|---:|---:|---:|---:|
| Claude Haiku 5.5 (zero-shot) | 0,899 | 401,75 | 0,06 | 250 |
| TF-IDF + Regressão Logística | 0,712 | 0,02 | 0,00 | 250 |
| BERTimbau ajustado | 0,677 | 124,50 | 0,00 | 250 |

## problemas · teste

| Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) | n |
|---|---:|---:|---:|---:|
| Claude Haiku 5.5 (zero-shot) | 0,881 | 352,86 | 0,05 | 500 |
| BERTimbau ajustado | 0,696 | 124,50 | 0,00 | 500 |
| TF-IDF + Regressão Logística | 0,678 | 0,02 | 0,00 | 500 |

## sentimento · validacao

| Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) | n |
|---|---:|---:|---:|---:|
| BERTimbau ajustado | 0,948 | 94,20 | 0,00 | 3712 |
| Claude Haiku 5.5 (zero-shot) | 0,936 | 81,38 | 0,01 | 3712 |
| TF-IDF + Regressão Logística | 0,931 | 0,02 | 0,00 | 3712 |

## sentimento · teste

| Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) | n |
|---|---:|---:|---:|---:|
| BERTimbau ajustado | 0,947 | 94,20 | 0,00 | 3712 |
| Claude Haiku 5.5 (zero-shot) | 0,935 | 84,73 | 0,01 | 3712 |
| TF-IDF + Regressão Logística | 0,930 | 0,02 | 0,00 | 3712 |
