"""Caminhos padrão do projeto.

Todos os módulos usam estes caminhos como valor padrão, mas aceitam outro
caminho por argumento. Assim os testes rodam em pastas temporárias sem tocar
nos dados reais.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# A divisão treino/validação/teste fica versionada no Git (só IDs, sem texto),
# para que os três modelos sejam avaliados exatamente no mesmo conjunto.
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_FILE = SPLITS_DIR / "olist_splits_v1.csv"

PROCESSED_REVIEWS_FILE = PROCESSED_DIR / "reviews.parquet"

# Anotações guardam só IDs e rótulos (sem texto), por isso ficam no Git.
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
QUEUE_FILE = ANNOTATIONS_DIR / "fila_v1.csv"
LABELS_FILE = ANNOTATIONS_DIR / "rotulos_v1.csv"
AGREEMENT_QUEUE_FILE = ANNOTATIONS_DIR / "fila_concordancia_v1.csv"
AGREEMENT_LABELS_FILE = ANNOTATIONS_DIR / "rotulos_concordancia_v1.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
