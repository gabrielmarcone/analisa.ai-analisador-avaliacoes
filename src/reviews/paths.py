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
