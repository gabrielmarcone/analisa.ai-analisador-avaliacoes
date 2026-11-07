"""Download da base Brazilian E-Commerce (Olist) pelo Kaggle CLI.

Uso:
    python -m reviews.download            # baixa só se faltar algum arquivo
    python -m reviews.download --force    # baixa de novo

Requer o Kaggle CLI autenticado (`kaggle auth login`). As credenciais ficam
fora do projeto e nunca passam por este código.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from reviews.paths import RAW_DIR

DATASET = "olistbr/brazilian-ecommerce"

# Arquivos que o projeto usa. Os outros arquivos da base também são baixados,
# mas não são obrigatórios.
REQUIRED_FILES = (
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
)


def missing_files(raw_dir: Path = RAW_DIR) -> list[str]:
    """Lista os arquivos obrigatórios que ainda não existem (ou estão vazios)."""
    raw_dir = Path(raw_dir)
    return [
        name
        for name in REQUIRED_FILES
        if not (raw_dir / name).is_file() or (raw_dir / name).stat().st_size == 0
    ]


def build_command(kaggle_exe: str, raw_dir: Path, force: bool = False) -> list[str]:
    """Monta o comando como lista (sem shell), o que evita injeção de comandos."""
    cmd = [kaggle_exe, "datasets", "download", DATASET, "-p", str(raw_dir), "--unzip"]
    if force:
        cmd.append("--force")
    return cmd


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Baixa e descompacta a base em `raw_dir`, e confere se os arquivos chegaram."""
    raw_dir = Path(raw_dir)

    if not force and not missing_files(raw_dir):
        print(f"Base já está em {raw_dir}. Use --force para baixar de novo.")
        return

    kaggle_exe = shutil.which("kaggle")
    if kaggle_exe is None:
        raise SystemExit(
            "Kaggle CLI não encontrado. Ative a .venv e rode `pip install -r requirements.txt`."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(build_command(kaggle_exe, raw_dir, force), check=False)
    if result.returncode != 0:
        raise SystemExit(
            "O download falhou. Confira se você está autenticado com `kaggle auth login`."
        )

    still_missing = missing_files(raw_dir)
    if still_missing:
        raise SystemExit(f"Download incompleto. Arquivos faltando: {', '.join(still_missing)}")

    print(f"Base baixada em {raw_dir}.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baixa a base Olist do Kaggle.")
    parser.add_argument("--force", action="store_true", help="baixa mesmo se já existir")
    args = parser.parse_args(argv)
    download(force=args.force)


if __name__ == "__main__":
    main()
