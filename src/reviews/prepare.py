"""Gera data/processed/reviews.parquet a partir dos CSVs da Olist.

Uso:
    python -m reviews.prepare               # usa o split versionado (ou cria na 1ª vez)
    python -m reviews.prepare --new-splits  # recria o split (só se a base mudou)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from reviews.data import load_orders, load_reviews, prepare_reviews
from reviews.paths import PROCESSED_REVIEWS_FILE, RAW_DIR, SPLITS_FILE
from reviews.splits import (
    SPLIT_NAMES,
    attach_splits,
    check_group_leakage,
    load_splits,
    make_splits,
    save_splits,
)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Escreve num temporário e substitui, para nunca deixar um arquivo pela metade."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run(
    raw_dir: Path = RAW_DIR,
    output: Path = PROCESSED_REVIEWS_FILE,
    splits_file: Path = SPLITS_FILE,
    new_splits: bool = False,
) -> pd.DataFrame:
    reviews = load_reviews(raw_dir)
    orders = load_orders(raw_dir)
    df = prepare_reviews(reviews, orders)

    if new_splits or not Path(splits_file).is_file():
        splits = make_splits(df)
        save_splits(splits, splits_file)
        print(f"Split criado em {splits_file}. Faça commit desse arquivo.")
    else:
        splits = load_splits(splits_file)

    df = attach_splits(df, splits)
    leaks = check_group_leakage(df)
    if leaks:
        raise ValueError(f"{leaks} textos repetidos aparecem em mais de um split.")

    write_parquet_atomic(df, Path(output))
    _print_summary(reviews["review_id"].nunique(), df, Path(output))
    return df


def _print_summary(n_raw: int, df: pd.DataFrame, output: Path) -> None:
    print(f"Avaliações na base (únicas):   {n_raw:>7,}".replace(",", "."))
    print(f"Avaliações com comentário:     {len(df):>7,}".replace(",", "."))
    print("\nPor split e sentimento:")
    table = pd.crosstab(df["split"], df["sentimento"], margins=True, margins_name="total")
    print(table.reindex([*SPLIT_NAMES, "total"]).to_string())
    print(f"\nArquivo salvo em {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepara a base de avaliações.")
    parser.add_argument(
        "--new-splits", action="store_true", help="recria a divisão treino/validação/teste"
    )
    args = parser.parse_args(argv)
    run(new_splits=args.new_splits)


if __name__ == "__main__":
    main()
