"""Divisão fixa treino/validação/teste.

Regras:
- Estratificada pela nota (1-5), para manter a proporção de cada nota.
- Agrupada pelo texto normalizado: textos iguais ("Recomendo", "Ótimo produto")
  ficam sempre no mesmo conjunto. Sem isso o teste mediria memorização.
- Gerada uma vez e salva em `splits/olist_splits_v1.csv` (versionado no Git).
  Depois disso é sempre lida do arquivo, nunca recalculada sem querer.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from reviews.paths import SPLITS_FILE

TRAIN, VAL, TEST = "treino", "validacao", "teste"
SPLIT_NAMES = (TRAIN, VAL, TEST)
SEED = 42


def _first_fold(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held_out = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return held_out


def make_splits(
    df: pd.DataFrame,
    label_col: str = "review_score",
    group_col: str = "texto_grupo",
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
    seed: int = SEED,
) -> pd.DataFrame:
    """Retorna um DataFrame (review_id, split) com ~80/10/10."""
    if not 0 < test_fraction < 0.5 or not 0 < val_fraction < 0.5:
        raise ValueError("As frações de teste e validação devem estar entre 0 e 0,5.")
    if df["review_id"].duplicated().any():
        raise ValueError("review_id repetido: deduplique antes de dividir.")

    labels = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    n_folds = round(1 / test_fraction)
    if pd.Series(groups).nunique() < 2 * n_folds:
        raise ValueError(
            f"Poucos textos distintos ({pd.Series(groups).nunique()}) para dividir. "
            "A base foi carregada por inteiro?"
        )
    split = np.full(len(df), TRAIN, dtype=object)

    test_idx = _first_fold(labels, groups, n_folds, seed)
    split[test_idx] = TEST

    rest = np.flatnonzero(split == TRAIN)
    val_share = val_fraction / (1 - test_fraction)
    val_local = _first_fold(labels[rest], groups[rest], round(1 / val_share), seed)
    split[rest[val_local]] = VAL

    return pd.DataFrame({"review_id": df["review_id"].to_numpy(), "split": split}).astype(
        {"review_id": "string", "split": "string"}
    )


def save_splits(splits: pd.DataFrame, path: Path = SPLITS_FILE) -> None:
    """Salva de forma atômica: escreve num arquivo temporário e só então substitui."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    splits.sort_values("review_id").to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def load_splits(path: Path = SPLITS_FILE) -> pd.DataFrame:
    splits = pd.read_csv(path, dtype={"review_id": "string", "split": "string"})
    if set(splits.columns) != {"review_id", "split"}:
        raise ValueError(f"{path}: colunas inesperadas {list(splits.columns)}")
    unknown = set(splits["split"].unique()) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"{path}: valores de split desconhecidos {unknown}")
    if splits["review_id"].duplicated().any():
        raise ValueError(f"{path}: review_id repetido.")
    return splits


def attach_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Junta a coluna `split` e exige que toda avaliação tenha um split."""
    merged = df.merge(splits, on="review_id", how="left", validate="one_to_one")
    missing = merged["split"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} avaliações sem split. A base mudou? "
            "Gere uma nova versão com `python -m reviews.prepare --new-splits`."
        )
    return merged


def check_group_leakage(df: pd.DataFrame, group_col: str = "texto_grupo") -> int:
    """Conta textos normalizados que aparecem em mais de um split (deve ser 0)."""
    per_group = df.groupby(group_col)["split"].nunique()
    return int((per_group > 1).sum())
