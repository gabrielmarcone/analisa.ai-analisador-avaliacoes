"""Regras de rótulo.

Sentimento vem da nota (review_score):
    1-2 -> negativa
    3   -> neutra   (fica na base, mas os modelos de sentimento a descartam)
    4-5 -> positiva

A nota 3 é ambígua: muitos textos de nota 3 elogiam o produto e reclamam da
entrega. Tratá-la como classe à parte evita ensinar ruído ao modelo binário.
"""

from __future__ import annotations

import pandas as pd

NEGATIVE = "negativa"
NEUTRAL = "neutra"
POSITIVE = "positiva"

SCORE_TO_SENTIMENT: dict[int, str] = {
    1: NEGATIVE,
    2: NEGATIVE,
    3: NEUTRAL,
    4: POSITIVE,
    5: POSITIVE,
}

BINARY_SENTIMENTS = (NEGATIVE, POSITIVE)


def sentiment_from_score(scores: pd.Series) -> pd.Series:
    """Converte notas 1-5 em sentimento. Levanta erro se houver nota inválida."""
    numeric = pd.to_numeric(scores, errors="coerce")
    invalid = ~numeric.isin(list(SCORE_TO_SENTIMENT))
    if invalid.any():
        examples = scores[invalid].unique()[:5].tolist()
        raise ValueError(f"Notas fora de 1-5 encontradas: {examples}")
    return numeric.astype(int).map(SCORE_TO_SENTIMENT).astype("string")
