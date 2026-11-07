"""Carga, validação e limpeza da base Olist.

O texto é limpo de forma conservadora: espaços e quebras de linha são
normalizados, mas maiúsculas, acentos e pontuação são mantidos, porque o
BERTimbau usado nas próximas fases é "cased" e aproveita essas pistas.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from reviews.labels import sentiment_from_score
from reviews.paths import RAW_DIR

REVIEWS_FILE = "olist_order_reviews_dataset.csv"
ORDERS_FILE = "olist_orders_dataset.csv"
ORDER_ITEMS_FILE = "olist_order_items_dataset.csv"

REVIEW_COLUMNS = (
    "review_id",
    "order_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
)
ORDER_COLUMNS = (
    "order_id",
    "order_status",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)
ORDER_ITEM_COLUMNS = ("order_id", "order_item_id", "product_id", "seller_id", "price")

REVIEW_DATES = ("review_creation_date", "review_answer_timestamp")
ORDER_DATES = (
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)

# Caracteres de controle invisíveis (exceto \t \n \r, tratados como espaço).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+")
_END_PUNCT = (".", "!", "?", "…")


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    """Garante que as colunas esperadas existem, com mensagem de erro clara."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: colunas ausentes {missing}. A base mudou de formato?")


def _read_csv(path: Path, columns: tuple[str, ...], dates: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode primeiro `python -m reviews.download`."
        )
    df = pd.read_csv(path, encoding="utf-8")
    validate_columns(df, columns, path.name)
    return _parse_dates(df.loc[:, list(columns)].copy(), dates)


def _parse_dates(df: pd.DataFrame, dates: tuple[str, ...]) -> pd.DataFrame:
    """Converte colunas de data. Valores inválidos viram NaT em vez de quebrar."""
    for col in dates:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_reviews(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return _read_csv(Path(raw_dir) / REVIEWS_FILE, REVIEW_COLUMNS, REVIEW_DATES)


def load_orders(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return _read_csv(Path(raw_dir) / ORDERS_FILE, ORDER_COLUMNS, ORDER_DATES)


def load_order_items(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return _read_csv(Path(raw_dir) / ORDER_ITEMS_FILE, ORDER_ITEM_COLUMNS, ())


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------


def clean_text(value: object) -> str:
    """Limpa um texto: Unicode NFC, sem caracteres de controle, espaços únicos.

    Valores ausentes (None, NaN) viram string vazia.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def build_text(title: str, message: str) -> str:
    """Junta título e comentário num texto só. O título costuma resumir a avaliação."""
    if not title:
        return message
    if not message:
        return title
    separator = " " if title.endswith(_END_PUNCT) else ". "
    return f"{title}{separator}{message}"


def normalize_for_grouping(text: str) -> str:
    """Forma canônica do texto, usada só para detectar textos repetidos.

    "Ótimo produto!!" e "otimo produto" viram a mesma chave. Isso impede que
    textos iguais caiam em treino e teste ao mesmo tempo (vazamento).
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", no_accents)).strip()


# ---------------------------------------------------------------------------
# Montagem da base de avaliações
# ---------------------------------------------------------------------------


def deduplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Mantém uma linha por review_id.

    Na Olist, o mesmo review_id às vezes aparece ligado a pedidos diferentes,
    com o mesmo texto. Fica a primeira ocorrência (por data de criação).
    """
    ordered = reviews.sort_values(
        ["review_creation_date", "review_answer_timestamp", "order_id"],
        kind="mergesort",
        na_position="last",
    )
    return ordered.drop_duplicates("review_id", keep="first")


def add_delivery_info(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta dados de entrega do pedido e o indicador de atraso.

    `entregue_com_atraso` fica vazio (<NA>) quando o pedido não tem data de
    entrega, para não confundir "não entregue" com "entregue no prazo".
    """
    validate_columns(orders, ORDER_COLUMNS, "orders")
    if orders["order_id"].duplicated().any():
        raise ValueError("orders: order_id repetido; a junção duplicaria avaliações.")

    orders = _parse_dates(orders.loc[:, list(ORDER_COLUMNS)].copy(), ORDER_DATES)
    merged = reviews.merge(orders, on="order_id", how="left", validate="many_to_one")
    delivered = merged["order_delivered_customer_date"].dt.normalize()
    estimated = merged["order_estimated_delivery_date"].dt.normalize()

    days_late = (delivered - estimated).dt.days
    merged["dias_de_atraso"] = days_late.astype("Int64")
    merged["entregue_com_atraso"] = (days_late > 0).astype("boolean")
    merged.loc[days_late.isna(), "entregue_com_atraso"] = pd.NA
    return merged


def prepare_reviews(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Gera a base final: avaliações com comentário, texto limpo, sentimento e entrega."""
    validate_columns(reviews, REVIEW_COLUMNS, "reviews")

    df = _parse_dates(reviews.loc[:, list(REVIEW_COLUMNS)].copy(), REVIEW_DATES)
    df = deduplicate_reviews(df)
    df = df.assign(
        titulo=df["review_comment_title"].map(clean_text),
        comentario=df["review_comment_message"].map(clean_text),
    )
    df = df[df["comentario"] != ""].copy()

    df["texto"] = [build_text(t, m) for t, m in zip(df["titulo"], df["comentario"], strict=True)]
    df["texto_grupo"] = df["texto"].map(normalize_for_grouping)
    df["sentimento"] = sentiment_from_score(df["review_score"])
    df["review_score"] = df["review_score"].astype(int)

    df = add_delivery_info(df, orders)

    columns = [
        "review_id",
        "order_id",
        "review_score",
        "sentimento",
        "titulo",
        "comentario",
        "texto",
        "texto_grupo",
        "review_creation_date",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "dias_de_atraso",
        "entregue_com_atraso",
    ]
    for col in ("review_id", "order_id", "titulo", "comentario", "texto", "texto_grupo"):
        df[col] = df[col].astype("string")
    return df.loc[:, columns].reset_index(drop=True)
