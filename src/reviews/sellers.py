"""Números por vendedor: a camada de dados do agente de relatórios.

Tudo aqui é pandas puro, determinístico e testável: o agente (``agent.py``) só
lê o que estas funções devolvem e nunca calcula números por conta própria.

Universo: avaliações **com comentário** (a base do projeto). Vendedor e
marketplace são sempre medidos nesse mesmo universo, para a comparação ser justa.
Pedidos com itens de mais de um vendedor ficam de fora, porque a avaliação não
diz a qual vendedor se refere.

Todas as funções de estatística devolvem dicionários prontos para JSON, com
porcentagens já multiplicadas por 100 e arredondadas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reviews.annotation import CATEGORIES, LABELS, load_labels
from reviews.paths import LABELS_FILE, PROCESSED_REVIEWS_FILE, RAW_DIR

ORDER_ITEMS_FILE = "olist_order_items_dataset.csv"
PRODUCTS_FILE = "olist_products_dataset.csv"
SELLERS_FILE = "olist_sellers_dataset.csv"

NEGATIVE = "negativa"
MIN_REVIEWS_FOR_RANKING = 30  # vendedores com menos avaliações não entram no percentil
MIN_MONTH_REVIEWS = 20  # meses com menos avaliações não entram em "pior/melhor mês"
MAX_QUOTE_CHARS = 280
NO_CATEGORY = "sem categoria"


# ---------------------------------------------------------------------------
# Montagem da base
# ---------------------------------------------------------------------------


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} não encontrado. Rode `python -m reviews.download`.")
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: colunas ausentes {missing}.")
    return df.loc[:, columns]


def _pretty_category(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        return NO_CATEGORY
    return name.strip().replace("_", " ")


def build_seller_reviews(
    reviews: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
) -> pd.DataFrame:
    """Liga cada avaliação ao vendedor, à categoria do produto e ao mês."""
    per_order = items.groupby("order_id")["seller_id"].nunique()
    single = per_order[per_order == 1].index
    first_item = (
        items[items["order_id"].isin(single)]
        .sort_values(["order_id", "order_item_id"], kind="mergesort")
        .drop_duplicates("order_id")
        .loc[:, ["order_id", "seller_id", "product_id"]]
    )
    df = reviews.merge(first_item, on="order_id", how="inner", validate="many_to_one")
    df = df.merge(
        products.loc[:, ["product_id", "product_category_name"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        sellers.loc[:, ["seller_id", "seller_city", "seller_state"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    df["categoria"] = df["product_category_name"].map(_pretty_category)
    df["mes"] = pd.to_datetime(df["review_creation_date"]).dt.strftime("%Y-%m")
    keep = [
        "review_id",
        "order_id",
        "seller_id",
        "seller_city",
        "seller_state",
        "categoria",
        "mes",
        "review_score",
        "sentimento",
        "texto",
        "dias_de_atraso",
        "entregue_com_atraso",
    ]
    return df.loc[:, keep].reset_index(drop=True)


def load_seller_reviews(
    processed_file: Path = PROCESSED_REVIEWS_FILE, raw_dir: Path = RAW_DIR
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    reviews = pd.read_parquet(processed_file)
    items = _read(
        raw_dir / ORDER_ITEMS_FILE, ["order_id", "order_item_id", "product_id", "seller_id"]
    )
    items["order_item_id"] = items["order_item_id"].astype(int)
    products = _read(raw_dir / PRODUCTS_FILE, ["product_id", "product_category_name"])
    sellers = _read(raw_dir / SELLERS_FILE, ["seller_id", "seller_city", "seller_state"])
    return build_seller_reviews(reviews, items, products, sellers)


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _pct(part: float, total: float) -> float | None:
    return round(float(part) / float(total) * 100, 1) if total else None


def _mean(series: pd.Series, decimals: int = 2) -> float | None:
    series = series.dropna()
    return round(float(series.mean()), decimals) if len(series) else None


def _diff(a: float | None, b: float | None, decimals: int = 1) -> float | None:
    return round(a - b, decimals) if a is not None and b is not None else None


def seller_frame(df: pd.DataFrame, seller_id: str) -> pd.DataFrame:
    part = df[df["seller_id"] == seller_id]
    if part.empty:
        raise ValueError(f"Vendedor {seller_id} sem avaliações com comentário na base.")
    return part


def _negative_share(df: pd.DataFrame) -> float | None:
    return _pct((df["sentimento"] == NEGATIVE).sum(), len(df))


# ---------------------------------------------------------------------------
# Estatísticas (cada uma vira uma ferramenta do agente)
# ---------------------------------------------------------------------------


def summary(df: pd.DataFrame, seller_id: str, top_categories: int = 5) -> dict:
    part = seller_frame(df, seller_id)
    counts = part["sentimento"].value_counts()
    categories = part["categoria"].value_counts().head(top_categories)
    first = part.iloc[0]
    return {
        "vendedor": seller_id,
        "cidade": first["seller_city"] if isinstance(first["seller_city"], str) else None,
        "estado": first["seller_state"] if isinstance(first["seller_state"], str) else None,
        "avaliacoes_com_comentario": int(len(part)),
        "nota_media": _mean(part["review_score"]),
        "pct_negativas": _pct(counts.get("negativa", 0), len(part)),
        "pct_neutras": _pct(counts.get("neutra", 0), len(part)),
        "pct_positivas": _pct(counts.get("positiva", 0), len(part)),
        "avaliacoes_negativas": int(counts.get("negativa", 0)),
        "primeiro_mes": str(part["mes"].min()),
        "ultimo_mes": str(part["mes"].max()),
        "categorias_principais": [
            {"categoria": name, "avaliacoes": int(n), "pct": _pct(n, len(part))}
            for name, n in categories.items()
        ],
    }


def compare_with_marketplace(df: pd.DataFrame, seller_id: str, top_categories: int = 3) -> dict:
    """Vendedor × marketplace: geral, posição entre vendedores e nas mesmas categorias."""
    part = seller_frame(df, seller_id)
    seller_neg = _negative_share(part)
    market_neg = _negative_share(df)

    per_seller = df.groupby("seller_id").agg(
        n=("review_id", "size"), neg=("sentimento", lambda s: (s == NEGATIVE).mean() * 100)
    )
    ranked = per_seller[per_seller["n"] >= MIN_REVIEWS_FOR_RANKING]
    position = None
    if seller_id in ranked.index and len(ranked) > 1:
        # % dos vendedores comparáveis com MENOS negativas que este (quanto maior, pior).
        position = _pct((ranked["neg"] < ranked.loc[seller_id, "neg"]).sum(), len(ranked) - 1)

    same_categories = []
    for category in part["categoria"].value_counts().head(top_categories).index:
        in_cat = df[df["categoria"] == category]
        seller_cat = part[part["categoria"] == category]
        s, m = _negative_share(seller_cat), _negative_share(in_cat)
        same_categories.append(
            {
                "categoria": category,
                "avaliacoes_vendedor": int(len(seller_cat)),
                "pct_negativas_vendedor": s,
                "pct_negativas_marketplace": m,
                "diferenca_pontos_percentuais": _diff(s, m),
            }
        )

    return {
        "nota_media_vendedor": _mean(part["review_score"]),
        "nota_media_marketplace": _mean(df["review_score"]),
        "diferenca_nota": _diff(_mean(part["review_score"]), _mean(df["review_score"]), 2),
        "pct_negativas_vendedor": seller_neg,
        "pct_negativas_marketplace": market_neg,
        "diferenca_pontos_percentuais": _diff(seller_neg, market_neg),
        "vendedores_comparaveis": int(len(ranked)),
        "minimo_de_avaliacoes_para_comparar": MIN_REVIEWS_FOR_RANKING,
        "pct_vendedores_com_menos_negativas": position,
        "mesmas_categorias": same_categories,
    }


def _delivery_block(df: pd.DataFrame) -> dict:
    delivered = df[df["entregue_com_atraso"].notna()]
    late = delivered[delivered["entregue_com_atraso"].astype(bool)]
    on_time = delivered[~delivered["entregue_com_atraso"].astype(bool)]
    return {
        "avaliacoes_de_pedidos_entregues": int(len(delivered)),
        "pct_entregues_com_atraso": _pct(len(late), len(delivered)),
        "mediana_dias_de_atraso": (
            float(late["dias_de_atraso"].astype(float).median()) if len(late) else None
        ),
        "nota_media_com_atraso": _mean(late["review_score"]),
        "nota_media_no_prazo": _mean(on_time["review_score"]),
        "pct_negativas_com_atraso": _negative_share(late) if len(late) else None,
        "pct_negativas_no_prazo": _negative_share(on_time) if len(on_time) else None,
        "avaliacoes_de_pedidos_nao_entregues": int(df["entregue_com_atraso"].isna().sum()),
    }


def delivery(df: pd.DataFrame, seller_id: str) -> dict:
    seller = _delivery_block(seller_frame(df, seller_id))
    market = _delivery_block(df)
    return {
        "vendedor": seller,
        "marketplace": market,
        "diferenca_pct_atraso_pontos_percentuais": _diff(
            seller["pct_entregues_com_atraso"], market["pct_entregues_com_atraso"]
        ),
    }


def _period(group: pd.DataFrame, months: list[str]) -> dict:
    return {
        "meses": months,
        "avaliacoes": int(len(group)),
        "pct_negativas": _negative_share(group),
        "nota_media": _mean(group["review_score"]),
    }


def monthly(df: pd.DataFrame, seller_id: str) -> dict:
    """Série mensal e um resumo da tendência já calculado (o LLM não lê tendência sozinho)."""
    part = seller_frame(df, seller_id)
    rows = []
    for month, group in part.groupby("mes", sort=True):
        rows.append(
            {
                "mes": str(month),
                "avaliacoes": int(len(group)),
                "pct_negativas": _negative_share(group),
                "nota_media": _mean(group["review_score"]),
            }
        )

    relevant = [r for r in rows if r["avaliacoes"] >= MIN_MONTH_REVIEWS]
    worst = max(relevant, key=lambda r: r["pct_negativas"], default=None)
    best = min(relevant, key=lambda r: r["pct_negativas"], default=None)
    biggest = max(rows, key=lambda r: r["avaliacoes"], default=None)

    months = [r["mes"] for r in rows]
    trend = None
    if len(months) >= 6:
        last, previous = months[-3:], months[-6:-3]
        now = _period(part[part["mes"].isin(last)], last)
        before = _period(part[part["mes"].isin(previous)], previous)
        trend = {
            "ultimos_3_meses": now,
            "3_meses_anteriores": before,
            "diferenca_pct_negativas_pontos_percentuais": _diff(
                now["pct_negativas"], before["pct_negativas"]
            ),
            "diferenca_nota": _diff(now["nota_media"], before["nota_media"], 2),
        }

    return {
        "meses": rows,
        "minimo_de_avaliacoes_para_mes_relevante": MIN_MONTH_REVIEWS,
        "meses_relevantes": len(relevant),
        "mes_com_mais_negativas": worst,
        "mes_com_menos_negativas": best,
        "mes_com_mais_avaliacoes": biggest,
        "tendencia": trend,
        "observacao": (
            "meses com menos avaliações que o mínimo oscilam muito e ficam fora de "
            "mes_com_mais_negativas e mes_com_menos_negativas; a tendência usa todos os meses"
        ),
    }


# ---------------------------------------------------------------------------
# Tipos de problema
# ---------------------------------------------------------------------------


def marketplace_problem_reference(labels_file: Path = LABELS_FILE) -> dict:
    """Distribuição de referência: as 1.500 negativas sorteadas e anotadas à mão."""
    labels = load_labels(labels_file)
    labels = labels[labels[list(LABELS)].sum(axis=1) > 0]
    return {
        "avaliacoes": int(len(labels)),
        "pct": {label: _pct(labels[label].sum(), len(labels)) for label in LABELS},
    }


def problems(seller_negatives: pd.DataFrame, matrix: np.ndarray, reference: dict) -> dict:
    """Tipos de problema nas negativas do vendedor (classificadas pelo LLM) × referência."""
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (len(seller_negatives), len(LABELS)):
        raise ValueError(
            f"Matriz {matrix.shape} não combina com {len(seller_negatives)} avaliações."
        )
    n = len(seller_negatives)
    rows = []
    for j, label in enumerate(LABELS):
        count = int(matrix[:, j].sum())
        seller_pct = _pct(count, n)
        market_pct = reference["pct"][label]
        rows.append(
            {
                "categoria": label,
                "nome": CATEGORIES[label],
                "avaliacoes": count,
                "pct_das_negativas": seller_pct,
                "pct_referencia_marketplace": market_pct,
                "diferenca_pontos_percentuais": _diff(seller_pct, market_pct),
            }
        )
    rows.sort(key=lambda r: -r["avaliacoes"])
    for row in rows:  # empates ficam com a mesma posição
        row["posicao"] = 1 + sum(other["avaliacoes"] > row["avaliacoes"] for other in rows)
    return {
        "avaliacoes_negativas_classificadas": n,
        "ordem": "da categoria mais citada para a menos citada (campo posicao)",
        "referencia_marketplace": (
            f"{reference['avaliacoes']} avaliações negativas sorteadas e anotadas à mão"
        ),
        "observacao": "uma avaliação pode ter mais de um tipo de problema",
        "categorias": rows,
    }


def examples(
    seller_negatives: pd.DataFrame, matrix: np.ndarray, category: str, n: int = 3, seed: int = 42
) -> dict:
    """Trechos reais de avaliações negativas de uma categoria (sorteio fixo)."""
    if category not in LABELS:
        raise ValueError(f"Categoria desconhecida: {category}")
    n = max(1, min(int(n), 5))
    mask = np.asarray(matrix, dtype=int)[:, list(LABELS).index(category)] == 1
    pool = seller_negatives[mask]
    chosen = pool.sample(min(n, len(pool)), random_state=seed) if len(pool) else pool
    quotes = []
    for _, row in chosen.iterrows():
        text = str(row["texto"])
        if len(text) > MAX_QUOTE_CHARS:
            text = text[: MAX_QUOTE_CHARS - 1].rsplit(" ", 1)[0] + "…"
        quotes.append({"texto": text, "nota": int(row["review_score"]), "mes": str(row["mes"])})
    return {
        "categoria": category,
        "nome": CATEGORIES[category],
        "total": int(mask.sum()),
        "exemplos": quotes,
    }


def top_sellers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Vendedores com mais avaliações negativas (com comentário)."""
    grouped = df.groupby("seller_id").agg(
        avaliacoes=("review_id", "size"),
        negativas=("sentimento", lambda s: int((s == NEGATIVE).sum())),
        nota_media=("review_score", "mean"),
    )
    grouped["pct_negativas"] = (grouped["negativas"] / grouped["avaliacoes"] * 100).round(1)
    grouped["nota_media"] = grouped["nota_media"].round(2)
    return grouped.sort_values(["negativas", "avaliacoes"], ascending=False).head(n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Vendedores com mais avaliações negativas.")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    print(top_sellers(load_seller_reviews(), args.top).to_string())


if __name__ == "__main__":
    main()
