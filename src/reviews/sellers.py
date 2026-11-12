"""Números por vendedor: a camada de dados do agente de relatórios.

Tudo aqui é pandas puro, determinístico e testável: o agente (``agent.py``) só
lê o que estas funções devolvem e nunca calcula números por conta própria.

Cada número usa a base certa para o que ele mede:

- nota média, % de negativas, comparação com o marketplace e evolução mensal:
  **todas as avaliações**, com ou sem comentário. Só 41% das avaliações têm
  comentário, e quem está insatisfeito escreve mais; medir a nota só nas
  comentadas mostraria um vendedor pior do que ele é (no marketplace, 25,0% de
  negativas entre as comentadas contra 13,7% no total);
- tipos de problema e trechos: só as **negativas com comentário**, as únicas
  com texto para classificar (a mesma base da anotação e dos classificadores);
- atraso na entrega: **todos os pedidos entregues**, com ou sem avaliação.

Vendedor e marketplace são sempre medidos na mesma base. Pedidos com itens de
mais de um vendedor ficam de fora, porque a avaliação não diz a qual vendedor se
refere.

Todas as funções de estatística devolvem dicionários prontos para JSON, com
porcentagens já multiplicadas por 100 e arredondadas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reviews.annotation import CATEGORIES, LABELS
from reviews.data import add_delivery_info, deduplicate_reviews, load_orders, load_reviews
from reviews.labels import sentiment_from_score
from reviews.paths import PROCESSED_REVIEWS_FILE, RAW_DIR

ORDER_ITEMS_FILE = "olist_order_items_dataset.csv"
PRODUCTS_FILE = "olist_products_dataset.csv"
SELLERS_FILE = "olist_sellers_dataset.csv"

NEGATIVE = "negativa"
MIN_REVIEWS_FOR_RANKING = 30  # vendedores com menos avaliações não entram no percentil
MIN_MONTH_REVIEWS = 20  # meses com menos avaliações não entram em "pior/melhor mês"
MIN_MONTHS_FOR_RELATION = 6  # mínimo de meses relevantes para medir volume × negativas
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


def _single_seller_orders(items: pd.DataFrame) -> pd.DataFrame:
    """Primeiro item de cada pedido que tem um vendedor só (order_id, seller_id, product_id)."""
    per_order = items.groupby("order_id")["seller_id"].nunique()
    single = per_order[per_order == 1].index
    return (
        items[items["order_id"].isin(single)]
        .sort_values(["order_id", "order_item_id"], kind="mergesort")
        .drop_duplicates("order_id")
        .loc[:, ["order_id", "seller_id", "product_id"]]
    )


def build_seller_reviews(
    reviews: pd.DataFrame,
    comments: pd.DataFrame,
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
) -> pd.DataFrame:
    """Todas as avaliações (com e sem comentário) ligadas ao vendedor, à categoria e ao mês.

    ``reviews`` é a tabela bruta da Olist; ``comments`` é a base processada do projeto
    (só as avaliações com comentário, com o texto já limpo). As duplicatas saem com a
    mesma regra da base processada (``deduplicate_reviews``), então as avaliações com
    comentário ficam aqui na mesma ordem de lá; isso mantém os lotes do classificador
    iguais e o cache válido.
    """
    df = deduplicate_reviews(reviews)
    df = df.assign(review_score=df["review_score"].astype(int))
    df["sentimento"] = sentiment_from_score(df["review_score"])
    df = add_delivery_info(df, orders)

    texts = comments.loc[:, ["review_id", "texto"]].astype({"review_id": str})
    unknown = int((~texts["review_id"].isin(df["review_id"].astype(str))).sum())
    if unknown:
        raise ValueError(
            f"{unknown} avaliações da base processada não estão na base bruta. "
            "Refaça a base com `python -m reviews.prepare`."
        )
    df = df.astype({"review_id": str}).merge(
        texts, on="review_id", how="left", validate="one_to_one"
    )
    df["tem_comentario"] = df["texto"].notna()

    df = df.merge(_single_seller_orders(items), on="order_id", how="inner", validate="many_to_one")
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
        "tem_comentario",
        "texto",
        "dias_de_atraso",
        "entregue_com_atraso",
    ]
    return df.loc[:, keep].reset_index(drop=True)


def build_seller_orders(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Pedidos de um vendedor só, com o atraso calculado pela mesma regra das avaliações."""
    first = _single_seller_orders(items).loc[:, ["order_id", "seller_id"]]
    df = add_delivery_info(first, orders)
    return df.loc[:, ["order_id", "seller_id", "dias_de_atraso", "entregue_com_atraso"]]


@dataclass(frozen=True)
class SellerData:
    """As duas tabelas que as ferramentas consultam."""

    reviews: pd.DataFrame  # todas as avaliações, com a marca tem_comentario
    orders: pd.DataFrame  # todos os pedidos, para medir atraso


def load_seller_data(
    processed_file: Path = PROCESSED_REVIEWS_FILE, raw_dir: Path = RAW_DIR
) -> SellerData:
    raw_dir, processed_file = Path(raw_dir), Path(processed_file)
    if not processed_file.is_file():
        raise FileNotFoundError(
            f"{processed_file} não encontrado. Rode `python -m reviews.prepare`."
        )
    reviews = load_reviews(raw_dir)
    orders = load_orders(raw_dir)
    comments = pd.read_parquet(processed_file, columns=["review_id", "texto"])
    items = _read(
        raw_dir / ORDER_ITEMS_FILE, ["order_id", "order_item_id", "product_id", "seller_id"]
    )
    items["order_item_id"] = items["order_item_id"].astype(int)
    products = _read(raw_dir / PRODUCTS_FILE, ["product_id", "product_category_name"])
    sellers = _read(raw_dir / SELLERS_FILE, ["seller_id", "seller_city", "seller_state"])
    return SellerData(
        reviews=build_seller_reviews(reviews, comments, orders, items, products, sellers),
        orders=build_seller_orders(orders, items),
    )


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
        raise ValueError(f"Vendedor {seller_id} sem avaliações na base.")
    return part


def commented_negatives(df: pd.DataFrame, seller_id: str) -> pd.DataFrame:
    """Negativas com comentário do vendedor, na ordem da base (a ordem define os lotes)."""
    part = seller_frame(df, seller_id)
    keep = part["tem_comentario"].astype(bool) & (part["sentimento"] == NEGATIVE)
    return part[keep].reset_index(drop=True)


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
        "base": "todas as avaliações do vendedor, com ou sem comentário",
        "avaliacoes": int(len(part)),
        "nota_media": _mean(part["review_score"]),
        "pct_negativas": _pct(counts.get("negativa", 0), len(part)),
        "pct_neutras": _pct(counts.get("neutra", 0), len(part)),
        "pct_positivas": _pct(counts.get("positiva", 0), len(part)),
        "avaliacoes_negativas": int(counts.get("negativa", 0)),
        "avaliacoes_com_comentario": int(part["tem_comentario"].sum()),
        "negativas_com_comentario": int(len(commented_negatives(df, seller_id))),
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
        "base": "todas as avaliações, com ou sem comentário",
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


def _order_delays(orders: pd.DataFrame) -> dict:
    delivered = orders[orders["entregue_com_atraso"].notna()]
    late = delivered[delivered["entregue_com_atraso"].astype(bool)]
    return {
        "pedidos_entregues": int(len(delivered)),
        "pedidos_com_atraso": int(len(late)),
        "pct_com_atraso": _pct(len(late), len(delivered)),
        "mediana_dias_de_atraso": (
            float(late["dias_de_atraso"].astype(float).median()) if len(late) else None
        ),
    }


def _delay_effect(reviews: pd.DataFrame) -> dict:
    delivered = reviews[reviews["entregue_com_atraso"].notna()]
    late = delivered[delivered["entregue_com_atraso"].astype(bool)]
    on_time = delivered[~delivered["entregue_com_atraso"].astype(bool)]
    return {
        "avaliacoes_com_atraso": int(len(late)),
        "avaliacoes_no_prazo": int(len(on_time)),
        "nota_media_com_atraso": _mean(late["review_score"]),
        "nota_media_no_prazo": _mean(on_time["review_score"]),
        "pct_negativas_com_atraso": _negative_share(late) if len(late) else None,
        "pct_negativas_no_prazo": _negative_share(on_time) if len(on_time) else None,
    }


def delivery(df: pd.DataFrame, orders: pd.DataFrame, seller_id: str) -> dict:
    """Atraso medido nos pedidos; efeito do atraso medido nas notas desses pedidos.

    O atraso usa todos os pedidos entregues, com ou sem avaliação. Medido só nas
    avaliações com comentário, ele sairia inflado (no marketplace, 9,8% em vez de
    6,9%), porque quem recebe com atraso comenta mais.
    """
    part = seller_frame(df, seller_id)
    seller_orders, market_orders = (
        _order_delays(orders[orders["seller_id"] == seller_id]),
        _order_delays(orders),
    )
    seller_effect, market_effect = _delay_effect(part), _delay_effect(df)
    return {
        "atraso_nos_pedidos": {
            "base": "todos os pedidos entregues, com ou sem avaliação",
            "vendedor": seller_orders,
            "marketplace": market_orders,
            "diferenca_pct_com_atraso_pontos_percentuais": _diff(
                seller_orders["pct_com_atraso"], market_orders["pct_com_atraso"]
            ),
        },
        "notas_com_e_sem_atraso": {
            "base": "avaliações de pedidos entregues, com ou sem comentário",
            "vendedor": seller_effect,
            "marketplace": market_effect,
            "diferenca_nota_com_atraso": _diff(
                seller_effect["nota_media_com_atraso"], market_effect["nota_media_com_atraso"], 2
            ),
            "diferenca_nota_no_prazo": _diff(
                seller_effect["nota_media_no_prazo"], market_effect["nota_media_no_prazo"], 2
            ),
            "diferenca_pct_negativas_com_atraso_pontos_percentuais": _diff(
                seller_effect["pct_negativas_com_atraso"],
                market_effect["pct_negativas_com_atraso"],
            ),
            "diferenca_pct_negativas_no_prazo_pontos_percentuais": _diff(
                seller_effect["pct_negativas_no_prazo"], market_effect["pct_negativas_no_prazo"]
            ),
        },
        "avaliacoes_de_pedidos_nao_entregues": int(part["entregue_com_atraso"].isna().sum()),
    }


def _period(group: pd.DataFrame, months: list[str]) -> dict:
    return {
        "meses": months,
        "avaliacoes": int(len(group)),
        "pct_negativas": _negative_share(group),
        "nota_media": _mean(group["review_score"]),
    }


def volume_relation(months: list[dict]) -> dict | None:
    """Correlação de Spearman entre volume mensal e % de negativas (meses relevantes).

    Responde, com número, a pergunta "os meses de mais vendas são piores?", que o LLM
    tenderia a responder olhando a tabela, e às vezes errado.
    """
    if len(months) < MIN_MONTHS_FOR_RELATION:
        return None
    frame = pd.DataFrame(months)
    r = frame["avaliacoes"].corr(frame["pct_negativas"], method="spearman")
    if pd.isna(r):
        return None
    r = round(float(r), 2)
    strength = "sem relação clara" if abs(r) < 0.3 else "moderada" if abs(r) < 0.6 else "forte"
    if strength == "sem relação clara":
        reading = "sem relação clara: o volume do mês não acompanha a proporção de negativas"
    elif r > 0:
        reading = (
            f"relação {strength}: meses com mais avaliações tendem a ter proporção maior "
            "de negativas"
        )
    else:
        reading = (
            f"relação {strength}: meses com mais avaliações tendem a ter proporção menor "
            "de negativas"
        )
    return {
        "meses_usados": len(months),
        "correlacao_spearman": r,
        "leitura": reading,
        "cuidado": "correlação não mostra causa",
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

    # Meses de calendário: um mês sem avaliações continua contando como mês.
    calendar = pd.period_range(rows[0]["mes"], rows[-1]["mes"], freq="M").strftime("%Y-%m")
    months = list(calendar)
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
        "base": "todas as avaliações do vendedor, com ou sem comentário",
        "meses": rows,
        "relacao_volume_negativas": volume_relation(relevant),
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


def reference_from_matrix(matrix: np.ndarray, description: str) -> dict:
    """Distribuição de referência dos tipos de problema (% das avaliações com cada um)."""
    matrix = np.asarray(matrix, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != len(LABELS) or len(matrix) == 0:
        raise ValueError(f"Matriz de referência inválida: {matrix.shape}.")
    return {
        "avaliacoes": int(len(matrix)),
        "pct": {label: _pct(matrix[:, j].sum(), len(matrix)) for j, label in enumerate(LABELS)},
        "descricao": description,
    }


def problems(seller_negatives: pd.DataFrame, matrix: np.ndarray, reference: dict) -> dict:
    """Tipos de problema nas negativas com comentário do vendedor (LLM) × referência."""
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
                "pct_das_negativas_com_comentario": seller_pct,
                "pct_referencia_marketplace": market_pct,
                "diferenca_pontos_percentuais": _diff(seller_pct, market_pct),
            }
        )
    rows.sort(key=lambda r: -r["avaliacoes"])
    for row in rows:  # empates ficam com a mesma posição
        row["posicao"] = 1 + sum(other["avaliacoes"] > row["avaliacoes"] for other in rows)
    return {
        "base": "só as avaliações negativas com comentário, as únicas com texto para classificar",
        "avaliacoes_negativas_classificadas": n,
        "ordem": "da categoria mais citada para a menos citada (campo posicao)",
        "referencia_marketplace": reference["descricao"],
        "observacao": "uma avaliação pode ter mais de um tipo de problema",
        "categorias": rows,
    }


def examples(
    seller_negatives: pd.DataFrame, matrix: np.ndarray, category: str, n: int = 3, seed: int = 42
) -> dict:
    """Trechos reais de avaliações negativas com comentário de uma categoria (sorteio fixo)."""
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
    """Vendedores com mais avaliações negativas (todas, com ou sem comentário)."""
    grouped = df.groupby("seller_id").agg(
        avaliacoes=("review_id", "size"),
        negativas=("sentimento", lambda s: int((s == NEGATIVE).sum())),
        nota_media=("review_score", "mean"),
    )
    grouped["pct_negativas"] = (grouped["negativas"] / grouped["avaliacoes"] * 100).round(1)
    grouped["nota_media"] = grouped["nota_media"].round(2)
    commented = df["tem_comentario"].astype(bool) & (df["sentimento"] == NEGATIVE)
    grouped["negativas_com_comentario"] = (
        df[commented].groupby("seller_id").size().reindex(grouped.index, fill_value=0).astype(int)
    )
    return grouped.sort_values(["negativas", "avaliacoes"], ascending=False).head(n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Vendedores com mais avaliações negativas.")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    print(top_sellers(load_seller_data().reviews, args.top).to_string())


if __name__ == "__main__":
    main()
