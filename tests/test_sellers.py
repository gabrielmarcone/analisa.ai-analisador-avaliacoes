"""Testes da camada de dados por vendedor (pandas puro, sem rede)."""

import numpy as np
import pandas as pd
import pytest

from reviews import sellers
from reviews.annotation import LABEL_COLUMNS, LABELS, save_labels

K = len(LABELS)


def _reviews():
    """Duas lojas: A (4 avaliações) e B (3), mais um pedido com dois vendedores."""
    rows = [
        # review_id, order_id, nota, sentimento, data, dias_atraso, atrasou
        ("r1", "o1", 1, "negativa", "2017-01-10", 5, True),
        ("r2", "o2", 5, "positiva", "2017-01-20", -3, False),
        ("r3", "o3", 2, "negativa", "2017-02-05", None, None),
        ("r4", "o4", 4, "positiva", "2017-02-15", 0, False),
        ("r5", "o5", 5, "positiva", "2017-01-03", -1, False),
        ("r6", "o6", 3, "neutra", "2017-03-01", 2, True),
        ("r7", "o7", 1, "negativa", "2017-03-09", -2, False),
        ("r8", "o8", 1, "negativa", "2017-03-09", -2, False),  # pedido com 2 vendedores
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "order_id",
            "review_score",
            "sentimento",
            "review_creation_date",
            "dias_de_atraso",
            "entregue_com_atraso",
        ],
    )
    df["texto"] = [f"texto {i}" for i in range(len(df))]
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"])
    df["dias_de_atraso"] = df["dias_de_atraso"].astype("Int64")
    df["entregue_com_atraso"] = df["entregue_com_atraso"].astype("boolean")
    return df


def _items():
    return pd.DataFrame(
        [
            ("o1", 1, "p1", "A"),
            ("o2", 1, "p2", "A"),
            ("o3", 1, "p1", "A"),
            ("o4", 1, "p3", "A"),
            ("o4", 2, "p1", "A"),
            ("o5", 1, "p2", "B"),
            ("o6", 1, "p2", "B"),
            ("o7", 1, "p4", "B"),
            ("o8", 1, "p1", "A"),
            ("o8", 2, "p2", "B"),
        ],
        columns=["order_id", "order_item_id", "product_id", "seller_id"],
    )


def _products():
    return pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4"],
            "product_category_name": ["cama_mesa_banho", "brinquedos", "cama_mesa_banho", None],
        }
    )


def _sellers():
    return pd.DataFrame(
        {
            "seller_id": ["A", "B"],
            "seller_city": ["ibitinga", "sao paulo"],
            "seller_state": ["SP", "SP"],
        }
    )


@pytest.fixture
def df():
    return sellers.build_seller_reviews(_reviews(), _items(), _products(), _sellers())


def test_monta_base_por_vendedor(df):
    assert "r8" not in set(df["review_id"])  # pedido com dois vendedores fica de fora
    assert len(df) == 7
    a = df[df["seller_id"] == "A"].set_index("review_id")
    assert a.loc["r4", "categoria"] == "cama mesa banho"  # primeiro item do pedido
    b = df[df["seller_id"] == "B"].set_index("review_id")
    assert b.loc["r7", "categoria"] == sellers.NO_CATEGORY
    assert a.loc["r1", "mes"] == "2017-01"


def test_resumo(df):
    out = sellers.summary(df, "A")
    assert out["avaliacoes_com_comentario"] == 4
    assert out["nota_media"] == 3.0
    assert out["pct_negativas"] == 50.0 and out["pct_positivas"] == 50.0
    assert out["primeiro_mes"] == "2017-01" and out["ultimo_mes"] == "2017-02"
    assert out["categorias_principais"][0] == {
        "categoria": "cama mesa banho",
        "avaliacoes": 3,
        "pct": 75.0,
    }
    with pytest.raises(ValueError, match="sem avaliações"):
        sellers.summary(df, "Z")


def test_comparacao_com_marketplace(df, monkeypatch):
    monkeypatch.setattr(sellers, "MIN_REVIEWS_FOR_RANKING", 3)
    out = sellers.compare_with_marketplace(df, "A")
    # marketplace: 3 negativas em 7
    assert out["pct_negativas_marketplace"] == 42.9
    assert out["diferenca_pontos_percentuais"] == 7.1
    assert out["vendedores_comparaveis"] == 2
    assert out["pct_vendedores_com_menos_negativas"] == 100.0  # B tem 33,3%
    cama = out["mesmas_categorias"][0]
    assert cama["categoria"] == "cama mesa banho" and cama["pct_negativas_vendedor"] == 66.7


def test_entregas(df):
    out = sellers.delivery(df, "A")
    v = out["vendedor"]
    assert v["avaliacoes_de_pedidos_entregues"] == 3  # r3 sem data de entrega
    assert v["pct_entregues_com_atraso"] == 33.3
    assert v["mediana_dias_de_atraso"] == 5.0
    assert v["nota_media_com_atraso"] == 1.0 and v["nota_media_no_prazo"] == 4.5
    assert v["avaliacoes_de_pedidos_nao_entregues"] == 1
    assert out["marketplace"]["pct_entregues_com_atraso"] == 33.3
    assert out["diferenca_pct_atraso_pontos_percentuais"] == 0.0


def test_tendencia_e_meses_extremos(monkeypatch):
    monkeypatch.setattr(sellers, "MIN_MONTH_REVIEWS", 2)
    rows = []
    # 7 meses; nos 3 últimos as negativas caem. 2017-04 tem só 1 avaliação (fora do ranking).
    plan = {
        "2017-01": (2, 1),
        "2017-02": (2, 2),
        "2017-03": (2, 1),
        "2017-04": (1, 1),
        "2017-05": (2, 0),
        "2017-06": (4, 1),
        "2017-07": (2, 0),
    }
    for month, (n, neg) in plan.items():
        for i in range(n):
            bad = i < neg
            rows.append(
                {
                    "review_id": f"{month}-{i}",
                    "seller_id": "A",
                    "mes": month,
                    "sentimento": "negativa" if bad else "positiva",
                    "review_score": 1 if bad else 5,
                }
            )
    out = sellers.monthly(pd.DataFrame(rows), "A")
    assert out["meses_relevantes"] == 6
    assert out["mes_com_mais_negativas"]["mes"] == "2017-02"  # 100%, com volume mínimo
    assert out["mes_com_menos_negativas"]["mes"] == "2017-05"  # 0%; empate com 07: o primeiro
    assert out["mes_com_mais_avaliacoes"]["mes"] == "2017-06"
    trend = out["tendencia"]
    assert trend["ultimos_3_meses"]["meses"] == ["2017-05", "2017-06", "2017-07"]
    assert trend["ultimos_3_meses"]["pct_negativas"] == 12.5  # 1 de 8
    assert trend["3_meses_anteriores"]["pct_negativas"] == 80.0  # 4 de 5
    assert trend["diferenca_pct_negativas_pontos_percentuais"] == -67.5


def test_evolucao_mensal(df):
    out = sellers.monthly(df, "B")
    assert out["tendencia"] is None and out["mes_com_mais_negativas"] is None  # pouco dado
    meses = out["meses"]
    assert [m["mes"] for m in meses] == ["2017-01", "2017-03"]
    assert meses[1] == {"mes": "2017-03", "avaliacoes": 2, "pct_negativas": 50.0, "nota_media": 2.0}


def test_referencia_problemas_e_exemplos(tmp_path, df):
    labels = pd.DataFrame(0, index=range(4), columns=list(LABEL_COLUMNS))
    labels["review_id"] = [f"x{i}" for i in range(4)]
    labels["atraso"] = [1, 1, 0, 0]
    labels["sem_reclamacao"] = [0, 0, 0, 1]
    labels["defeito_qualidade"] = [0, 1, 1, 0]
    labels["revisar"] = False
    labels["observacao"] = ""
    labels["anotado_em"] = "2026-11-01T00:00:00+00:00"
    path = tmp_path / "rotulos.csv"
    save_labels(labels, path)
    ref = sellers.marketplace_problem_reference(path)
    assert ref["avaliacoes"] == 4 and ref["pct"]["atraso"] == 50.0

    neg = df[(df["seller_id"] == "A") & (df["sentimento"] == "negativa")].reset_index(drop=True)
    matrix = np.zeros((len(neg), K), dtype=int)
    matrix[:, 0] = 1  # as duas com atraso
    matrix[1, 4] = 1  # uma também com defeito
    out = sellers.problems(neg, matrix, ref)
    first = out["categorias"][0]
    assert first["categoria"] == "atraso" and first["pct_das_negativas"] == 100.0
    assert first["posicao"] == 1 and out["categorias"][1]["posicao"] == 2
    assert out["categorias"][-1]["posicao"] == 3  # seis categorias empatadas em zero
    assert first["diferenca_pontos_percentuais"] == 50.0
    with pytest.raises(ValueError, match="não combina"):
        sellers.problems(neg, matrix[:1], ref)

    neg.loc[0, "texto"] = "palavra " * 100
    ex = sellers.examples(neg, matrix, "atraso", n=9)
    assert ex["total"] == 2 and len(ex["exemplos"]) == 2
    assert all(len(e["texto"]) <= sellers.MAX_QUOTE_CHARS for e in ex["exemplos"])
    assert sellers.examples(neg, matrix, "outro")["exemplos"] == []
    with pytest.raises(ValueError, match="desconhecida"):
        sellers.examples(neg, matrix, "inexistente")


def test_top_vendedores(df):
    top = sellers.top_sellers(df, 2)
    assert list(top.index) == ["A", "B"]
    assert top.loc["A", "negativas"] == 2 and top.loc["A", "pct_negativas"] == 50.0


def test_carrega_dos_arquivos(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _items().to_csv(raw / sellers.ORDER_ITEMS_FILE, index=False)
    _products().to_csv(raw / sellers.PRODUCTS_FILE, index=False)
    _sellers().to_csv(raw / sellers.SELLERS_FILE, index=False)
    parquet = tmp_path / "reviews.parquet"
    _reviews().to_parquet(parquet)
    df = sellers.load_seller_reviews(parquet, raw)
    assert len(df) == 7 and set(df["seller_id"]) == {"A", "B"}
    with pytest.raises(FileNotFoundError, match="reviews.download"):
        sellers.load_seller_reviews(parquet, tmp_path / "vazio")
