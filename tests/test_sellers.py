"""Testes da camada de dados por vendedor (pandas puro, sem rede)."""

import numpy as np
import pandas as pd
import pytest

from reviews import sellers
from reviews.annotation import LABELS
from reviews.data import ORDERS_FILE, REVIEWS_FILE, prepare_reviews

K = len(LABELS)


ESTIMATED = pd.Timestamp("2017-01-15")


def _reviews():
    """Tabela bruta: A (6 avaliações, 2 sem comentário), B (3) e um pedido com dois vendedores."""
    rows = [
        # review_id, order_id, nota, comentário, data
        ("r1", "o1", 1, "Não chegou no prazo", "2017-01-10"),
        ("r2", "o2", 5, "Ótimo", "2017-01-20"),
        ("r3", "o3", 2, "Nunca recebi", "2017-02-05"),
        ("r4", "o4", 4, "Bom", "2017-02-15"),
        ("r5", "o5", 5, "Perfeito", "2017-01-03"),
        ("r6", "o6", 3, "Ok", "2017-03-01"),
        ("r7", "o7", 1, "Veio quebrado", "2017-03-09"),
        ("r8", "o8", 1, "Péssimo", "2017-03-09"),  # pedido com 2 vendedores
        ("r9", "o10", 5, None, "2017-02-20"),  # sem comentário
        ("r10", "o11", 1, "  ", "2017-01-25"),  # só espaços: sem comentário
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "order_id",
            "review_score",
            "review_comment_message",
            "review_creation_date",
        ],
    )
    df["review_comment_title"] = None
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"])
    df["review_answer_timestamp"] = df["review_creation_date"] + pd.Timedelta(days=1)
    return df


def _orders():
    """Dias de atraso por pedido (None = não entregue). o9 não tem avaliação."""
    delays = {
        "o1": 5,
        "o2": -3,
        "o3": None,
        "o4": 0,
        "o5": -1,
        "o6": 2,
        "o7": -2,
        "o8": -2,
        "o9": 3,
        "o10": -1,
        "o11": 2,
    }
    rows = []
    for order_id, days in delays.items():
        delivered = ESTIMATED + pd.Timedelta(days=days, hours=10) if days is not None else pd.NaT
        rows.append(
            {
                "order_id": order_id,
                "order_status": "delivered" if days is not None else "shipped",
                "order_purchase_timestamp": ESTIMATED - pd.Timedelta(days=20),
                "order_delivered_customer_date": delivered,
                "order_estimated_delivery_date": ESTIMATED,
            }
        )
    return pd.DataFrame(rows)


def _comments():
    """A base processada do projeto: só as avaliações com comentário."""
    return prepare_reviews(_reviews(), _orders())


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
            ("o9", 1, "p2", "A"),
            ("o10", 1, "p1", "A"),
            ("o11", 1, "p2", "A"),
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


def seller_data():
    return sellers.SellerData(
        reviews=sellers.build_seller_reviews(
            _reviews(), _comments(), _orders(), _items(), _products(), _sellers()
        ),
        orders=sellers.build_seller_orders(_orders(), _items()),
    )


@pytest.fixture
def data():
    return seller_data()


@pytest.fixture
def df(data):
    return data.reviews


def test_monta_base_com_todas_as_avaliacoes(df):
    assert "r8" not in set(df["review_id"])  # pedido com dois vendedores fica de fora
    assert len(df) == 9
    a = df[df["seller_id"] == "A"].set_index("review_id")
    assert a.loc["r4", "categoria"] == "cama mesa banho"  # primeiro item do pedido
    b = df[df["seller_id"] == "B"].set_index("review_id")
    assert b.loc["r7", "categoria"] == sellers.NO_CATEGORY
    assert a.loc["r1", "mes"] == "2017-01"
    assert a.loc["r1", "texto"] == "Não chegou no prazo" and bool(a.loc["r1", "tem_comentario"])
    assert not a.loc["r9", "tem_comentario"] and pd.isna(a.loc["r9", "texto"])
    assert not a.loc["r10", "tem_comentario"]  # só espaços não é comentário
    assert a.loc["r1", "sentimento"] == "negativa" and bool(a.loc["r1", "entregue_com_atraso"])
    assert pd.isna(a.loc["r3", "entregue_com_atraso"])


def test_comentadas_seguem_a_ordem_da_base_processada(df):
    """A ordem das negativas define os lotes do classificador, e os lotes, o cache."""
    processed = _comments()
    in_base = processed[processed["review_id"].isin(df["review_id"])]
    commented = df[df["tem_comentario"]]
    assert list(commented["review_id"]) == list(in_base["review_id"])
    assert list(commented["texto"]) == list(in_base["texto"])
    assert list(sellers.commented_negatives(df, "A")["review_id"]) == ["r1", "r3"]


def test_base_processada_desatualizada_e_rejeitada():
    extra = pd.concat([_comments(), _comments().assign(review_id="r99")], ignore_index=True)
    with pytest.raises(ValueError, match="reviews.prepare"):
        sellers.build_seller_reviews(
            _reviews(), extra, _orders(), _items(), _products(), _sellers()
        )


def test_resumo(df):
    out = sellers.summary(df, "A")
    assert out["avaliacoes"] == 6 and out["avaliacoes_com_comentario"] == 4
    assert out["avaliacoes_negativas"] == 3 and out["negativas_com_comentario"] == 2
    assert out["nota_media"] == 3.0
    assert out["pct_negativas"] == 50.0 and out["pct_positivas"] == 50.0
    assert out["primeiro_mes"] == "2017-01" and out["ultimo_mes"] == "2017-02"
    assert out["categorias_principais"][0] == {
        "categoria": "cama mesa banho",
        "avaliacoes": 4,
        "pct": 66.7,
    }
    with pytest.raises(ValueError, match="sem avaliações"):
        sellers.summary(df, "Z")


def test_comparacao_com_marketplace(df, monkeypatch):
    monkeypatch.setattr(sellers, "MIN_REVIEWS_FOR_RANKING", 3)
    out = sellers.compare_with_marketplace(df, "A")
    # marketplace: 4 negativas em 9 (todas as avaliações, com ou sem comentário)
    assert out["pct_negativas_marketplace"] == 44.4
    assert out["diferenca_pontos_percentuais"] == 5.6
    assert out["vendedores_comparaveis"] == 2
    assert out["pct_vendedores_com_menos_negativas"] == 100.0  # B tem 33,3%
    cama = out["mesmas_categorias"][0]
    assert cama["categoria"] == "cama mesa banho" and cama["pct_negativas_vendedor"] == 50.0


def test_entregas_medem_atraso_nos_pedidos_e_efeito_nas_notas(data):
    out = sellers.delivery(data.reviews, data.orders, "A")
    orders = out["atraso_nos_pedidos"]
    # A: o1, o9 e o11 atrasaram entre 6 entregues (o9 não tem avaliação; o3 não chegou)
    assert orders["vendedor"] == {
        "pedidos_entregues": 6,
        "pedidos_com_atraso": 3,
        "pct_com_atraso": 50.0,
        "mediana_dias_de_atraso": 3.0,
    }
    assert orders["marketplace"]["pedidos_entregues"] == 9  # o8 (dois vendedores) fica de fora
    assert orders["marketplace"]["pct_com_atraso"] == 44.4
    assert orders["diferenca_pct_com_atraso_pontos_percentuais"] == 5.6

    notes = out["notas_com_e_sem_atraso"]
    assert notes["vendedor"] == {
        "avaliacoes_com_atraso": 2,  # r1 e r10 (sem comentário também conta)
        "avaliacoes_no_prazo": 3,
        "nota_media_com_atraso": 1.0,
        "nota_media_no_prazo": 4.67,
        "pct_negativas_com_atraso": 100.0,
        "pct_negativas_no_prazo": 0.0,
    }
    assert notes["marketplace"]["nota_media_com_atraso"] == 1.67  # r1, r10, r6
    assert notes["marketplace"]["pct_negativas_no_prazo"] == 20.0  # r7 em 5
    assert notes["diferenca_nota_com_atraso"] == -0.67
    assert notes["diferenca_nota_no_prazo"] == 0.67
    assert notes["diferenca_pct_negativas_com_atraso_pontos_percentuais"] == 33.3
    assert notes["diferenca_pct_negativas_no_prazo_pontos_percentuais"] == -20.0
    assert out["avaliacoes_de_pedidos_nao_entregues"] == 1  # r3


def test_atraso_dos_pedidos_usa_a_regra_das_avaliacoes():
    orders = sellers.build_seller_orders(_orders(), _items()).set_index("order_id")
    assert "o8" not in orders.index and len(orders) == 10
    assert orders.loc["o4", "dias_de_atraso"] == 0 and not orders.loc["o4", "entregue_com_atraso"]
    assert orders.loc["o9", "dias_de_atraso"] == 3 and orders.loc["o9", "entregue_com_atraso"]
    assert pd.isna(orders.loc["o3", "entregue_com_atraso"])
    reviews = _comments().set_index("order_id")
    for order_id in ("o1", "o2", "o4", "o6"):
        assert orders.loc[order_id, "dias_de_atraso"] == reviews.loc[order_id, "dias_de_atraso"]


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
    relation = out["relacao_volume_negativas"]
    assert relation["meses_usados"] == 6 and -1 <= relation["correlacao_spearman"] <= 1
    assert "correlação não mostra causa" == relation["cuidado"]
    trend = out["tendencia"]
    assert trend["ultimos_3_meses"]["meses"] == ["2017-05", "2017-06", "2017-07"]
    assert trend["ultimos_3_meses"]["pct_negativas"] == 12.5  # 1 de 8
    assert trend["3_meses_anteriores"]["pct_negativas"] == 80.0  # 4 de 5
    assert trend["diferenca_pct_negativas_pontos_percentuais"] == -67.5


def test_tendencia_usa_meses_de_calendario():
    """Um mês sem avaliações não some da janela de "últimos 3 meses"."""
    rows = []
    for month in ("2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-07"):
        rows.append(
            {
                "review_id": month,
                "seller_id": "A",
                "mes": month,
                "sentimento": "positiva",
                "review_score": 5,
            }
        )
    trend = sellers.monthly(pd.DataFrame(rows), "A")["tendencia"]
    assert trend["ultimos_3_meses"]["meses"] == ["2017-05", "2017-06", "2017-07"]
    assert trend["ultimos_3_meses"]["avaliacoes"] == 2
    assert trend["3_meses_anteriores"]["meses"] == ["2017-02", "2017-03", "2017-04"]


@pytest.mark.parametrize(
    "volumes, negatives, expected",
    [
        ([20, 30, 40, 50, 60, 70], [10, 20, 30, 40, 50, 60], "proporção maior de negativas"),
        ([20, 30, 40, 50, 60, 70], [60, 50, 40, 30, 20, 10], "proporção menor de negativas"),
        ([20, 30, 40, 50, 60, 70], [30, 60, 10, 50, 20, 40], "sem relação clara"),
    ],
)
def test_relacao_volume_negativas(volumes, negatives, expected):
    months = [
        {"avaliacoes": v, "pct_negativas": n} for v, n in zip(volumes, negatives, strict=True)
    ]
    assert expected in sellers.volume_relation(months)["leitura"]
    assert sellers.volume_relation(months[:5]) is None


def test_evolucao_mensal(df):
    out = sellers.monthly(df, "B")
    assert out["tendencia"] is None and out["mes_com_mais_negativas"] is None  # pouco dado
    assert out["relacao_volume_negativas"] is None
    meses = out["meses"]
    assert [m["mes"] for m in meses] == ["2017-01", "2017-03"]
    assert meses[1] == {"mes": "2017-03", "avaliacoes": 2, "pct_negativas": 50.0, "nota_media": 2.0}


def test_referencia_problemas_e_exemplos(df):
    ref_matrix = np.zeros((4, K), dtype=int)
    ref_matrix[[0, 1], 0] = 1  # atraso em 2 de 4
    ref_matrix[3, K - 1] = 1  # sem reclamação em 1 de 4
    ref = sellers.reference_from_matrix(ref_matrix, "referência de teste")
    assert ref["avaliacoes"] == 4 and ref["pct"]["atraso"] == 50.0
    assert ref["pct"]["sem_reclamacao"] == 25.0 and ref["descricao"] == "referência de teste"
    with pytest.raises(ValueError, match="inválida"):
        sellers.reference_from_matrix(np.zeros((0, K)), "vazia")

    neg = sellers.commented_negatives(df, "A")
    assert list(neg["review_id"]) == ["r1", "r3"]  # r10 é negativa, mas sem comentário
    matrix = np.zeros((len(neg), K), dtype=int)
    matrix[:, 0] = 1  # as duas com atraso
    matrix[1, 4] = 1  # uma também com defeito
    out = sellers.problems(neg, matrix, ref)
    first = out["categorias"][0]
    assert first["categoria"] == "atraso" and first["pct_das_negativas_com_comentario"] == 100.0
    assert "com comentário" in out["base"]
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
    assert top.loc["A", "negativas"] == 3 and top.loc["A", "pct_negativas"] == 50.0
    assert top.loc["A", "negativas_com_comentario"] == 2
    assert top.loc["B", "negativas_com_comentario"] == 1


def test_carrega_dos_arquivos(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _reviews().to_csv(raw / REVIEWS_FILE, index=False)
    _orders().to_csv(raw / ORDERS_FILE, index=False)
    _items().to_csv(raw / sellers.ORDER_ITEMS_FILE, index=False)
    _products().to_csv(raw / sellers.PRODUCTS_FILE, index=False)
    _sellers().to_csv(raw / sellers.SELLERS_FILE, index=False)
    parquet = tmp_path / "reviews.parquet"
    _comments().to_parquet(parquet)
    data = sellers.load_seller_data(parquet, raw)
    assert len(data.reviews) == 9 and set(data.reviews["seller_id"]) == {"A", "B"}
    assert int(data.reviews["tem_comentario"].sum()) == 7 and len(data.orders) == 10
    expected = seller_data()
    pd.testing.assert_frame_equal(
        data.reviews.drop(columns="texto"),
        expected.reviews.drop(columns="texto"),
        check_dtype=False,
    )
    with pytest.raises(FileNotFoundError, match="reviews.download"):
        sellers.load_seller_data(parquet, tmp_path / "vazio")
    with pytest.raises(FileNotFoundError, match="reviews.prepare"):
        sellers.load_seller_data(tmp_path / "nao_existe.parquet", raw)
