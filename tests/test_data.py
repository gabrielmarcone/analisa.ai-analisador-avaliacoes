import pandas as pd
import pytest

from reviews.data import (
    build_text,
    clean_text,
    load_orders,
    load_reviews,
    normalize_for_grouping,
    prepare_reviews,
    validate_columns,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ótimo  produto,\r\nchegou rápido", "Ótimo produto, chegou rápido"),
        ("  espaços\tdemais  ", "espaços demais"),
        ("controle\x00invisível", "controle invisível"),
        (None, ""),
        (float("nan"), ""),
        (pd.NA, ""),
        ("   ", ""),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_unifica_unicode():
    # "é" pode vir como um caractere ou como "e" + acento combinado.
    assert clean_text("café") == clean_text("café")


def test_clean_text_preserva_maiusculas_e_pontuacao():
    assert clean_text("NÃO recomendo!!!") == "NÃO recomendo!!!"


@pytest.mark.parametrize(
    ("title", "message", "expected"),
    [
        ("", "texto", "texto"),
        ("Título", "", "Título"),
        ("Péssimo", "veio quebrado", "Péssimo. veio quebrado"),
        ("Atrasou!", "não chegou", "Atrasou! não chegou"),
    ],
)
def test_build_text(title, message, expected):
    assert build_text(title, message) == expected


def test_normalize_for_grouping_junta_variacoes():
    assert normalize_for_grouping("Ótimo produto!!") == normalize_for_grouping("otimo  PRODUTO")


def test_validate_columns_erro_claro():
    with pytest.raises(ValueError, match="colunas ausentes"):
        validate_columns(pd.DataFrame({"a": [1]}), ("a", "b"), "teste")


def test_load_sem_arquivo_orienta_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="reviews.download"):
        load_reviews(tmp_path)


def test_prepare_reviews(raw_dir):
    df = prepare_reviews(load_reviews(raw_dir), load_orders(raw_dir))

    # r3 (sem comentário) e r5 (só espaços) saem; r4 duplicado vira uma linha.
    assert df["review_id"].tolist() == ["r1", "r2", "r4"]
    assert df["review_id"].is_unique

    by_id = df.set_index("review_id")
    assert by_id.loc["r1", "texto"] == "Ótimo produto, chegou rápido"
    assert by_id.loc["r2", "texto"] == "Péssimo. veio quebrado"
    assert by_id.loc["r1", "sentimento"] == "positiva"
    assert by_id.loc["r2", "sentimento"] == "negativa"

    # r4 duplicado: fica a ocorrência mais antiga (pedido o4).
    assert by_id.loc["r4", "order_id"] == "o4"
    assert by_id.loc["r4", "dias_de_atraso"] == 10
    assert bool(by_id.loc["r4", "entregue_com_atraso"]) is True
    # Entregue no próprio dia do prazo não conta como atraso.
    assert bool(by_id.loc["r2", "entregue_com_atraso"]) is False


def test_pedido_nao_entregue_fica_sem_indicador(raw_reviews, raw_orders):
    raw_reviews.loc[5, "review_comment_message"] = "ainda não recebi"
    df = prepare_reviews(raw_reviews, raw_orders).set_index("review_id")
    assert pd.isna(df.loc["r5", "entregue_com_atraso"])
    assert pd.isna(df.loc["r5", "dias_de_atraso"])


def test_prepare_reviews_nao_altera_entrada(raw_reviews, raw_orders):
    before = raw_reviews.copy()
    prepare_reviews(raw_reviews, raw_orders)
    pd.testing.assert_frame_equal(raw_reviews, before)


def test_orders_duplicados_sao_rejeitados(raw_reviews, raw_orders):
    duplicated = pd.concat([raw_orders, raw_orders.iloc[[0]]])
    with pytest.raises(ValueError, match="order_id repetido"):
        prepare_reviews(raw_reviews, duplicated)
