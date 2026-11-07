from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from reviews.annotation import (
    DEFAULT_SIZES,
    LABELS,
    NO_COMPLAINT,
    PILOT,
    agreement_markdown,
    agreement_report,
    build_agreement_queue,
    build_queue,
    clean_note,
    empty_labels,
    load_labels,
    load_queue,
    progress,
    save_labels,
    save_queue,
    upsert_label,
    validate_selection,
)


@pytest.fixture
def reviews() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for split, n in (("treino", 900), ("validacao", 300), ("teste", 300)):
        for i in range(n):
            rows.append(
                {
                    "review_id": f"{split}-{i}",
                    "split": split,
                    "sentimento": rng.choice(["negativa", "positiva", "neutra"], p=[0.6, 0.3, 0.1]),
                }
            )
    return pd.DataFrame(rows)


SMALL = {PILOT: 20, "teste": 50, "validacao": 30, "treino": 60}


def test_fila_ordem_tamanhos_e_splits(reviews):
    queue = build_queue(reviews, SMALL)
    assert queue["etapa"].drop_duplicates().tolist() == ["piloto", "teste", "validacao", "treino"]
    assert queue["etapa"].value_counts().to_dict() == SMALL
    assert queue["ordem"].tolist() == list(range(1, 161))
    assert queue["review_id"].is_unique
    # Piloto vem do treino; cada etapa vem do split certo.
    assert set(queue.loc[queue["etapa"] == PILOT, "split"]) == {"treino"}
    assert (queue["etapa"].replace({PILOT: "treino"}) == queue["split"]).all()


def test_fila_so_negativas(reviews):
    queue = build_queue(reviews, SMALL)
    negativas = set(reviews.loc[reviews["sentimento"] == "negativa", "review_id"])
    assert set(queue["review_id"]) <= negativas


def test_fila_deterministica_e_independente_da_ordem(reviews):
    a = build_queue(reviews, SMALL)
    b = build_queue(reviews.sample(frac=1, random_state=9), SMALL)
    pd.testing.assert_frame_equal(a, b)


def test_fila_falta_de_dados(reviews):
    with pytest.raises(ValueError, match="só há"):
        build_queue(reviews, {"teste": 10_000})


def test_tamanhos_padrao_cabem_na_base_real():
    # Na base real há 1.076 negativas no teste e na validação e 8.606 no treino.
    assert DEFAULT_SIZES["teste"] <= 1076 and DEFAULT_SIZES["validacao"] <= 1076
    assert DEFAULT_SIZES[PILOT] + DEFAULT_SIZES["treino"] <= 8606


def test_salvar_e_carregar_fila(reviews, tmp_path):
    queue = build_queue(reviews, SMALL)
    path = tmp_path / "fila.csv"
    save_queue(queue, path)
    pd.testing.assert_frame_equal(load_queue(path), queue, check_dtype=False)


def test_fila_adulterada_e_rejeitada(reviews, tmp_path):
    queue = build_queue(reviews, SMALL)
    queue.loc[queue["etapa"] == "teste", "split"] = "treino"
    with pytest.raises(ValueError, match="não bate com o split"):
        save_queue(queue, tmp_path / "fila.csv")


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (["outro", "atraso"], ["atraso", "outro"]),
        (["sem_reclamacao"], ["sem_reclamacao"]),
    ],
)
def test_validate_selection_ok(selected, expected):
    assert validate_selection(selected) == expected


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ([], "pelo menos um"),
        (None, "pelo menos um"),
        (["sem_reclamacao", "atraso"], "não pode ir junto"),
        (["inventado"], "desconhecidos"),
    ],
)
def test_validate_selection_erros(selected, message):
    with pytest.raises(ValueError, match=message):
        validate_selection(selected)


def test_clean_note():
    assert clean_note("  linha 1\nlinha 2  ") == "linha 1 linha 2"
    assert len(clean_note("x" * 1000)) == 300
    assert clean_note(None) == ""


def test_upsert_insere_e_substitui():
    t = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    labels = upsert_label(empty_labels(), "a", ["atraso", "nao_recebido"], now=t)
    labels = upsert_label(labels, "b", ["sem_reclamacao"], revisar=True, now=t)
    labels = upsert_label(labels, "a", ["incompleto"], observacao="mudei", now=t)

    assert len(labels) == 2
    a = labels.set_index("review_id").loc["a"]
    assert [label for label in LABELS if a[label] == 1] == ["incompleto"]
    assert a["observacao"] == "mudei"
    assert a["anotado_em"] == "2026-11-07T12:00:00+00:00"
    assert labels.set_index("review_id").loc["b", "revisar"] == 1


def test_upsert_invalido_nao_altera(tmp_path):
    labels = upsert_label(empty_labels(), "a", ["atraso"])
    with pytest.raises(ValueError):
        upsert_label(labels, "a", [])
    assert len(labels) == 1


def test_salvar_carregar_rotulos(tmp_path):
    path = tmp_path / "rotulos.csv"
    assert load_labels(path).empty  # arquivo ainda não existe
    labels = upsert_label(empty_labels(), "a", ["atraso"], observacao="ação, “aspas”; ok")
    labels = upsert_label(labels, "b", ["outro", "atendimento"])
    save_labels(labels, path)
    loaded = load_labels(path)
    pd.testing.assert_frame_equal(loaded, labels.reset_index(drop=True))
    assert not path.with_suffix(".tmp").exists()
    assert b"\r\n" not in path.read_bytes()


def test_rotulos_corrompidos_sao_rejeitados(tmp_path):
    path = tmp_path / "rotulos.csv"
    labels = upsert_label(empty_labels(), "a", ["atraso"])
    bad = labels.copy()
    bad[NO_COMPLAINT] = 1  # sem_reclamacao junto com atraso
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="junto com outro"):
        load_labels(path)


def test_progresso(reviews):
    queue = build_queue(reviews, SMALL)
    first = queue["review_id"].iloc[:3]
    labels = empty_labels()
    for i, rid in enumerate(first):
        labels = upsert_label(labels, rid, ["atraso"], revisar=(i == 0))
    table = progress(queue, labels)
    assert table.loc[PILOT].tolist() == [20, 3, 1]
    assert table.loc["total"].tolist() == [160, 3, 1]


def test_fila_de_concordancia(reviews):
    queue = build_queue(reviews, SMALL)
    labels = empty_labels()
    for rid in queue["review_id"]:
        labels = upsert_label(labels, rid, ["atraso"])
    agreement = build_agreement_queue(queue, labels, size=40)
    assert len(agreement) == 40 and agreement["review_id"].is_unique
    assert PILOT not in set(agreement["etapa"])
    with pytest.raises(ValueError, match="Anote mais"):
        build_agreement_queue(queue, labels.iloc[:30], size=40)


def _labels_from(rows):
    labels = empty_labels()
    for rid, selected in rows:
        labels = upsert_label(labels, rid, selected)
    return labels


def test_concordancia_perfeita_e_parcial():
    first = _labels_from([("a", ["atraso"]), ("b", ["incompleto"]), ("c", ["atraso", "outro"])])
    table, summary = agreement_report(first, first)
    by_cat = table.set_index("categoria")
    assert by_cat.loc["Atraso na entrega", "kappa"] == pytest.approx(1.0)
    assert np.isnan(by_cat.loc["Não recebido", "kappa"])  # nunca marcado
    assert summary["conjunto_identico"] == 1.0

    second = _labels_from([("a", ["atraso"]), ("b", ["atraso"]), ("c", ["atraso"])])
    table, summary = agreement_report(first, second)
    assert summary["avaliacoes"] == 3
    assert summary["conjunto_identico"] == pytest.approx(1 / 3)
    assert 0 < summary["jaccard_medio"] < 1
    md = agreement_markdown(table, summary)
    assert "| Atraso na entrega |" in md and "0." not in md.split("Jaccard")[1]


def test_concordancia_sem_itens_em_comum():
    with pytest.raises(ValueError, match="em comum"):
        agreement_report(_labels_from([("a", ["atraso"])]), _labels_from([("b", ["atraso"])]))
