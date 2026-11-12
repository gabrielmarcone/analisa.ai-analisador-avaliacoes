import json

import pytest

from reviews.evaluate import (
    comparison_markdown,
    evaluate_classification,
    evaluate_multilabel,
    fmt_br,
    latency_metrics,
    load_results,
    save_result,
    timed,
)


def test_classificacao_com_valores_conhecidos():
    # a: precisão 1, revocação 1/2, F1 2/3 · b: precisão 2/3, revocação 1, F1 4/5
    result = evaluate_classification(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert result["n"] == 4
    assert result["acuracia"] == pytest.approx(0.75)
    assert result["f1_macro"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert result["por_classe"]["a"]["f1"] == pytest.approx(2 / 3)
    assert result["por_classe"]["b"]["suporte"] == 2


def test_classificacao_rejeita_entradas_invalidas():
    with pytest.raises(ValueError, match="previstos"):
        evaluate_classification(["a"], ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="desconhecidas"):
        evaluate_classification(["a"], ["c"], ["a", "b"])


def test_multirrotulo_com_valores_conhecidos():
    y_true = [[1, 0], [1, 1], [0, 1]]
    y_pred = [[1, 0], [1, 0], [0, 1]]
    result = evaluate_multilabel(y_true, y_pred, ["x", "y"])
    assert result["f1_macro"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["conjunto_identico"] == pytest.approx(2 / 3)
    assert result["jaccard_medio"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert result["por_classe"]["y"]["revocacao"] == pytest.approx(0.5)


def test_multirrotulo_rejeita_formas_erradas():
    with pytest.raises(ValueError, match="formas diferentes"):
        evaluate_multilabel([[1, 0]], [[1, 0], [0, 1]], ["x", "y"])
    with pytest.raises(ValueError, match="n×3"):
        evaluate_multilabel([[1, 0]], [[1, 0]], ["x", "y", "z"])


def test_timed_e_latencia():
    value, seconds = timed(lambda: "ok")
    assert value == "ok" and seconds >= 0
    metrics = latency_metrics(2.0, 1000)
    assert metrics["latencia_ms_por_avaliacao"] == pytest.approx(2.0)
    assert metrics["avaliacoes_por_segundo"] == pytest.approx(500.0)
    with pytest.raises(ValueError):
        latency_metrics(1.0, 0)


def test_resultados_acumulam_e_viram_tabela(tmp_path):
    path = tmp_path / "resultados.json"
    save_result(
        "problemas",
        "teste",
        "baseline",
        {"modelo": "Baseline", "f1_macro": 0.5, "n": 10},
        path=path,
    )
    results = save_result(
        "problemas",
        "teste",
        "llm",
        {
            "modelo": "LLM zero-shot",
            "f1_macro": 0.7,
            "latencia_ms_por_avaliacao": 400.0,
            "custo_usd_por_1000": 1.25,
            "n": 10,
        },
        path=path,
    )
    assert set(results["problemas"]["teste"]) == {"baseline", "llm"}
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == load_results(path)
    assert b"\r\n" not in path.read_bytes()  # LF mesmo no Windows

    table = comparison_markdown(results, "problemas", "teste")
    rows = table.splitlines()
    assert rows[2].startswith("| LLM zero-shot | 0,700 ")  # maior F1 primeiro
    assert "| Baseline | 0,500 | – | – | 10 |" in table
    assert "| 1,250 | 10 |" in table  # custo com 3 casas: US$ 0,014 não vira 0,01

    assert "Sem resultados" in comparison_markdown(results, "sentimento", "teste")


def test_fmt_br():
    assert fmt_br(0.8571) == "0,857"
    assert fmt_br(12.5, 2) == "12,50"
