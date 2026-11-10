import numpy as np
import pandas as pd
import pytest

from reviews.annotation import LABELS, NO_COMPLAINT
from reviews.baseline import (
    TASK_PROBLEMS,
    TASK_SENTIMENT,
    _subsample,
    apply_rules,
    check_disjoint,
    make_pipeline,
    split_frames,
)

SEM = list(LABELS).index(NO_COMPLAINT)
K = len(LABELS)


def test_apply_rules_limiar_e_linha_vazia():
    proba = np.full((2, K), 0.1)
    proba[0, 0] = 0.9  # acima do limiar
    proba[1, 2] = 0.4  # nenhum acima: fica o argmax
    pred = apply_rules(proba)
    assert pred[0].tolist() == [1] + [0] * (K - 1)
    assert pred[1, 2] == 1 and pred[1].sum() == 1


def test_apply_rules_sem_reclamacao_nunca_acompanha_outro():
    proba = np.full((1, K), 0.05)
    proba[0, 0] = 0.8
    proba[0, SEM] = 0.9
    pred = apply_rules(proba)
    assert pred[0, SEM] == 0 and pred[0, 0] == 1

    proba_only = np.full((1, K), 0.05)
    proba_only[0, SEM] = 0.9
    assert apply_rules(proba_only)[0, SEM] == 1


def test_check_disjoint():
    check_disjoint(["a", "b"], ["c"], "treino", "teste")
    with pytest.raises(ValueError, match="treino.*teste"):
        check_disjoint(["a", "b"], ["b"], "treino", "teste")


def test_split_frames_rejeita_id_repetido():
    df = pd.DataFrame({"review_id": ["a", "a"], "split": ["treino", "teste"], "texto": ["x", "y"]})
    with pytest.raises(ValueError, match="repetido"):
        split_frames(df)


def _frase(rng, words, n=6):
    return " ".join(rng.choice(words, n))


def test_pipeline_sentimento_aprende_separacao_simples():
    rng = np.random.default_rng(0)
    bons = [f"{_frase(rng, ['entrega', 'rápida', 'perfeito', 'recomendo'])}" for _ in range(40)]
    ruins = [f"{_frase(rng, ['quebrado', 'péssimo', 'defeito', 'atraso'])}" for _ in range(40)]
    texts = pd.Series(bons + ruins)
    y = pd.Series(["positiva"] * 40 + ["negativa"] * 40)
    pipeline = make_pipeline(TASK_SENTIMENT, 1.0).fit(texts, y)
    pred = pipeline.predict(pd.Series(["chegou rápida recomendo", "veio quebrado péssimo"]))
    assert pred.tolist() == ["positiva", "negativa"]


def test_pipeline_problemas_multirrotulo():
    rng = np.random.default_rng(1)
    atraso = [f"{_frase(rng, ['atrasou', 'demorou', 'prazo', 'estourado'])}" for _ in range(30)]
    quebrado = [f"{_frase(rng, ['quebrado', 'defeito', 'trincado', 'rachado'])}" for _ in range(30)]
    ambos = [f"{a} {q}" for a, q in zip(atraso[:10], quebrado[:10], strict=True)]
    texts = pd.Series(atraso + quebrado + ambos)

    # Matriz com duas categorias; nos dados reais todas têm positivos no treino.
    y = np.zeros((70, 2), dtype=int)
    y[:30, 0] = 1  # atraso
    y[30:60, 1] = 1  # defeito
    y[60:, :] = 1  # os dois juntos
    pipeline = make_pipeline(TASK_PROBLEMS, 1.0).fit(texts, y)
    proba = pipeline.predict_proba(pd.Series(["atrasou e veio quebrado"]))
    pred = apply_rules(proba, labels=["atraso", "defeito"])
    assert pred[0].tolist() == [1, 1]


def test_subsample_mantem_todas_as_categorias():
    rng = np.random.default_rng(2)
    n = 60
    base = pd.DataFrame({"texto": [f"t{i}" for i in range(n)]})
    for j, label in enumerate(LABELS):
        base[label] = 0
        base.loc[(np.arange(n) % K) == j, label] = 1
    sample = _subsample(base, 20, rng)
    assert len(sample) == 20
    counts = sample[list(LABELS)].sum()
    assert counts.gt(0).all() and counts.lt(20).all()
