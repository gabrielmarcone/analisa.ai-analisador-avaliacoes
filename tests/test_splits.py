import numpy as np
import pandas as pd
import pytest

from reviews.splits import (
    SPLIT_NAMES,
    TEST,
    TRAIN,
    VAL,
    attach_splits,
    check_group_leakage,
    load_splits,
    make_splits,
    save_splits,
)


@pytest.fixture
def base() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 3000
    scores = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.05, 0.1, 0.2, 0.55])
    # ~20% dos textos se repetem, como "Recomendo" na base real.
    groups = [f"texto {i}" if rng.random() > 0.2 else f"comum {i % 30}" for i in range(n)]
    return pd.DataFrame(
        {"review_id": [f"r{i}" for i in range(n)], "review_score": scores, "texto_grupo": groups}
    )


def test_proporcoes_aproximadas(base):
    shares = make_splits(base)["split"].value_counts(normalize=True)
    assert shares[TRAIN] == pytest.approx(0.8, abs=0.03)
    assert shares[VAL] == pytest.approx(0.1, abs=0.03)
    assert shares[TEST] == pytest.approx(0.1, abs=0.03)


def test_sem_vazamento_de_textos_repetidos(base):
    df = base.merge(make_splits(base), on="review_id")
    assert check_group_leakage(df) == 0


def test_estratificacao_pela_nota(base):
    df = base.merge(make_splits(base), on="review_id")
    overall = df["review_score"].value_counts(normalize=True)
    test = df[df["split"] == TEST]["review_score"].value_counts(normalize=True)
    assert (overall - test).abs().max() < 0.05


def test_deterministico(base):
    pd.testing.assert_frame_equal(make_splits(base), make_splits(base))


def test_review_id_repetido_rejeitado(base):
    with pytest.raises(ValueError, match="repetido"):
        make_splits(pd.concat([base, base.iloc[[0]]]))


def test_salvar_e_carregar(base, tmp_path):
    path = tmp_path / "splits" / "s.csv"
    splits = make_splits(base)
    save_splits(splits, path)
    loaded = load_splits(path)
    assert set(loaded["split"]) == set(SPLIT_NAMES)
    assert loaded.set_index("review_id")["split"].equals(
        splits.set_index("review_id")["split"].loc[loaded["review_id"]]
    )
    assert not path.with_suffix(".tmp").exists()


def test_carregar_rejeita_split_desconhecido(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({"review_id": ["a"], "split": ["outro"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="desconhecidos"):
        load_splits(path)


def test_attach_exige_split_para_todas(base):
    splits = make_splits(base).iloc[1:]
    with pytest.raises(ValueError, match="sem split"):
        attach_splits(base, splits)
