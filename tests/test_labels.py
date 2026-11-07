import pandas as pd
import pytest

from reviews.labels import sentiment_from_score


def test_mapeamento_das_notas():
    result = sentiment_from_score(pd.Series([1, 2, 3, 4, 5]))
    assert result.tolist() == ["negativa", "negativa", "neutra", "positiva", "positiva"]


def test_aceita_notas_como_texto():
    assert sentiment_from_score(pd.Series(["1", "5"])).tolist() == ["negativa", "positiva"]


@pytest.mark.parametrize("bad", [0, 6, None, "abc", 4.5])
def test_nota_invalida_gera_erro(bad):
    with pytest.raises(ValueError, match="fora de 1-5"):
        sentiment_from_score(pd.Series([5, bad]))
