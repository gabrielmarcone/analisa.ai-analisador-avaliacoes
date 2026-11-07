from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviews.data import ORDERS_FILE, REVIEWS_FILE
from reviews.prepare import run


@pytest.fixture
def big_raw_dir(tmp_path: Path) -> Path:
    """300 avaliações inventadas, suficientes para a divisão 80/10/10."""
    rng = np.random.default_rng(1)
    n = 300
    ids = [f"o{i}" for i in range(n)]
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "order_id": ids,
            "review_score": rng.choice([1, 2, 3, 4, 5], size=n),
            "review_comment_title": None,
            "review_comment_message": [
                f"comentário número {i}" if i % 4 else "Recomendo" for i in range(n)
            ],
            "review_creation_date": "2018-02-01 00:00:00",
            "review_answer_timestamp": "2018-02-02 00:00:00",
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ids,
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-01 00:00:00",
            "order_delivered_customer_date": "2018-01-15 00:00:00",
            "order_estimated_delivery_date": "2018-01-20 00:00:00",
        }
    )
    folder = tmp_path / "raw"
    folder.mkdir()
    reviews.to_csv(folder / REVIEWS_FILE, index=False)
    orders.to_csv(folder / ORDERS_FILE, index=False)
    return folder


def test_pipeline_completo_e_reprodutivel(big_raw_dir, tmp_path):
    output = tmp_path / "processed" / "reviews.parquet"
    splits_file = tmp_path / "splits.csv"

    first = run(raw_dir=big_raw_dir, output=output, splits_file=splits_file)
    assert output.is_file() and splits_file.is_file()
    assert not output.with_suffix(".tmp").exists()

    saved = pd.read_parquet(output)
    assert len(saved) == 300
    assert saved["split"].notna().all()
    # Todos os "Recomendo" ficam no mesmo split.
    assert saved.loc[saved["texto"] == "Recomendo", "split"].nunique() == 1

    # Segunda execução reaproveita o split salvo: mesmo resultado, arquivo intacto.
    splits_before = splits_file.read_bytes()
    second = run(raw_dir=big_raw_dir, output=output, splits_file=splits_file)
    assert splits_file.read_bytes() == splits_before
    pd.testing.assert_frame_equal(first, second)


def test_base_pequena_demais_gera_erro_claro(raw_dir, tmp_path):
    with pytest.raises(ValueError, match="Poucos textos distintos"):
        run(raw_dir=raw_dir, output=tmp_path / "x.parquet", splits_file=tmp_path / "s.csv")
