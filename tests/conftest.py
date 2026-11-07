"""Dados de teste no mesmo formato dos CSVs da Olist (pequenos e inventados)."""

from pathlib import Path

import pandas as pd
import pytest

from reviews.data import ORDERS_FILE, REVIEWS_FILE


@pytest.fixture
def raw_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r4", "r5"],
            "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
            "review_score": [5, 1, 3, 2, 2, 4],
            "review_comment_title": [None, "Péssimo", None, "Atrasou!", "Atrasou!", None],
            "review_comment_message": [
                "Ótimo  produto,\r\nchegou rápido",
                "veio quebrado",
                None,
                "não chegou no prazo",
                "não chegou no prazo",
                "   ",
            ],
            "review_creation_date": [
                "2018-01-10 00:00:00",
                "2018-01-11 00:00:00",
                "2018-01-12 00:00:00",
                "2018-01-13 00:00:00",
                "2018-01-14 00:00:00",
                "2018-01-15 00:00:00",
            ],
            "review_answer_timestamp": ["2018-01-11 10:00:00"] * 6,
        }
    )


@pytest.fixture
def raw_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
            "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "order_status": ["delivered"] * 5 + ["shipped"],
            "order_purchase_timestamp": ["2018-01-01 09:00:00"] * 6,
            "order_delivered_customer_date": [
                "2018-01-05 14:30:00",  # antes do prazo
                "2018-01-10 23:59:00",  # no dia do prazo -> não é atraso
                "2018-01-08 08:00:00",
                "2018-01-20 08:00:00",  # 10 dias depois do prazo
                "2018-01-20 08:00:00",
                None,  # não entregue
            ],
            "order_estimated_delivery_date": ["2018-01-10 00:00:00"] * 6,
        }
    )


@pytest.fixture
def raw_dir(tmp_path: Path, raw_reviews: pd.DataFrame, raw_orders: pd.DataFrame) -> Path:
    folder = tmp_path / "raw"
    folder.mkdir()
    raw_reviews.to_csv(folder / REVIEWS_FILE, index=False)
    raw_orders.to_csv(folder / ORDERS_FILE, index=False)
    return folder
