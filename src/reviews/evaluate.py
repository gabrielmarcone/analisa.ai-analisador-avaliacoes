"""Avaliação compartilhada pelos modelos do projeto.

As mesmas funções medem o baseline TF-IDF, o BERTimbau e o LLM, sempre no
mesmo conjunto. Cada medição é acumulada em ``reports/resultados.json``
(por tarefa, conjunto e modelo), e a tabela comparativa em markdown sai
desse arquivo. Assim nenhum modelo é avaliado com régua diferente.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from reviews.paths import RESULTS_FILE

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def evaluate_classification(y_true, y_pred, labels: Sequence[str]) -> dict:
    """Métricas de classe única (sentimento): F1 macro, acurácia e por classe."""
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} rótulos reais para {len(y_pred)} previstos.")
    unknown = set(y_pred) - set(labels)
    if unknown:
        raise ValueError(f"Previsões com classes desconhecidas: {sorted(map(str, unknown))}")

    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    one_hot = np.asarray(labels, dtype=object)
    return {
        "n": int(len(y_true)),
        "f1_macro": float(macro),
        "f1_macro_ic95": bootstrap_f1_macro(
            y_true[:, None] == one_hot[None, :], y_pred[:, None] == one_hot[None, :]
        ),
        "acuracia": float((y_true == y_pred).mean()),
        "por_classe": _per_class(labels, precision, recall, f1, support),
    }


def evaluate_multilabel(y_true, y_pred, labels: Sequence[str]) -> dict:
    """Métricas multirrótulo (tipos de problema), nas matrizes binárias n×k."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.shape != yp.shape:
        raise ValueError(f"Matrizes com formas diferentes: {yt.shape} e {yp.shape}.")
    if yt.ndim != 2 or yt.shape[1] != len(labels):
        raise ValueError(f"Esperava uma matriz n×{len(labels)}; recebi {yt.shape}.")

    precision, recall, f1, support = precision_recall_fscore_support(yt, yp, zero_division=0)
    union = (yt | yp).sum(axis=1)
    intersection = (yt & yp).sum(axis=1)
    jaccard = np.where(union == 0, 1.0, intersection / np.maximum(union, 1))
    return {
        "n": int(len(yt)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "f1_macro_ic95": bootstrap_f1_macro(yt, yp),
        "f1_micro": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "conjunto_identico": float((yt == yp).all(axis=1).mean()),
        "jaccard_medio": float(jaccard.mean()),
        "por_classe": _per_class(list(labels), precision, recall, f1, support),
    }


def _f1_macro_from_counts(tp, fp, fn) -> np.ndarray:
    """F1 macro a partir das contagens por classe (última dimensão); 0 quando 0/0."""
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(denominator.shape), where=denominator > 0)
    return f1.mean(axis=-1)


def bootstrap_f1_macro(
    truth, predicted, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> list[float]:
    """Intervalo de 95% do F1 macro por bootstrap (reamostra as avaliações com reposição).

    Recebe matrizes binárias n×k (uma coluna por classe). Semente fixa: o mesmo modelo
    no mesmo conjunto sempre dá o mesmo intervalo.
    """
    yt = np.asarray(truth, dtype=bool)
    yp = np.asarray(predicted, dtype=bool)
    rng = np.random.default_rng(seed)
    scores = np.empty(samples)
    for start in range(0, samples, 250):  # em blocos, para não estourar a memória
        idx = rng.integers(0, len(yt), size=(min(250, samples - start), len(yt)))
        t, p = yt[idx], yp[idx]
        tp = (t & p).sum(axis=1)
        fp = (~t & p).sum(axis=1)
        fn = (t & ~p).sum(axis=1)
        scores[start : start + len(idx)] = _f1_macro_from_counts(tp, fp, fn)
    low, high = np.percentile(scores, [2.5, 97.5])
    return [float(low), float(high)]


def _per_class(labels, precision, recall, f1, support) -> dict:
    return {
        label: {
            "precisao": float(p),
            "revocacao": float(r),
            "f1": float(f),
            "suporte": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }


def timed(predict: Callable[[], object]) -> tuple[object, float]:
    """Executa a predição e devolve (resultado, segundos gastos)."""
    start = time.perf_counter()
    result = predict()
    return result, time.perf_counter() - start


def latency_metrics(seconds: float, n_items: int) -> dict:
    if n_items <= 0:
        raise ValueError("n_items deve ser positivo.")
    return {
        "latencia_ms_por_avaliacao": float(seconds / n_items * 1000),
        "avaliacoes_por_segundo": float(n_items / seconds) if seconds > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Acúmulo de resultados e tabela comparativa
# ---------------------------------------------------------------------------


def load_results(path: Path = RESULTS_FILE) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_result(
    task: str, split: str, model: str, metrics: dict, path: Path = RESULTS_FILE
) -> dict:
    """Grava as métricas de um modelo em uma tarefa+conjunto e devolve tudo."""
    results = load_results(path)
    entry = dict(metrics)
    entry["registrado_em"] = datetime.now(UTC).isoformat(timespec="seconds")
    results.setdefault(task, {}).setdefault(split, {})[model] = entry

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",  # LF também no Windows, igual ao que o Git guarda
    )
    os.replace(tmp, path)
    return results


def fmt_br(value: float, decimals: int = 3) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def comparison_markdown(results: dict, task: str, split: str) -> str:
    """Tabela comparativa dos modelos de uma tarefa num conjunto."""
    models = results.get(task, {}).get(split, {})
    if not models:
        return f"_Sem resultados para {task} em {split}._"
    lines = [
        "| Modelo | F1 macro | Latência (ms/aval.) | Custo (US$/1.000) | n |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(models.items(), key=lambda kv: -kv[1].get("f1_macro", 0))
    for name, m in ordered:
        latency = m.get("latencia_ms_por_avaliacao")
        cost = m.get("custo_usd_por_1000")
        lines.append(
            f"| {m.get('modelo', name)} | {fmt_br(m['f1_macro'])} | "
            f"{fmt_br(latency, 2) if latency is not None else '–'} | "
            f"{fmt_br(cost, 3) if cost is not None else '–'} | {m['n']} |"
        )
    return "\n".join(lines)
