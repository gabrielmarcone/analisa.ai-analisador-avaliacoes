"""Baseline das duas tarefas: TF-IDF + Regressão Logística.

Regras de honestidade da avaliação:
- o modelo só vê texto do conjunto de **treino**;
- o hiperparâmetro C é escolhido pela **validação**;
- o **teste** é medido uma única vez, com ``--teste``, e nunca orienta ajustes.

Uso:
    python -m reviews.baseline sentimento [--teste]
    python -m reviews.baseline problemas [--teste]
    python -m reviews.baseline curva
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from reviews.annotation import LABELS as PROBLEM_LABELS
from reviews.annotation import NO_COMPLAINT, load_labels
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
from reviews.labels import BINARY_SENTIMENTS
from reviews.paths import (
    FIGURES_DIR,
    LABELS_FILE,
    MODELS_DIR,
    PROCESSED_REVIEWS_FILE,
    REPORTS_DIR,
)
from reviews.splits import TEST, TRAIN, VAL

MODEL_KEY = "baseline_tfidf"
MODEL_NAME = "TF-IDF + Regressão Logística"
SEED = 42
C_GRID = (0.25, 1.0, 4.0)
THRESHOLD = 0.5
CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
CURVE_SEEDS = (0, 1, 2)

TASK_SENTIMENT = "sentimento"
TASK_PROBLEMS = "problemas"

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------


def _load_processed(columns: Sequence[str]) -> pd.DataFrame:
    if not PROCESSED_REVIEWS_FILE.is_file():
        raise SystemExit("Base processada não encontrada. Rode `python -m reviews.prepare`.")
    return pd.read_parquet(PROCESSED_REVIEWS_FILE, columns=list(columns))


def sentiment_frame() -> pd.DataFrame:
    """Avaliações com sentimento binário (a nota 3, neutra, fica de fora)."""
    df = _load_processed(["review_id", "texto", "sentimento", "split"])
    return df[df["sentimento"].isin(BINARY_SENTIMENTS)].reset_index(drop=True)


def problem_frame() -> pd.DataFrame:
    """Avaliações anotadas com tipos de problema, com texto e split."""
    labels = load_labels(LABELS_FILE)
    if labels.empty:
        raise SystemExit("Nenhuma anotação encontrada. Anote com `python -m reviews.annotate_app`.")
    df = _load_processed(["review_id", "texto", "split"])
    merged = labels.merge(df, on="review_id", how="left", validate="one_to_one")
    if merged["texto"].isna().any():
        raise SystemExit("Há anotações sem texto na base. Rode `python -m reviews.prepare`.")
    return merged.reset_index(drop=True)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separa treino/validação/teste e garante que são disjuntos."""
    if df["review_id"].duplicated().any():
        raise ValueError("review_id repetido: a mesma avaliação apareceria em dois conjuntos.")
    parts = {split: df[df["split"] == split].reset_index(drop=True) for split in (TRAIN, VAL, TEST)}
    check_disjoint(parts[TRAIN]["review_id"], parts[VAL]["review_id"], TRAIN, VAL)
    check_disjoint(parts[TRAIN]["review_id"], parts[TEST]["review_id"], TRAIN, TEST)
    check_disjoint(parts[VAL]["review_id"], parts[TEST]["review_id"], VAL, TEST)
    return parts


def check_disjoint(ids_a, ids_b, name_a: str, name_b: str) -> None:
    """Falha alto se um conjunto vazar no outro. É a trava contra treinar com o teste."""
    overlap = set(ids_a) & set(ids_b)
    if overlap:
        raise ValueError(
            f"{len(overlap)} avaliações aparecem em '{name_a}' e em '{name_b}'. "
            "Avaliação inválida: refaça a divisão antes de treinar."
        )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def make_pipeline(task: str, c: float) -> Pipeline:
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    if task == TASK_PROBLEMS:
        classifier = OneVsRestClassifier(classifier)
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


def apply_rules(
    probabilities: np.ndarray,
    labels: Sequence[str] = PROBLEM_LABELS,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """Converte probabilidades multirrótulo em 0/1 seguindo o guia de anotação.

    - rótulo previsto quando a probabilidade passa do limiar;
    - se nenhum passa, fica o mais provável (toda avaliação tem ao menos um rótulo);
    - "sem reclamação" nunca acompanha outro rótulo.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError(f"Esperava probabilidades n×{len(labels)}; recebi {probabilities.shape}.")
    predictions = (probabilities >= threshold).astype(int)

    empty = predictions.sum(axis=1) == 0
    if empty.any():
        best = probabilities[empty].argmax(axis=1)
        predictions[np.flatnonzero(empty), best] = 1

    if NO_COMPLAINT in labels:
        no_complaint = list(labels).index(NO_COMPLAINT)
        conflicted = (predictions[:, no_complaint] == 1) & (predictions.sum(axis=1) > 1)
        predictions[conflicted, no_complaint] = 0
    return predictions


def _predictor(task: str, pipeline: Pipeline, texts: pd.Series):
    if task == TASK_SENTIMENT:
        return lambda: pipeline.predict(texts)
    return lambda: apply_rules(pipeline.predict_proba(texts))


def _evaluate(task: str, truth, predictions) -> dict:
    if task == TASK_SENTIMENT:
        return evaluate_classification(truth, predictions, labels=list(BINARY_SENTIMENTS))
    return evaluate_multilabel(truth, predictions, labels=list(PROBLEM_LABELS))


def _truth(task: str, frame: pd.DataFrame):
    if task == TASK_SENTIMENT:
        return frame["sentimento"]
    return frame[list(PROBLEM_LABELS)].to_numpy(dtype=int)


def select_c(task: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[float, dict]:
    """Treina um modelo por valor de C e escolhe o melhor F1 macro na validação."""
    scores: dict[float, float] = {}
    for c in C_GRID:
        pipeline = make_pipeline(task, c).fit(train["texto"], _truth(task, train))
        predictions = _predictor(task, pipeline, validation["texto"])()
        scores[c] = _evaluate(task, _truth(task, validation), predictions)["f1_macro"]
    best = max(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------------------
# Execução das tarefas
# ---------------------------------------------------------------------------


def run_task(task: str, include_test: bool) -> dict:
    frame = sentiment_frame() if task == TASK_SENTIMENT else problem_frame()
    parts = split_frames(frame)
    train, validation, test = parts[TRAIN], parts[VAL], parts[TEST]
    print(f"Tarefa: {task} · treino {len(train)} · validação {len(validation)} · teste {len(test)}")

    best_c, scores = select_c(task, train, validation)
    shown = ", ".join(f"C={c:g}: {fmt_br(f1)}" for c, f1 in scores.items())
    print(f"F1 macro na validação por C → {shown}\nMelhor C: {best_c:g}")

    pipeline = make_pipeline(task, best_c).fit(train["texto"], _truth(task, train))
    _save_model(task, pipeline)

    for split_name, part in [(VAL, validation)] + ([(TEST, test)] if include_test else []):
        predictions, seconds = timed(_predictor(task, pipeline, part["texto"]))
        metrics = _evaluate(task, _truth(task, part), predictions)
        metrics.update(latency_metrics(seconds, len(part)))
        metrics.update(
            modelo=MODEL_NAME,
            custo_usd_por_1000=0.0,
            hiperparametros={"C": best_c, "ngram": "1-2", "min_df": 2, "limiar": THRESHOLD},
        )
        save_result(task, split_name, MODEL_KEY, metrics)
        _print_metrics(task, split_name, metrics)

    write_results_markdown()
    return scores


def _save_model(task: str, pipeline: Pipeline) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"baseline_{task}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(pipeline, tmp)
    os.replace(tmp, path)
    print(f"Modelo salvo em {path}")


def _print_metrics(task: str, split_name: str, metrics: dict) -> None:
    print(f"\n== {task} · {split_name} ==")
    headline = f"F1 macro: {fmt_br(metrics['f1_macro'])}"
    if task == TASK_SENTIMENT:
        headline += f" · acurácia: {fmt_br(metrics['acuracia'])}"
    else:
        headline += (
            f" · conjunto idêntico: {fmt_br(metrics['conjunto_identico'])}"
            f" · Jaccard: {fmt_br(metrics['jaccard_medio'])}"
        )
    print(headline + f" · {fmt_br(metrics['latencia_ms_por_avaliacao'], 2)} ms/avaliação")
    table = pd.DataFrame(metrics["por_classe"]).T
    print(table.round(3).to_string())


def write_results_markdown() -> None:
    """Regenera reports/resultados.md a partir do JSON acumulado."""
    results = load_results()
    lines = [
        "# Resultados por modelo",
        "",
        "Gerado automaticamente a partir de `reports/resultados.json`;",
        "para atualizar, rode as avaliações de cada modelo.",
    ]
    for task in sorted(results):
        for split_name in (VAL, TEST):
            if split_name in results[task]:
                lines += [
                    "",
                    f"## {task} · {split_name}",
                    "",
                    comparison_markdown(results, task, split_name),
                ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "resultados.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nTabela atualizada em {out}")


# ---------------------------------------------------------------------------
# Curva de aprendizado
# ---------------------------------------------------------------------------


def _subsample(train: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    """Amostra do treino em que toda categoria mantém positivos e negativos."""
    for _ in range(200):
        sample = train.iloc[rng.permutation(len(train))[:size]]
        counts = sample[list(PROBLEM_LABELS)].sum()
        if counts.gt(0).all() and counts.lt(size).all():
            return sample
    raise RuntimeError("Não consegui amostrar com todas as categorias presentes.")


def learning_curve() -> pd.DataFrame:
    parts = split_frames(problem_frame())
    train, validation = parts[TRAIN], parts[VAL]
    truth_val = _truth(TASK_PROBLEMS, validation)

    rows = []
    for fraction in CURVE_FRACTIONS:
        size = round(len(train) * fraction)
        seeds = CURVE_SEEDS if fraction < 1.0 else (CURVE_SEEDS[0],)
        for seed in seeds:
            sample = _subsample(train, size, np.random.default_rng(seed))
            pipeline = make_pipeline(TASK_PROBLEMS, 1.0)
            pipeline.fit(sample["texto"], _truth(TASK_PROBLEMS, sample))
            predictions = apply_rules(pipeline.predict_proba(validation["texto"]))
            f1 = evaluate_multilabel(truth_val, predictions, list(PROBLEM_LABELS))["f1_macro"]
            rows.append({"avaliacoes_treino": size, "semente": seed, "f1_macro_validacao": f1})
            print(f"treino={size:>4} semente={seed} → F1 macro validação {fmt_br(f1)}")

    curve = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(REPORTS_DIR / "curva_aprendizado.csv", index=False, lineterminator="\n")
    _plot_curve(curve)
    return curve


def _plot_curve(curve: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = curve.groupby("avaliacoes_treino")["f1_macro_validacao"]
    sizes = list(grouped.mean().index)
    mean, low, high = grouped.mean(), grouped.min(), grouped.max()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.fill_between(sizes, low, high, color="#2a78d6", alpha=0.15, linewidth=0)
    ax.plot(sizes, mean, color="#2a78d6", linewidth=2, marker="o", markersize=5)
    for x, y in zip(sizes, mean, strict=True):
        ax.annotate(
            fmt_br(y, 2),
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#3a3935",
        )
    ax.set_xlabel("Avaliações de treino anotadas")
    ax.set_ylabel("F1 macro (validação)")
    ax.set_title("Baseline TF-IDF: quanto mais anotação, melhor o modelo?")
    ax.set_ylim(0, min(1.0, float(high.max()) + 0.1))
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "curva_aprendizado.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Gráfico salvo em {out}")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + Regressão Logística.")
    sub = parser.add_subparsers(dest="comando", required=True)
    for task in (TASK_SENTIMENT, TASK_PROBLEMS):
        cmd = sub.add_parser(task, help=f"treina e avalia a tarefa de {task}")
        cmd.add_argument(
            "--teste",
            action="store_true",
            help="além da validação, mede no teste (uma vez, ao final)",
        )
    sub.add_parser("curva", help="curva de aprendizado da tarefa de problemas")
    args = parser.parse_args(argv)

    if args.comando == "curva":
        learning_curve()
    else:
        run_task(args.comando, include_test=args.teste)


if __name__ == "__main__":
    main()
