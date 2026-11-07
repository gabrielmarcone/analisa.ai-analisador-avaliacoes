"""Anotação dos tipos de problema (multirrótulo).

Arquivos (em annotations/, versionados no Git, só com IDs e rótulos):
    fila_v1.csv                    ordem de anotação: piloto, teste, validação, treino
    rotulos_v1.csv                 rótulos da 1ª rodada
    fila_concordancia_v1.csv       amostra reanotada para medir concordância
    rotulos_concordancia_v1.csv    rótulos da 2ª rodada

Uso:
    python -m reviews.annotation fila           # cria a fila (uma vez)
    python -m reviews.annotation status         # progresso
    python -m reviews.annotation concordancia   # kappa entre as rodadas
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from reviews.labels import NEGATIVE
from reviews.paths import (
    AGREEMENT_LABELS_FILE,
    AGREEMENT_QUEUE_FILE,
    LABELS_FILE,
    PROCESSED_REVIEWS_FILE,
    QUEUE_FILE,
    REPORTS_DIR,
)
from reviews.splits import TEST, TRAIN, VAL

# ---------------------------------------------------------------------------
# Categorias
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "atraso": "Atraso na entrega",
    "nao_recebido": "Não recebido",
    "incompleto": "Pedido incompleto",
    "produto_diferente": "Produto errado ou diferente",
    "defeito_qualidade": "Defeito, avaria ou má qualidade",
    "atendimento": "Atendimento, cancelamento ou reembolso",
    "outro": "Outro problema",
    "sem_reclamacao": "Sem reclamação no texto",
}
LABELS = tuple(CATEGORIES)
NO_COMPLAINT = "sem_reclamacao"
PROBLEM_LABELS = tuple(label for label in LABELS if label != NO_COMPLAINT)

LABEL_COLUMNS = ("review_id", *LABELS, "revisar", "observacao", "anotado_em")
MAX_NOTE_LENGTH = 300

# ---------------------------------------------------------------------------
# Fila de anotação
# ---------------------------------------------------------------------------

PILOT = "piloto"
STAGE_SPLIT = {PILOT: TRAIN, TEST: TEST, VAL: VAL, TRAIN: TRAIN}
STAGES = tuple(STAGE_SPLIT)
DEFAULT_SIZES: dict[str, int] = {PILOT: 100, TEST: 500, VAL: 250, TRAIN: 650}
AGREEMENT_SIZE = 100
SEED = 42
QUEUE_COLUMNS = ("ordem", "review_id", "split", "etapa")


def build_queue(
    reviews: pd.DataFrame, sizes: Mapping[str, int] = DEFAULT_SIZES, seed: int = SEED
) -> pd.DataFrame:
    """Sorteia as avaliações negativas a anotar, na ordem piloto → teste → validação → treino.

    O piloto sai do treino, para que ajustes no guia não contaminem o teste.
    """
    unknown = set(sizes) - set(STAGES)
    if unknown:
        raise ValueError(f"Etapas desconhecidas: {sorted(unknown)}")

    negatives = reviews.loc[reviews["sentimento"] == NEGATIVE, ["review_id", "split"]]
    negatives = negatives.sort_values("review_id", kind="mergesort")
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    parts = []
    for stage in STAGES:
        size = int(sizes.get(stage, 0))
        if size <= 0:
            continue
        pool = negatives[
            (negatives["split"] == STAGE_SPLIT[stage]) & ~negatives["review_id"].isin(used)
        ]
        if len(pool) < size:
            raise ValueError(
                f"Etapa '{stage}' pede {size} avaliações, mas só há {len(pool)} disponíveis."
            )
        chosen = pool.iloc[rng.permutation(len(pool))[:size]].assign(etapa=stage)
        used.update(chosen["review_id"])
        parts.append(chosen)

    queue = pd.concat(parts, ignore_index=True)
    queue.insert(0, "ordem", np.arange(1, len(queue) + 1))
    return queue.loc[:, list(QUEUE_COLUMNS)].astype(
        {"review_id": "string", "split": "string", "etapa": "string"}
    )


def build_agreement_queue(
    queue: pd.DataFrame, labels: pd.DataFrame, size: int = AGREEMENT_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """Sorteia avaliações já anotadas (fora do piloto) para a 2ª rodada."""
    done = set(labels["review_id"])
    pool = queue[(queue["etapa"] != PILOT) & queue["review_id"].isin(done)]
    pool = pool.sort_values("ordem", kind="mergesort")
    if len(pool) < size:
        raise ValueError(
            f"A concordância pede {size} avaliações já anotadas fora do piloto; "
            f"há {len(pool)}. Anote mais antes da 2ª rodada."
        )
    rng = np.random.default_rng(seed + 1)
    chosen = pool.iloc[rng.permutation(len(pool))[:size]].copy()
    chosen["ordem"] = np.arange(1, size + 1)
    return chosen.loc[:, list(QUEUE_COLUMNS)].reset_index(drop=True)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    try:
        os.replace(tmp, path)
    except PermissionError as err:
        tmp.unlink(missing_ok=True)
        raise PermissionError(
            f"Não foi possível gravar {path.name}. Ele está aberto em outro programa "
            "(Excel, por exemplo)? Feche e tente de novo."
        ) from err


def save_queue(queue: pd.DataFrame, path: Path = QUEUE_FILE) -> None:
    _validate_queue(queue, Path(path).name)
    _write_csv_atomic(queue, path)


def load_queue(path: Path = QUEUE_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe. Rode `python -m reviews.annotation fila`.")
    queue = pd.read_csv(path, dtype={"review_id": "string", "split": "string", "etapa": "string"})
    _validate_queue(queue, path.name)
    return queue.sort_values("ordem", kind="mergesort").reset_index(drop=True)


def _validate_queue(queue: pd.DataFrame, name: str) -> None:
    missing = [col for col in QUEUE_COLUMNS if col not in queue.columns]
    if missing:
        raise ValueError(f"{name}: colunas ausentes {missing}")
    if queue["review_id"].duplicated().any():
        raise ValueError(f"{name}: review_id repetido na fila.")
    unknown = set(queue["etapa"]) - set(STAGES)
    if unknown:
        raise ValueError(f"{name}: etapas desconhecidas {sorted(unknown)}")
    expected_split = queue["etapa"].map(STAGE_SPLIT)
    if (expected_split != queue["split"]).any():
        raise ValueError(f"{name}: há avaliações numa etapa que não bate com o split.")


# ---------------------------------------------------------------------------
# Rótulos
# ---------------------------------------------------------------------------


def validate_selection(selected: Iterable[str]) -> list[str]:
    """Confere uma seleção de rótulos e devolve na ordem oficial das categorias."""
    chosen = set(selected or [])
    unknown = chosen - set(LABELS)
    if unknown:
        raise ValueError(f"Rótulos desconhecidos: {sorted(unknown)}")
    if not chosen:
        raise ValueError("Marque pelo menos um rótulo.")
    if NO_COMPLAINT in chosen and len(chosen) > 1:
        raise ValueError(f"'{CATEGORIES[NO_COMPLAINT]}' não pode ir junto com outro rótulo.")
    return [label for label in LABELS if label in chosen]


def clean_note(note: object) -> str:
    """Observação em uma linha, sem espaços extras, com tamanho limitado."""
    if note is None or (not isinstance(note, str) and pd.isna(note)):
        return ""
    return " ".join(str(note).split())[:MAX_NOTE_LENGTH]


def empty_labels() -> pd.DataFrame:
    df = pd.DataFrame({col: pd.Series(dtype="object") for col in LABEL_COLUMNS})
    return _coerce_label_types(df)


def _coerce_label_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_id"] = df["review_id"].astype("string")
    for col in (*LABELS, "revisar"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int8")
    df["observacao"] = df["observacao"].map(clean_note).astype("string")
    df["anotado_em"] = df["anotado_em"].fillna("").astype("string")
    return df


def validate_labels(df: pd.DataFrame, name: str = "rótulos") -> None:
    missing = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: colunas ausentes {missing}")
    if df["review_id"].duplicated().any():
        raise ValueError(f"{name}: review_id repetido.")
    values = df[[*LABELS, "revisar"]]
    if not values.isin([0, 1]).all().all():
        raise ValueError(f"{name}: os rótulos só podem ser 0 ou 1.")
    totals = df[list(LABELS)].sum(axis=1)
    if (totals == 0).any():
        raise ValueError(f"{name}: há avaliações sem nenhum rótulo.")
    if ((df[NO_COMPLAINT] == 1) & (totals > 1)).any():
        raise ValueError(f"{name}: '{NO_COMPLAINT}' aparece junto com outro rótulo.")


def load_labels(path: Path = LABELS_FILE) -> pd.DataFrame:
    """Lê os rótulos. Se o arquivo não existe ainda, devolve uma tabela vazia."""
    path = Path(path)
    if not path.is_file():
        return empty_labels()
    df = pd.read_csv(path, dtype={"review_id": "string"}, keep_default_na=False)
    validate_labels(df, path.name)
    return _coerce_label_types(df)


def save_labels(df: pd.DataFrame, path: Path = LABELS_FILE) -> None:
    validate_labels(df, Path(path).name)
    _write_csv_atomic(df.loc[:, list(LABEL_COLUMNS)], path)


def upsert_label(
    labels: pd.DataFrame,
    review_id: str,
    selected: Iterable[str],
    revisar: bool = False,
    observacao: str = "",
    now: datetime | None = None,
) -> pd.DataFrame:
    """Grava (ou substitui) os rótulos de uma avaliação e devolve a nova tabela."""
    chosen = set(validate_selection(selected))
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    row = {"review_id": review_id}
    row.update({label: int(label in chosen) for label in LABELS})
    row.update({"revisar": int(bool(revisar)), "observacao": observacao, "anotado_em": timestamp})

    rest = labels[labels["review_id"] != review_id]
    new = _coerce_label_types(pd.DataFrame([row], columns=list(LABEL_COLUMNS)))
    if rest.empty:
        return new.reset_index(drop=True)
    return pd.concat([rest, new], ignore_index=True)


def selection_of(row: Mapping[str, object]) -> list[str]:
    return [label for label in LABELS if int(row[label]) == 1]


# ---------------------------------------------------------------------------
# Progresso e concordância
# ---------------------------------------------------------------------------


def progress(queue: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Quantas avaliações já foram anotadas e marcadas para revisar, por etapa."""
    merged = queue.merge(labels[["review_id", "revisar"]], on="review_id", how="left")
    merged["anotadas"] = merged["revisar"].notna()
    merged["a_revisar"] = merged["revisar"].fillna(0).astype(int)
    table = merged.groupby("etapa", sort=False).agg(
        total=("review_id", "size"), anotadas=("anotadas", "sum"), a_revisar=("a_revisar", "sum")
    )
    order = [stage for stage in queue["etapa"].drop_duplicates()]
    table = table.reindex(order)
    table.loc["total"] = table.sum()
    return table.astype(int)


def agreement_report(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compara duas rodadas nas avaliações em comum.

    Por categoria: concordância simples e kappa de Cohen (que desconta o acaso).
    Kappa fica vazio quando a categoria não aparece em nenhuma das rodadas.
    """
    merged = first.merge(second, on="review_id", suffixes=("_1", "_2"))
    if merged.empty:
        raise ValueError("As duas rodadas não têm avaliações em comum.")

    rows = []
    for label in LABELS:
        a, b = merged[f"{label}_1"].astype(int), merged[f"{label}_2"].astype(int)
        both_constant = a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]
        kappa = np.nan if both_constant else cohen_kappa_score(a, b)
        rows.append(
            {
                "categoria": CATEGORIES[label],
                "rodada_1": int(a.sum()),
                "rodada_2": int(b.sum()),
                "concordancia": float((a == b).mean()),
                "kappa": float(kappa),
            }
        )

    m1 = merged[[f"{label}_1" for label in LABELS]].to_numpy(dtype=bool)
    m2 = merged[[f"{label}_2" for label in LABELS]].to_numpy(dtype=bool)
    union = (m1 | m2).sum(axis=1)
    jaccard = np.where(union == 0, 1.0, (m1 & m2).sum(axis=1) / np.maximum(union, 1))
    summary = {
        "avaliacoes": len(merged),
        "conjunto_identico": float((m1 == m2).all(axis=1).mean()),
        "jaccard_medio": float(jaccard.mean()),
    }
    return pd.DataFrame(rows), summary


def kappa_meaning(kappa: float) -> str:
    """Faixas de Landis & Koch (1977), as mais citadas para interpretar o kappa."""
    if np.isnan(kappa):
        return "sem casos"
    if kappa < 0.21:
        return "fraca"
    if kappa < 0.41:
        return "razoável"
    if kappa < 0.61:
        return "moderada"
    if kappa < 0.81:
        return "substancial"
    return "quase perfeita"


def agreement_markdown(table: pd.DataFrame, summary: dict) -> str:
    lines = [
        "# Concordância de anotação",
        "",
        f"Mesma pessoa, duas rodadas, {summary['avaliacoes']} avaliações.",
        "",
        "| Categoria | Rodada 1 | Rodada 2 | Concordância | Kappa | Leitura |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in table.itertuples():
        kappa = "–" if np.isnan(row.kappa) else f"{row.kappa:.2f}".replace(".", ",")
        lines.append(
            f"| {row.categoria} | {row.rodada_1} | {row.rodada_2} | "
            f"{row.concordancia:.0%} | {kappa} | {kappa_meaning(row.kappa)} |"
        )
    lines += [
        "",
        f"- Conjunto de rótulos idêntico: {summary['conjunto_identico']:.0%}",
        f"- Jaccard médio: {summary['jaccard_medio']:.2f}".replace(".", ","),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def _cmd_queue(force: bool) -> None:
    if QUEUE_FILE.is_file() and not force:
        raise SystemExit(
            f"{QUEUE_FILE.name} já existe. Recriar muda a ordem e a amostra; "
            "só faça isso antes de começar a anotar (use --force)."
        )
    if not PROCESSED_REVIEWS_FILE.is_file():
        raise SystemExit("Base processada não encontrada. Rode `python -m reviews.prepare`.")
    reviews = pd.read_parquet(PROCESSED_REVIEWS_FILE, columns=["review_id", "sentimento", "split"])
    queue = build_queue(reviews)
    save_queue(queue)
    print(f"Fila criada em {QUEUE_FILE} com {len(queue)} avaliações:")
    print(queue["etapa"].value_counts(sort=False).to_string())


def _cmd_status() -> None:
    queue = load_queue()
    print(progress(queue, load_labels()).to_string())
    if AGREEMENT_QUEUE_FILE.is_file():
        done = len(load_labels(AGREEMENT_LABELS_FILE))
        print(f"\nConcordância (2ª rodada): {done}/{len(load_queue(AGREEMENT_QUEUE_FILE))}")


def _cmd_agreement() -> None:
    first, second = load_labels(LABELS_FILE), load_labels(AGREEMENT_LABELS_FILE)
    if second.empty:
        raise SystemExit("Ainda não há rótulos da 2ª rodada. Rode o app com --rodada 2.")
    table, summary = agreement_report(first, second)
    text = agreement_markdown(table, summary)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "concordancia_anotacao.md"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"Relatório salvo em {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ferramentas de anotação.")
    sub = parser.add_subparsers(dest="comando", required=True)
    fila = sub.add_parser("fila", help="cria a fila de anotação")
    fila.add_argument("--force", action="store_true", help="recria mesmo se já existir")
    sub.add_parser("status", help="mostra o progresso")
    sub.add_parser("concordancia", help="calcula o kappa entre as rodadas")
    args = parser.parse_args(argv)

    if args.comando == "fila":
        _cmd_queue(args.force)
    elif args.comando == "status":
        _cmd_status()
    else:
        _cmd_agreement()


if __name__ == "__main__":
    main()
