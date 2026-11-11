"""Fine-tuning do BERTimbau nas duas tarefas do projeto.

Feito para rodar no Google Colab com GPU (veja ``notebooks/03_bertimbau_colab.ipynb``),
mas funciona em qualquer máquina com PyTorch e Transformers instalados.

Regras de honestidade da avaliação (as mesmas do baseline):
- o modelo só treina com o conjunto de **treino**;
- a **validação** escolhe a melhor época (parada antecipada);
- o **teste** é medido uma vez, ao final, com ``--teste``.

Uso:
    python -m reviews.bert sentimento --teste
    python -m reviews.bert problemas --teste
    python -m reviews.bert publicar --usuario SEU_USUARIO_HF

PyTorch e Transformers são importados só dentro das funções que treinam ou
predizem. Assim o restante do projeto (e seus testes) roda sem eles.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reviews.annotation import CATEGORIES
from reviews.annotation import LABELS as PROBLEM_LABELS
from reviews.baseline import (
    TASK_PROBLEMS,
    TASK_SENTIMENT,
    apply_rules,
    problem_frame,
    sentiment_frame,
    split_frames,
    write_results_markdown,
)
from reviews.evaluate import (
    evaluate_classification,
    evaluate_multilabel,
    fmt_br,
    latency_metrics,
    load_results,
    save_result,
    timed,
)
from reviews.labels import BINARY_SENTIMENTS
from reviews.paths import MODELS_DIR, REPORTS_DIR
from reviews.splits import TEST, TRAIN, VAL

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
MODEL_KEY = "bertimbau"
MODEL_NAME = "BERTimbau ajustado"
SEED = 42
MAX_LENGTH = 128  # EDA: percentil 99 = 38 palavras, máximo 48 (≈ 70 tokens)
CPU_LATENCY_SAMPLE = 200
DATA_LICENSE = "cc-by-nc-sa-4.0"
DATASET_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    patience: int = 2
    warmup_fraction: float = 0.1
    weight_decay: float = 0.01


# Escolhidos antes de ver o teste. A validação só decide a melhor época.
CONFIGS: dict[str, TrainConfig] = {
    TASK_SENTIMENT: TrainConfig(epochs=3, learning_rate=2e-5, batch_size=32),
    TASK_PROBLEMS: TrainConfig(epochs=12, learning_rate=3e-5, batch_size=16, patience=3),
}


# ---------------------------------------------------------------------------
# Rótulos
# ---------------------------------------------------------------------------


def task_labels(task: str) -> list[str]:
    if task == TASK_SENTIMENT:
        return list(BINARY_SENTIMENTS)
    if task == TASK_PROBLEMS:
        return list(PROBLEM_LABELS)
    raise ValueError(f"Tarefa desconhecida: {task}")


def encode_targets(task: str, frame: pd.DataFrame) -> np.ndarray:
    """Alvos no formato que o modelo espera: índice da classe ou vetor 0/1."""
    if task == TASK_SENTIMENT:
        index = {label: i for i, label in enumerate(task_labels(task))}
        unknown = set(frame["sentimento"]) - set(index)
        if unknown:
            raise ValueError(f"Sentimentos fora da tarefa binária: {sorted(unknown)}")
        return frame["sentimento"].map(index).to_numpy(dtype=np.int64)
    return frame[list(PROBLEM_LABELS)].to_numpy(dtype=np.float32)


def logits_to_predictions(task: str, logits: np.ndarray):
    """Converte a saída do modelo em previsões no formato do ``evaluate.py``."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = task_labels(task)
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError(f"Esperava logits n×{len(labels)}; recebi {logits.shape}.")
    if task == TASK_SENTIMENT:
        return np.asarray(labels, dtype=object)[logits.argmax(axis=1)]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return apply_rules(probabilities)


def score(task: str, truth, predictions) -> dict:
    labels = task_labels(task)
    if task == TASK_SENTIMENT:
        return evaluate_classification(truth, predictions, labels)
    return evaluate_multilabel(truth, predictions, labels)


def truth_of(task: str, frame: pd.DataFrame):
    if task == TASK_SENTIMENT:
        return frame["sentimento"].to_numpy(dtype=object)
    return frame[list(PROBLEM_LABELS)].to_numpy(dtype=int)


def warmup_steps(n_train: int, batch_size: int, epochs: int, fraction: float) -> int:
    """Passos de aquecimento como inteiro (funciona em todas as versões do Transformers)."""
    total = math.ceil(n_train / batch_size) * epochs
    return int(total * fraction)


# ---------------------------------------------------------------------------
# Treino e predição
# ---------------------------------------------------------------------------


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _ReviewDataset:
    """Dataset do PyTorch com textos já tokenizados (padding feito por lote)."""

    def __init__(self, encodings: dict, targets: np.ndarray):
        self.encodings = encodings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        import torch

        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets[idx])
        return item


def _make_collator(tokenizer):
    """Padding por lote nos textos; os alvos são empilhados à parte.

    O ``DataCollatorWithPadding`` sozinho não aceita vetores de alvos (multirrótulo).
    """
    import torch
    from transformers import DataCollatorWithPadding

    pad = DataCollatorWithPadding(tokenizer)

    def collate(features: list[dict]) -> dict:
        targets = torch.stack([feature["labels"] for feature in features])
        batch = pad([{k: v for k, v in feature.items() if k != "labels"} for feature in features])
        batch["labels"] = targets
        return batch

    return collate


def _tokenize(tokenizer, texts: Sequence[str]) -> dict:
    return tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)


def load_model(task: str, source: str):
    """Carrega tokenizador e modelo de uma pasta local ou de um repositório do Hub."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = task_labels(task)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        problem_type=(
            "multi_label_classification" if task == TASK_PROBLEMS else "single_label_classification"
        ),
    )
    return tokenizer, model


def predict_logits(model, tokenizer, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Logits para uma lista de textos, em lotes, sem calcular gradientes."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    outputs = []
    texts = list(texts)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, model.config.num_labels))


def train(
    task: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: Path,
    base_model: str = BASE_MODEL,
    config: TrainConfig | None = None,
):
    """Treina com parada antecipada na validação e devolve (tokenizer, modelo, histórico)."""
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    config = config or CONFIGS[task]
    set_seed(SEED)
    tokenizer, model = load_model(task, base_model)

    train_set = _ReviewDataset(
        _tokenize(tokenizer, train_frame["texto"]), encode_targets(task, train_frame)
    )
    val_set = _ReviewDataset(
        _tokenize(tokenizer, validation_frame["texto"]), encode_targets(task, validation_frame)
    )
    truth_val = truth_of(task, validation_frame)

    def compute_metrics(eval_pred) -> dict:
        logits = eval_pred.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        return {"f1_macro": score(task, truth_val, logits_to_predictions(task, logits))["f1_macro"]}

    import torch

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=warmup_steps(
            len(train_set), config.batch_size, config.epochs, config.warmup_fraction
        ),
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=_make_collator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )
    trainer.train()
    history = pd.DataFrame(
        [
            {"epoca": round(row["epoch"]), "f1_macro_validacao": row["eval_f1_macro"]}
            for row in trainer.state.log_history
            if "eval_f1_macro" in row
        ]
    )
    return tokenizer, trainer.model, history


# ---------------------------------------------------------------------------
# Execução das tarefas
# ---------------------------------------------------------------------------


def model_dir(task: str) -> Path:
    return MODELS_DIR / f"bertimbau_{task}"


def run_task(
    task: str,
    include_test: bool,
    base_model: str = BASE_MODEL,
    config: TrainConfig | None = None,
    frame: pd.DataFrame | None = None,
) -> dict:
    """Treina, salva o modelo e registra as métricas no ``resultados.json``."""
    config = config or CONFIGS[task]
    if frame is None:
        frame = sentiment_frame() if task == TASK_SENTIMENT else problem_frame()
    parts = split_frames(frame)  # aborta se algum conjunto vazar no outro
    train_frame, validation_frame, test_frame = parts[TRAIN], parts[VAL], parts[TEST]
    print(
        f"Tarefa: {task} · treino {len(train_frame)} · validação {len(validation_frame)}"
        f" · teste {len(test_frame)} · dispositivo: {_device()}"
    )

    out = model_dir(task)
    tokenizer, model, history = train(task, train_frame, validation_frame, out, base_model, config)
    _save_local(out, tokenizer, model)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # o melhor já foi salvo
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(
        REPORTS_DIR / f"bertimbau_{task}_historico.csv", index=False, lineterminator="\n"
    )
    best_epoch = int(history.loc[history["f1_macro_validacao"].idxmax(), "epoca"])
    print(f"Melhor época na validação: {best_epoch}")

    evaluated = {}
    for split_name, part in [(VAL, validation_frame)] + (
        [(TEST, test_frame)] if include_test else []
    ):
        logits, seconds = timed(lambda part=part: predict_logits(model, tokenizer, part["texto"]))
        metrics = score(task, truth_of(task, part), logits_to_predictions(task, logits))
        gpu = latency_metrics(seconds, len(part))
        metrics.update(
            modelo=MODEL_NAME,
            custo_usd_por_1000=0.0,
            latencia_gpu_ms_por_avaliacao=gpu["latencia_ms_por_avaliacao"]
            if _device() == "cuda"
            else None,
            hiperparametros={
                "modelo_base": base_model,
                "max_length": MAX_LENGTH,
                "epocas_max": config.epochs,
                "melhor_epoca": best_epoch,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "limiar": 0.5 if task == TASK_PROBLEMS else None,
            },
        )
        evaluated[split_name] = metrics

    # Latência em CPU, comparável à do baseline e à da demo no Hugging Face Spaces.
    cpu = measure_cpu_latency(model, tokenizer, validation_frame["texto"])
    for split_name, metrics in evaluated.items():
        metrics.update(cpu)
        save_result(task, split_name, MODEL_KEY, metrics)
        _print_metrics(task, split_name, metrics)

    write_results_markdown()
    return evaluated


def measure_cpu_latency(model, tokenizer, texts: pd.Series, n: int = CPU_LATENCY_SAMPLE) -> dict:
    """Mede a latência em CPU numa amostra fixa (o modelo termina na CPU)."""
    sample = texts.sample(min(n, len(texts)), random_state=SEED).tolist()
    model.to("cpu")
    _, seconds = timed(lambda: predict_logits(model, tokenizer, sample, batch_size=32))
    metrics = latency_metrics(seconds, len(sample))
    return {
        **metrics,
        "latencia_medida_em": f"CPU, {len(sample)} avaliações em lotes de 32",
    }


def _save_local(out: Path, tokenizer, model) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Modelo salvo em {out}")


def _print_metrics(task: str, split_name: str, metrics: dict) -> None:
    print(f"\n== {task} · {split_name} ==")
    line = f"F1 macro: {fmt_br(metrics['f1_macro'])}"
    if metrics.get("latencia_gpu_ms_por_avaliacao") is not None:
        line += f" · GPU {fmt_br(metrics['latencia_gpu_ms_por_avaliacao'], 2)} ms/aval."
    line += f" · CPU {fmt_br(metrics['latencia_ms_por_avaliacao'], 2)} ms/aval."
    print(line)
    print(pd.DataFrame(metrics["por_classe"]).T.round(3).to_string())


# ---------------------------------------------------------------------------
# Publicação no Hugging Face Hub
# ---------------------------------------------------------------------------


def repo_id(user: str, task: str) -> str:
    if not user or "/" in user or user.strip() != user:
        raise ValueError("Informe só o nome de usuário do Hugging Face, sem barras nem espaços.")
    return f"{user}/bertimbau-olist-{task}"


def model_card(task: str, results: dict, repo: str) -> str:
    """Model card (README.md do repositório no Hub) a partir dos resultados registrados."""
    by_split = results.get(task, {})
    rows = []
    for split_name in (VAL, TEST):
        entry = by_split.get(split_name, {})
        for key, name in ((MODEL_KEY, MODEL_NAME), ("baseline_tfidf", "Baseline TF-IDF")):
            if key in entry:
                f1, n = fmt_br(entry[key]["f1_macro"]), entry[key]["n"]
                rows.append(f"| {name} | {split_name} | {f1} | {n} |")
    metrics_table = (
        "\n".join(["| Modelo | Conjunto | F1 macro | n |", "|---|---|---:|---:|", *rows])
        if rows
        else "_Sem resultados registrados._"
    )

    test_entry = by_split.get(TEST, {}).get(MODEL_KEY) or by_split.get(VAL, {}).get(MODEL_KEY, {})
    per_class = test_entry.get("por_classe", {})
    class_rows = [
        f"| {CATEGORIES.get(label, label)} | {fmt_br(v['f1'])} | {v['suporte']} |"
        for label, v in per_class.items()
    ]
    class_table = "\n".join(["| Classe | F1 | Suporte |", "|---|---:|---:|", *class_rows])

    if task == TASK_SENTIMENT:
        description = (
            "Classifica avaliações de e-commerce em **negativa** ou **positiva**. "
            "O rótulo vem da nota do cliente (1–2 negativa, 4–5 positiva; a nota 3 fica fora)."
        )
        usage = (
            'pred = classifier("Chegou antes do prazo, recomendo!")\n'
            "# [{'label': 'positiva', 'score': ...}]"
        )
        pipeline_kwargs = ""
        caveat = ""
    else:
        description = (
            "Identifica, numa avaliação **negativa**, os tipos de problema relatados "
            "(multirrótulo, 8 categorias): "
            + ", ".join(CATEGORIES.values())
            + ". Os rótulos foram anotados à mão seguindo um guia publicado "
            "no repositório do projeto."
        )
        usage = (
            'pred = classifier("Atrasou e ainda veio quebrado")\n'
            "# considere as categorias com score ≥ 0,5"
        )
        pipeline_kwargs = ', top_k=None, function_to_apply="sigmoid"'
        caveat = (
            "- Treinado com 750 avaliações anotadas por **uma única pessoa** "
            "(concordância intra-anotador: kappa ≥ 0,87 em todas as categorias).\n"
            '- "Outro problema" e "Sem reclamação" têm poucos exemplos '
            "e são as classes mais fracas.\n"
        )

    return f"""---
language: pt
license: {DATA_LICENSE}
base_model: {BASE_MODEL}
pipeline_tag: text-classification
tags:
- bertimbau
- portuguese
- e-commerce
- reviews
- olist
---

# BERTimbau · avaliações Olist · {task}

{description}

Modelo ajustado a partir de [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) com a
[base pública da Olist]({DATASET_URL}) (avaliações reais de clientes, 2016–2018).

## Resultados

Mesma divisão treino/validação/teste do projeto (agrupada por texto, para que avaliações
idênticas não caiam em conjuntos diferentes). O teste foi medido uma única vez.

{metrics_table}

Por classe (teste):

{class_table}

## Como usar

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{repo}"{pipeline_kwargs})
{usage}
```

## Treino

- Texto: título + comentário, até {MAX_LENGTH} tokens.
- Parada antecipada pela F1 macro na validação; hiperparâmetros fixados antes de ver o teste.

## Limitações

- Domínio: marketplace brasileiro, 2016–2018. Gírias e produtos novos podem degradar o desempenho.
{caveat}- Os nomes de loja da base são pseudônimos criados pela Olist.

## Licença

Os dados da Olist são distribuídos sob **CC BY-NC-SA 4.0**; por ser derivado deles,
este modelo segue a mesma licença: uso **não comercial**, com atribuição e compartilhamento
pela mesma licença.
"""


def publish(
    user: str, tasks: Sequence[str] = (TASK_SENTIMENT, TASK_PROBLEMS), private: bool = False
) -> list[str]:
    """Envia os modelos treinados ao Hub. O token vem da variável HF_TOKEN e nunca é impresso."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "Defina o token de escrita do Hugging Face na variável HF_TOKEN "
            "(no Colab: Secrets → HF_TOKEN)."
        )
    api = HfApi(token=token)
    results = load_results()
    urls = []
    for task in tasks:
        folder = model_dir(task)
        if not (folder / "config.json").is_file():
            raise SystemExit(
                f"Modelo de {task} não encontrado em {folder}. Treine antes de publicar."
            )
        repo = repo_id(user, task)
        (folder / "README.md").write_text(model_card(task, results, repo), encoding="utf-8")
        api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            folder_path=str(folder),
            commit_message="Modelo ajustado e model card",
            ignore_patterns=["checkpoints/*", "*.tmp"],
        )
        url = f"https://huggingface.co/{repo}"
        urls.append(url)
        print(f"Publicado: {url}")
    return urls


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning do BERTimbau.")
    sub = parser.add_subparsers(dest="comando", required=True)
    for task in (TASK_SENTIMENT, TASK_PROBLEMS):
        cmd = sub.add_parser(task, help=f"treina e avalia a tarefa de {task}")
        cmd.add_argument("--teste", action="store_true", help="mede também no teste (uma vez)")
    pub = sub.add_parser("publicar", help="envia os modelos ao Hugging Face Hub")
    pub.add_argument("--usuario", required=True, help="seu nome de usuário no Hugging Face")
    pub.add_argument("--privado", action="store_true", help="cria os repositórios como privados")
    args = parser.parse_args(argv)

    if args.comando == "publicar":
        publish(args.usuario, private=args.privado)
    else:
        run_task(args.comando, include_test=args.teste)


if __name__ == "__main__":
    main()
