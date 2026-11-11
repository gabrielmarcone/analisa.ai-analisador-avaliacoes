"""LLM zero-shot (Claude Haiku 5.5) nas duas tarefas do projeto.

O modelo não é treinado: recebe a definição da tarefa (e, nos tipos de problema,
as regras do guia de anotação) e classifica as avaliações em lotes, com saída
estruturada em JSON. Mede-se F1, custo real (tokens cobrados) e latência.

Regras de honestidade (as mesmas dos outros modelos):
- o esforço de raciocínio e o prompt são escolhidos olhando só a **validação**;
- o **teste** é medido uma vez, com ``--teste``. Depois disso, rodar o teste com
  outra configuração é bloqueado (a não ser com ``--forcar``, que fica registrado).

Toda resposta da API vai para um cache local (``data/llm_cache/``, fora do Git).
Se a execução cair no meio, rodar de novo continua de onde parou, sem pagar de novo.

Uso:
    python -m reviews.llm verificar                  # testa a chave com 2 frases inventadas
    python -m reviews.llm problemas --simular        # mostra o plano, sem chamar a API
    python -m reviews.llm problemas --esforco low    # validação
    python -m reviews.llm problemas --esforco low --teste

A chave vem de ``ANTHROPIC_API_KEY`` (arquivo ``.env`` na raiz, que não vai para o Git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from reviews.annotation import LABELS as PROBLEM_LABELS
from reviews.annotation import NO_COMPLAINT
from reviews.baseline import (
    TASK_PROBLEMS,
    TASK_SENTIMENT,
    problem_frame,
    sentiment_frame,
    split_frames,
    write_results_markdown,
)
from reviews.bert import score, task_labels, truth_of
from reviews.evaluate import fmt_br, latency_metrics, load_results, save_result
from reviews.llm_prompts import PROBLEMS_SYSTEM, PROMPT_VERSION, SENTIMENT_SYSTEM, user_message
from reviews.paths import DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from reviews.splits import TEST, VAL

MODEL = "claude-haiku-5-5"
MODEL_KEY = "llm_zero_shot"
MODEL_NAME = "Claude Haiku 5.5 (zero-shot)"
SEED = 42
BATCH_SIZE = 25
MAX_TOKENS = 16_000  # teto de saída por chamada (raciocínio + JSON); só o usado é cobrado
EFFORTS = ("low", "medium", "high")
DEFAULT_EFFORT = "low"
MAX_ATTEMPTS = 2  # tentativas por lote antes de dividir em chamadas individuais
DEFAULT_PARALLEL = 4
DEFAULT_LIMIT_USD = 2.0  # trava de gasto por execução (chamadas novas)
SDK_MAX_RETRIES = 6  # o SDK repete sozinho erros 429/5xx, respeitando retry-after

# US$ por milhão de tokens, Claude Haiku 5.5, prompts de até 100 mil tokens
# (https://platform.claude.com/docs/en/about-claude/pricing, conferido em 11/11/2026).
PRICES_USD_PER_MTOK = {
    "input_tokens": 0.10,
    "cache_creation_input_tokens": 0.125,
    "cache_read_input_tokens": 0.01,
    "output_tokens": 0.50,
}
PROMPT_TOKEN_LIMIT = 100_000  # acima disso o preço muda; nossos prompts ficam muito abaixo

CACHE_DIR = DATA_DIR / "llm_cache"
RUNS_LOG = REPORTS_DIR / "llm_execucoes.csv"
ENV_FILE = PROJECT_ROOT / ".env"

FALLBACK = {TASK_SENTIMENT: "positiva", TASK_PROBLEMS: ["outro"]}


class InvalidResponse(ValueError):
    """Resposta fora do formato combinado (ids faltando, rótulo inválido, JSON quebrado)."""


class SpendLimitReached(RuntimeError):
    """A execução atingiu a trava de gasto definida em ``--limite-usd``."""


# ---------------------------------------------------------------------------
# Configuração e formato da resposta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    task: str
    effort: str = DEFAULT_EFFORT
    model: str = MODEL
    batch_size: int = BATCH_SIZE
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        task_labels(self.task)  # valida a tarefa
        if self.effort not in EFFORTS:
            raise ValueError(f"Esforço deve ser um de {EFFORTS}; recebi {self.effort!r}.")
        if self.batch_size < 1:
            raise ValueError("O lote precisa ter ao menos 1 avaliação.")

    def key(self) -> str:
        """Identificador curto e estável da configuração (entra no nome do cache).

        Inclui o texto do prompt e o esquema: mesmo que alguém edite o prompt sem subir
        a versão, respostas antigas não são reaproveitadas por engano.
        """
        payload = {
            **asdict(self),
            "prompt": system_prompt(self.task),
            "esquema": response_schema(self.task),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]


def system_prompt(task: str) -> str:
    return SENTIMENT_SYSTEM if task == TASK_SENTIMENT else PROBLEMS_SYSTEM


def response_schema(task: str) -> dict:
    """JSON Schema da saída estruturada (a API garante respostas nesse formato)."""
    if task == TASK_SENTIMENT:
        answer = {"sentimento": {"type": "string", "enum": list(task_labels(task))}}
    else:
        answer = {
            "categorias": {
                "type": "array",
                "items": {"type": "string", "enum": list(PROBLEM_LABELS)},
                "minItems": 1,
            }
        }
    item = {
        "type": "object",
        "properties": {"id": {"type": "integer"}, **answer},
        "required": ["id", *answer],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"respostas": {"type": "array", "items": item}},
        "required": ["respostas"],
        "additionalProperties": False,
    }


def parse_response(task: str, text: str, n_items: int) -> list:
    """Valida a resposta e devolve uma previsão por avaliação, na ordem dos ids 1..n."""
    try:
        answers = json.loads(text)["respostas"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise InvalidResponse(f"JSON inválido: {error}") from error
    if not isinstance(answers, list):
        raise InvalidResponse("'respostas' não é uma lista.")

    field_name = "sentimento" if task == TASK_SENTIMENT else "categorias"
    valid = set(task_labels(task))
    by_id: dict[int, object] = {}
    for answer in answers:
        if not isinstance(answer, dict) or not isinstance(answer.get("id"), int):
            raise InvalidResponse(f"Item sem id inteiro: {answer!r}")
        item_id = answer["id"]
        if item_id in by_id:
            raise InvalidResponse(f"id {item_id} repetido.")
        value = answer.get(field_name)
        if task == TASK_SENTIMENT:
            if value not in valid:
                raise InvalidResponse(f"Sentimento inválido no id {item_id}: {value!r}")
        else:
            if not isinstance(value, list) or not value or not set(value) <= valid:
                raise InvalidResponse(f"Categorias inválidas no id {item_id}: {value!r}")
        by_id[item_id] = value

    expected = set(range(1, n_items + 1))
    if set(by_id) != expected:
        missing, extra = sorted(expected - set(by_id)), sorted(set(by_id) - expected)
        raise InvalidResponse(f"ids faltando {missing} ou sobrando {extra}.")
    return [by_id[i] for i in range(1, n_items + 1)]


def problems_to_matrix(predictions: Sequence[Sequence[str]]) -> np.ndarray:
    """Listas de categorias → matriz 0/1, aplicando a regra do guia para "sem reclamação"."""
    index = {label: i for i, label in enumerate(PROBLEM_LABELS)}
    no_complaint = index[NO_COMPLAINT]
    matrix = np.zeros((len(predictions), len(PROBLEM_LABELS)), dtype=int)
    for row, labels in enumerate(predictions):
        for label in labels:
            matrix[row, index[label]] = 1
        if matrix[row].sum() > 1:  # "sem reclamação" nunca acompanha outro rótulo
            matrix[row, no_complaint] = 0
    return matrix


# ---------------------------------------------------------------------------
# Custo
# ---------------------------------------------------------------------------


def usage_to_dict(usage) -> dict:
    """Extrai do objeto ``usage`` da API só os contadores que entram no custo."""
    data = {key: int(getattr(usage, key, 0) or 0) for key in PRICES_USD_PER_MTOK}
    details = getattr(usage, "output_tokens_details", None)
    data["thinking_tokens"] = int(getattr(details, "thinking_tokens", 0) or 0)
    return data


def cost_usd(usage: dict) -> float:
    return sum(usage.get(key, 0) * price for key, price in PRICES_USD_PER_MTOK.items()) / 1e6


def _sum_usage(records: Sequence[dict]) -> dict:
    keys = [*PRICES_USD_PER_MTOK, "thinking_tokens"]
    return {key: sum(record["uso"].get(key, 0) for record in records) for key in keys}


# ---------------------------------------------------------------------------
# Chamadas à API
# ---------------------------------------------------------------------------


def make_client():
    """Cliente da Anthropic com a chave do ``.env``. A chave nunca é impressa."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit(
            "Chave não encontrada. Crie o arquivo .env na raiz do projeto com a linha\n"
            "ANTHROPIC_API_KEY=sua_chave   (veja .env.example; o .env não vai para o Git)."
        )
    import anthropic

    return anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)


@dataclass
class BatchResult:
    review_ids: list[str]
    predictions: list
    seconds: float
    usage: dict
    calls: int
    failed_ids: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "review_ids": self.review_ids,
            "predicoes": self.predictions,
            "segundos": self.seconds,
            "uso": self.usage,
            "chamadas": self.calls,
            "falhas": self.failed_ids,
        }


class Classifier:
    """Classifica lotes de textos com o LLM, validando e repetindo quando preciso."""

    def __init__(self, client, config: RunConfig):
        self.client = client
        self.config = config
        self.schema = response_schema(config.task)
        self.system = system_prompt(config.task)

    def _call(self, texts: Sequence[str]) -> tuple[list | None, float, dict]:
        """Uma chamada. Devolve (previsões ou None se inválidas, segundos, uso)."""
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_message(texts)}],
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": self.schema},
            },
        )
        seconds = time.perf_counter() - start
        usage = usage_to_dict(response.usage)
        prompt_tokens = sum(
            usage[k]
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        if prompt_tokens > PROMPT_TOKEN_LIMIT:
            raise RuntimeError("Prompt acima de 100 mil tokens: o preço mudaria. Reduza o lote.")
        if response.stop_reason not in ("end_turn", "stop_sequence"):
            return None, seconds, usage  # recusa ou corte por max_tokens: resposta incompleta
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
        try:
            return parse_response(self.config.task, text, len(texts)), seconds, usage
        except InvalidResponse:
            return None, seconds, usage

    def classify(self, review_ids: Sequence[str], texts: Sequence[str]) -> BatchResult:
        seconds, usages, calls = 0.0, [], 0
        for _ in range(MAX_ATTEMPTS):
            predictions, spent, usage = self._call(texts)
            seconds, calls = seconds + spent, calls + 1
            usages.append({"uso": usage})
            if predictions is not None:
                return BatchResult(
                    list(review_ids), predictions, seconds, _sum_usage(usages), calls
                )

        # O lote falhou duas vezes: classifica uma avaliação por vez.
        predictions, failed = [], []
        for review_id, text in zip(review_ids, texts, strict=True):
            single, spent, usage = self._call([text])
            seconds, calls = seconds + spent, calls + 1
            usages.append({"uso": usage})
            if single is None:
                failed.append(review_id)
                predictions.append(FALLBACK[self.config.task])
            else:
                predictions.append(single[0])
        return BatchResult(
            list(review_ids), predictions, seconds, _sum_usage(usages), calls, failed
        )


# ---------------------------------------------------------------------------
# Lotes, cache e execução de um conjunto
# ---------------------------------------------------------------------------


def make_batches(frame: pd.DataFrame, batch_size: int, seed: int = SEED) -> list[pd.DataFrame]:
    """Embaralha uma vez (semente fixa) e corta em lotes: mesma entrada, mesmos lotes."""
    order = np.random.default_rng(seed).permutation(len(frame))
    shuffled = frame.iloc[order].reset_index(drop=True)
    return [shuffled.iloc[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def cache_path(config: RunConfig, split: str, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{config.task}_{split}_{config.key()}.jsonl"


def load_cache(path: Path) -> dict[tuple[str, ...], dict]:
    """Respostas já pagas, indexadas pelos review_ids do lote. Linha quebrada é ignorada."""
    cached: dict[tuple[str, ...], dict] = {}
    if not path.is_file():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            cached[tuple(record["review_ids"])] = record
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # p. ex. a última linha, se o programa caiu no meio da escrita
    return cached


@dataclass
class SplitRun:
    predictions: list  # na ordem do frame recebido
    records: list[dict]
    new_cost_usd: float

    @property
    def failed_ids(self) -> list[str]:
        return [rid for record in self.records for rid in record["falhas"]]


def _call_pending(
    pending: list[pd.DataFrame],
    classifier: Classifier,
    path: Path,
    cached: dict,
    parallel: int,
    limit_usd: float,
    progress: Callable[[str], None],
) -> float:
    """Chama a API para os lotes pendentes, gravando cada resposta no cache assim que chega.

    Se uma chamada falhar ou a trava de gasto for atingida, para de enviar lotes novos,
    espera os que já estão em andamento (e grava as respostas, que já foram pagas)
    e só então interrompe.
    """
    queue = iter(pending)
    running: dict = {}
    new_cost, done, error = 0.0, 0, None
    if path.is_file() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a", encoding="utf-8", newline="\n") as out:
            out.write("\n")  # isola uma linha cortada por uma queda anterior
    with (
        ThreadPoolExecutor(max_workers=max(1, parallel)) as pool,
        path.open("a", encoding="utf-8", newline="\n") as out,
    ):

        def submit_next() -> None:
            batch = next(queue, None)
            if batch is not None:
                ids, texts = list(batch["review_id"]), list(batch["texto"])
                running[pool.submit(classifier.classify, ids, texts)] = batch

        for _ in range(max(1, parallel)):
            submit_next()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                running.pop(future)
                try:
                    record = future.result().to_record()
                except Exception as exc:  # erro da API: guarda e para de enviar
                    error = error or exc
                    continue
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                cached[tuple(record["review_ids"])] = record
                new_cost += cost_usd(record["uso"])
                done += 1
                if done % 10 == 0 or done == len(pending):
                    progress(
                        f"  {done}/{len(pending)} lotes · gasto nesta execução: US$ {new_cost:.4f}"
                    )
                if new_cost > limit_usd and error is None:
                    error = SpendLimitReached(
                        f"Gasto desta execução passou de US$ {limit_usd:.2f}. O que já foi feito "
                        "está no cache; aumente --limite-usd se quiser continuar."
                    )
                if error is None:
                    submit_next()
    if error is not None:
        raise error
    return new_cost


def run_split(
    frame: pd.DataFrame,
    config: RunConfig,
    split: str,
    classifier: Classifier | None,
    parallel: int = DEFAULT_PARALLEL,
    limit_usd: float = DEFAULT_LIMIT_USD,
    cache_dir: Path = CACHE_DIR,
    progress: Callable[[str], None] = print,
) -> SplitRun:
    """Classifica um conjunto inteiro, reaproveitando o cache e gravando cada lote novo."""
    batches = make_batches(frame, config.batch_size)
    path = cache_path(config, split, cache_dir)
    cached = load_cache(path)
    pending = [b for b in batches if tuple(b["review_id"]) not in cached]
    progress(
        f"{config.task} · {split}: {len(batches)} lotes, {len(batches) - len(pending)} no cache,"
        f" {len(pending)} a chamar (esforço {config.effort})"
    )
    if pending and classifier is None:
        raise RuntimeError("Há lotes fora do cache e nenhum cliente para chamar a API.")

    new_cost = 0.0
    if pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_cost = _call_pending(pending, classifier, path, cached, parallel, limit_usd, progress)

    records = [cached[tuple(b["review_id"])] for b in batches]
    by_id = {
        rid: pred
        for record in records
        for rid, pred in zip(record["review_ids"], record["predicoes"], strict=True)
    }
    return SplitRun([by_id[rid] for rid in frame["review_id"]], records, new_cost)


def split_metrics(task: str, frame: pd.DataFrame, run: SplitRun, config: RunConfig) -> dict:
    predictions = run.predictions
    if task == TASK_SENTIMENT:
        predictions = np.asarray(predictions, dtype=object)
    else:
        predictions = problems_to_matrix(predictions)
    metrics = score(task, truth_of(task, frame), predictions)

    usage = _sum_usage(run.records)
    total_cost = cost_usd(usage)
    seconds = sum(record["segundos"] for record in run.records)
    calls = sum(record["chamadas"] for record in run.records)
    metrics.update(latency_metrics(seconds, len(frame)))
    metrics.update(
        modelo=MODEL_NAME,
        custo_usd_por_1000=total_cost / len(frame) * 1000,
        custo_total_usd=total_cost,
        latencia_ms_por_chamada=seconds / calls * 1000,
        latencia_medida_em=(
            f"API da Anthropic, soma do tempo das chamadas ÷ avaliações (lotes de "
            f"{config.batch_size})"
        ),
        chamadas=calls,
        falhas=len(run.failed_ids),
        tokens=usage,
        hiperparametros={
            "modelo_api": config.model,
            "esforco": config.effort,
            "lote": config.batch_size,
            "versao_prompt": config.prompt_version,
            "max_tokens": MAX_TOKENS,
            "configuracao": config.key(),
        },
    )
    return metrics


# ---------------------------------------------------------------------------
# Tarefa completa
# ---------------------------------------------------------------------------


def check_test_once(task: str, config: RunConfig, force: bool, results: dict) -> None:
    """Bloqueia medir o teste de novo com outra configuração (seria escolher pelo teste)."""
    previous = results.get(task, {}).get(TEST, {}).get(MODEL_KEY)
    if not previous:
        return
    old_key = previous.get("hiperparametros", {}).get("configuracao")
    if old_key != config.key() and not force:
        raise SystemExit(
            f"O teste de {task} já foi medido com outra configuração ({old_key}). "
            "Medir de novo seria ajustar pelo teste. Se for mesmo necessário, use --forcar "
            "e registre o motivo no README."
        )


def append_run_log(rows: list[dict], path: Path = RUNS_LOG) -> None:
    """Histórico de execuções (inclui as de validação descartadas na escolha do esforço)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.is_file():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _pending_batches(frame: pd.DataFrame, config: RunConfig, split: str, cache_dir: Path) -> int:
    cached = load_cache(cache_path(config, split, cache_dir))
    return sum(tuple(b["review_id"]) not in cached for b in make_batches(frame, config.batch_size))


def run_task(
    task: str,
    include_test: bool,
    effort: str = DEFAULT_EFFORT,
    parallel: int = DEFAULT_PARALLEL,
    limit_usd: float = DEFAULT_LIMIT_USD,
    force: bool = False,
    client=None,
    frame: pd.DataFrame | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    config = RunConfig(task=task, effort=effort)
    if include_test:
        check_test_once(task, config, force, load_results())
    if frame is None:
        frame = sentiment_frame() if task == TASK_SENTIMENT else problem_frame()
    parts = split_frames(frame)  # aborta se algum conjunto vazar no outro
    splits = [VAL, TEST] if include_test else [VAL]

    pending = sum(_pending_batches(parts[s], config, s, cache_dir) for s in splits)
    if client is None and pending:
        client = make_client()  # só exige a chave se houver algo a chamar
    classifier = Classifier(client, config) if client is not None else None

    evaluated, log_rows = {}, []
    for split_name in splits:
        run = run_split(
            parts[split_name], config, split_name, classifier, parallel, limit_usd, cache_dir
        )
        metrics = split_metrics(task, parts[split_name], run, config)
        save_result(task, split_name, MODEL_KEY, metrics)
        _print_metrics(task, split_name, metrics)
        evaluated[split_name] = metrics
        log_rows.append(
            {
                "registrado_em": datetime.now(UTC).isoformat(timespec="seconds"),
                "tarefa": task,
                "conjunto": split_name,
                "esforco": effort,
                "versao_prompt": config.prompt_version,
                "configuracao": config.key(),
                "f1_macro": round(metrics["f1_macro"], 4),
                "custo_usd_por_1000": round(metrics["custo_usd_por_1000"], 4),
                "latencia_ms_por_avaliacao": round(metrics["latencia_ms_por_avaliacao"], 1),
                "falhas": metrics["falhas"],
                "gasto_novo_usd": round(run.new_cost_usd, 4),
                "teste_forcado": bool(force and split_name == TEST),
            }
        )
    append_run_log(log_rows, RUNS_LOG)
    write_results_markdown()
    return evaluated


def _print_metrics(task: str, split_name: str, metrics: dict) -> None:
    print(f"\n== {task} · {split_name} · esforço {metrics['hiperparametros']['esforco']} ==")
    print(
        f"F1 macro: {fmt_br(metrics['f1_macro'])} · custo US$ "
        f"{fmt_br(metrics['custo_usd_por_1000'], 4)}/1.000 · "
        f"{fmt_br(metrics['latencia_ms_por_avaliacao'], 1)} ms/aval. · "
        f"{metrics['chamadas']} chamadas · {metrics['falhas']} falhas"
    )
    print(pd.DataFrame(metrics["por_classe"]).T.round(3).to_string())


def simulate(task: str, effort: str) -> None:
    """Mostra o que seria enviado, sem chamar a API nem precisar de chave."""
    config = RunConfig(task=task, effort=effort)
    frame = sentiment_frame() if task == TASK_SENTIMENT else problem_frame()
    parts = split_frames(frame)
    for split_name in (VAL, TEST):
        n_batches = len(make_batches(parts[split_name], config.batch_size))
        todo = _pending_batches(parts[split_name], config, split_name, CACHE_DIR)
        print(
            f"{split_name}: {len(parts[split_name])} avaliações, {n_batches} lotes, "
            f"{todo} fora do cache"
        )
    first = make_batches(parts[VAL], config.batch_size)[0]
    print(f"\nConfiguração {config.key()}: {asdict(config)}")
    print(f"\n--- Prompt de sistema ({len(system_prompt(task))} caracteres) ---")
    print(system_prompt(task))
    print("\n--- Primeiro lote da validação (início) ---")
    print(user_message(list(first["texto"]))[:1500])


def verify(client=None) -> None:
    """Chamada mínima com 2 frases inventadas, para conferir chave, modelo e formato."""
    config = RunConfig(task=TASK_SENTIMENT, effort="low", batch_size=2)
    classifier = Classifier(client or make_client(), config)
    texts = ["Chegou antes do prazo, adorei!", "Veio quebrado e ninguém responde."]
    result = classifier.classify(["exemplo-1", "exemplo-2"], texts)
    print(f"Respostas: {result.predictions} (esperado: ['positiva', 'negativa'])")
    print(f"Tempo: {result.seconds:.1f} s · custo: US$ {cost_usd(result.usage):.6f}")
    if result.failed_ids:
        raise SystemExit("A API respondeu fora do formato. Confira a versão do pacote anthropic.")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM zero-shot (Claude Haiku 5.5).")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("verificar", help="testa a chave com 2 frases inventadas (custo ~0)")
    for task in (TASK_SENTIMENT, TASK_PROBLEMS):
        cmd = sub.add_parser(task, help=f"classifica a tarefa de {task}")
        cmd.add_argument("--esforco", choices=EFFORTS, default=DEFAULT_EFFORT)
        cmd.add_argument("--teste", action="store_true", help="mede também no teste (uma vez)")
        cmd.add_argument(
            "--simular", action="store_true", help="só mostra o plano; não chama a API"
        )
        cmd.add_argument(
            "--paralelo", type=int, default=DEFAULT_PARALLEL, help="chamadas simultâneas"
        )
        cmd.add_argument(
            "--limite-usd",
            type=float,
            default=DEFAULT_LIMIT_USD,
            help="interrompe se o gasto desta execução passar disso",
        )
        cmd.add_argument("--forcar", action="store_true", help="permite medir o teste de novo")
    args = parser.parse_args(argv)

    if args.comando == "verificar":
        verify()
    elif args.simular:
        simulate(args.comando, args.esforco)
    else:
        run_task(
            args.comando,
            include_test=args.teste,
            effort=args.esforco,
            parallel=args.paralelo,
            limit_usd=args.limite_usd,
            force=args.forcar,
        )


if __name__ == "__main__":
    main()
