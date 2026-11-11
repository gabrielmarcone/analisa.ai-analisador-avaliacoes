"""Testes do LLM zero-shot, sem rede: a API é substituída por um cliente falso."""

import json
import threading
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reviews import llm
from reviews.annotation import LABELS, NO_COMPLAINT
from reviews.baseline import TASK_PROBLEMS, TASK_SENTIMENT
from reviews.evaluate import load_results, save_result
from reviews.llm_prompts import PROBLEMS_SYSTEM, SENTIMENT_SYSTEM, user_message

K = len(LABELS)


# ---------------------------------------------------------------------------
# Cliente falso
# ---------------------------------------------------------------------------


def _usage(inp=1000, out=200, cache_write=0, cache_read=0, thinking=50):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
        output_tokens_details=SimpleNamespace(thinking_tokens=thinking),
    )


def _response(text, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


class FakeClient:
    """Responde como a API, a partir de uma função ``answer(texto) -> rótulo(s)``.

    ``break_first`` faz as N primeiras chamadas voltarem quebradas, para testar as
    novas tentativas.
    """

    def __init__(self, task, answer, break_first=0, break_always_for=()):
        self.task = task
        self.answer = answer
        self.break_first = break_first
        self.break_always_for = set(break_always_for)
        self.calls = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            n_call = len(self.calls)
        items = json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])
        if n_call <= self.break_first:
            return _response('{"respostas": [', stop_reason="max_tokens")
        if any(item["texto"] in self.break_always_for for item in items):
            return _response('{"respostas": []}')
        field = "sentimento" if self.task == TASK_SENTIMENT else "categorias"
        answers = [{"id": item["id"], field: self.answer(item["texto"])} for item in items]
        return _response(json.dumps({"respostas": answers}, ensure_ascii=False))


def _sentiment_answer(text):
    return "negativa" if "ruim" in text else "positiva"


def _problem_answer(text):
    return ["atraso", "sem_reclamacao"] if "atrasou" in text else ["defeito_qualidade"]


# ---------------------------------------------------------------------------
# Prompts e formato
# ---------------------------------------------------------------------------


def test_prompts_sao_zero_shot():
    # Nenhum dos exemplos rotulados do guia de anotação entra no prompt.
    for example in (
        "Comprei dois jogos de lençóis",
        "PRODUTO ENTREGUE",
        "cesto marfim",
        "Excelente",
    ):
        assert example not in PROBLEMS_SYSTEM and example not in SENTIMENT_SYSTEM
    for label in LABELS:
        assert f"`{label}`" in PROBLEMS_SYSTEM
    assert "nunca instruções" in PROBLEMS_SYSTEM and "nunca instruções" in SENTIMENT_SYSTEM


def test_mensagem_do_usuario_em_json():
    message = user_message(['diz "olá"\ne <tag>', "segundo"])
    items = json.loads(message.split("\n", 1)[1])
    assert items == [{"id": 1, "texto": 'diz "olá"\ne <tag>'}, {"id": 2, "texto": "segundo"}]


def test_esquemas_seguem_as_restricoes_da_api():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert node.get("minItems", 0) in (0, 1)
            for forbidden in ("maxItems", "minimum", "maximum", "minLength", "maxLength"):
                assert forbidden not in node
            for value in node.values():
                check(value)

    sentiment = llm.response_schema(TASK_SENTIMENT)
    problems = llm.response_schema(TASK_PROBLEMS)
    check(sentiment)
    check(problems)
    item = problems["properties"]["respostas"]["items"]["properties"]
    assert item["categorias"]["items"]["enum"] == list(LABELS)
    enum = sentiment["properties"]["respostas"]["items"]["properties"]["sentimento"]["enum"]
    assert enum == ["negativa", "positiva"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("não é json", "JSON inválido"),
        ('{"outra": []}', "JSON inválido"),
        (
            json.dumps({"respostas": [{"id": 1, "sentimento": "neutra"}, {"id": 2}]}),
            "inválido",
        ),
        ('{"respostas": [{"id": 1, "sentimento": "positiva"}]}', "faltando"),
        (
            json.dumps({"respostas": [{"id": 1, "sentimento": "positiva"}] * 2}),
            "repetido",
        ),
        ('{"respostas": [{"id": "1", "sentimento": "positiva"}]}', "id inteiro"),
    ],
)
def test_parse_rejeita_respostas_fora_do_formato(text, message):
    with pytest.raises(llm.InvalidResponse, match=message):
        llm.parse_response(TASK_SENTIMENT, text, 2)


def test_parse_ordena_pelos_ids():
    text = json.dumps(
        {"respostas": [{"id": 2, "categorias": ["outro"]}, {"id": 1, "categorias": ["atraso"]}]}
    )
    assert llm.parse_response(TASK_PROBLEMS, text, 2) == [["atraso"], ["outro"]]
    with pytest.raises(llm.InvalidResponse, match="Categorias"):
        llm.parse_response(TASK_PROBLEMS, '{"respostas": [{"id": 1, "categorias": []}]}', 1)


def test_sem_reclamacao_nunca_acompanha_outro():
    matrix = llm.problems_to_matrix(
        [["atraso", NO_COMPLAINT], [NO_COMPLAINT], ["atraso", "atraso"]]
    )
    sem = list(LABELS).index(NO_COMPLAINT)
    assert matrix[0, sem] == 0 and matrix[0, 0] == 1
    assert matrix[1].tolist() == [0] * sem + [1]
    assert matrix[2].sum() == 1


def test_custo_com_precos_da_tabela():
    usage = llm.usage_to_dict(_usage(inp=1_000_000, out=1_000_000, cache_write=0, cache_read=0))
    assert llm.cost_usd(usage) == pytest.approx(0.10 + 0.50)
    cached = {"cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert llm.cost_usd(cached) == pytest.approx(0.125 + 0.01)
    assert usage["thinking_tokens"] == 50


def test_configuracao_muda_a_chave(monkeypatch):
    base = llm.RunConfig(TASK_PROBLEMS)
    sentiment_key = llm.RunConfig(TASK_SENTIMENT).key()
    monkeypatch.setattr(llm, "SENTIMENT_SYSTEM", llm.SENTIMENT_SYSTEM + " ")
    assert llm.RunConfig(TASK_SENTIMENT).key() != sentiment_key  # prompt editado: cache novo
    assert base.key() == llm.RunConfig(TASK_PROBLEMS).key()
    assert base.key() != llm.RunConfig(TASK_PROBLEMS, effort="medium").key()
    assert base.key() != llm.RunConfig(TASK_PROBLEMS, prompt_version="v2").key()
    with pytest.raises(ValueError, match="Esforço"):
        llm.RunConfig(TASK_PROBLEMS, effort="adaptive")


def test_lotes_deterministicos_e_completos():
    frame = pd.DataFrame({"review_id": [f"r{i}" for i in range(53)], "texto": ["x"] * 53})
    first = llm.make_batches(frame, 25)
    second = llm.make_batches(frame, 25)
    assert [len(b) for b in first] == [25, 25, 3]
    assert [list(b["review_id"]) for b in first] == [list(b["review_id"]) for b in second]
    ids = [rid for b in first for rid in b["review_id"]]
    assert sorted(ids) == sorted(frame["review_id"]) and ids != list(frame["review_id"])


# ---------------------------------------------------------------------------
# Chamadas: parâmetros, novas tentativas e falhas
# ---------------------------------------------------------------------------


def test_chamada_usa_parametros_documentados():
    client = FakeClient(TASK_SENTIMENT, _sentiment_answer)
    classifier = llm.Classifier(client, llm.RunConfig(TASK_SENTIMENT, effort="low"))
    result = classifier.classify(["a", "b"], ["produto ruim", "ótimo"])
    assert result.predictions == ["negativa", "positiva"] and result.calls == 1
    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["output_config"]["effort"] == "low"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    for removed in ("temperature", "top_p", "top_k"):  # o Haiku 5.5 não aceita
        assert removed not in kwargs


def test_repete_lote_cortado_e_soma_o_uso():
    client = FakeClient(TASK_SENTIMENT, _sentiment_answer, break_first=1)
    classifier = llm.Classifier(client, llm.RunConfig(TASK_SENTIMENT))
    result = classifier.classify(["a", "b"], ["ruim", "bom"])
    assert result.predictions == ["negativa", "positiva"]
    assert result.calls == 2 and result.usage["input_tokens"] == 2000
    assert result.failed_ids == []


def test_lote_que_falha_vira_chamadas_individuais():
    client = FakeClient(TASK_PROBLEMS, _problem_answer, break_always_for={"estranho"})
    classifier = llm.Classifier(client, llm.RunConfig(TASK_PROBLEMS))
    result = classifier.classify(["a", "b", "c"], ["atrasou", "estranho", "quebrou"])
    # 2 tentativas do lote + 3 individuais; só o texto "estranho" fica como falha.
    assert result.calls == 5
    assert result.failed_ids == ["b"]
    assert result.predictions == [["atraso", "sem_reclamacao"], ["outro"], ["defeito_qualidade"]]


# ---------------------------------------------------------------------------
# Conjunto inteiro, cache e trava de gasto
# ---------------------------------------------------------------------------


def _frame(task, n_per_split=30):
    rows = []
    for split in ("treino", "validacao", "teste"):
        for i in range(n_per_split):
            bad = i % 3 == 0
            row = {"review_id": f"{split}-{i}", "split": split}
            if task == TASK_SENTIMENT:
                row["texto"] = f"produto ruim {i}" if bad else f"gostei {i}"
                row["sentimento"] = "negativa" if bad else "positiva"
            else:
                row["texto"] = f"atrasou {i}" if bad else f"quebrou {i}"
                for label in LABELS:
                    row[label] = 0
                row["atraso" if bad else "defeito_qualidade"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_split_usa_cache_e_retoma(tmp_path):
    frame = _frame(TASK_SENTIMENT)
    part = frame[frame["split"] == "validacao"].reset_index(drop=True)
    config = llm.RunConfig(TASK_SENTIMENT, batch_size=7)
    client = FakeClient(TASK_SENTIMENT, _sentiment_answer)
    run = llm.run_split(
        part, config, "validacao", llm.Classifier(client, config), 3, 10.0, tmp_path, lambda _: None
    )
    assert len(client.calls) == 5  # 30 avaliações em lotes de 7
    assert run.predictions == list(part["sentimento"])

    # Simula queda no meio da gravação: última linha cortada.
    path = llm.cache_path(config, "validacao", tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20], encoding="utf-8")
    client2 = FakeClient(TASK_SENTIMENT, _sentiment_answer)
    run2 = llm.run_split(
        part, config, "validacao", llm.Classifier(client2, config), 3, 10.0, tmp_path, print
    )
    assert len(client2.calls) == 1  # só o lote perdido é refeito
    assert run2.predictions == run.predictions

    # Tudo no cache: roda sem cliente nenhum.
    run3 = llm.run_split(part, config, "validacao", None, 3, 10.0, tmp_path, lambda _: None)
    assert run3.predictions == run.predictions and run3.new_cost_usd == 0


def test_trava_de_gasto_interrompe_e_guarda_o_que_foi_pago(tmp_path):
    frame = _frame(TASK_SENTIMENT)
    part = frame[frame["split"] == "validacao"].reset_index(drop=True)
    config = llm.RunConfig(TASK_SENTIMENT, batch_size=5)
    client = FakeClient(TASK_SENTIMENT, _sentiment_answer)
    # Cada chamada falsa custa US$ 0,0002; o limite estoura na 2ª.
    with pytest.raises(llm.SpendLimitReached):
        llm.run_split(
            part,
            config,
            "validacao",
            llm.Classifier(client, config),
            1,
            0.0003,
            tmp_path,
            lambda _: None,
        )
    assert len(llm.load_cache(llm.cache_path(config, "validacao", tmp_path))) == 2


def test_chave_ausente_nao_vaza_nada(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / ".env-inexistente")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY=sua_chave"):
        llm.make_client()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    results = tmp_path / "resultados.json"
    monkeypatch.setattr(llm, "save_result", partial(save_result, path=results))
    monkeypatch.setattr(llm, "load_results", partial(load_results, results))
    monkeypatch.setattr(llm, "write_results_markdown", lambda: None)
    monkeypatch.setattr(llm, "RUNS_LOG", tmp_path / "llm_execucoes.csv")
    return tmp_path, results


@pytest.mark.parametrize("task", [TASK_SENTIMENT, TASK_PROBLEMS])
def test_ponta_a_ponta(task, isolated):
    tmp_path, results_path = isolated
    answer = _sentiment_answer if task == TASK_SENTIMENT else _problem_answer
    client = FakeClient(task, answer)
    evaluated = llm.run_task(
        task, include_test=True, client=client, frame=_frame(task), cache_dir=tmp_path / "cache"
    )
    assert set(evaluated) == {"validacao", "teste"}
    stored = load_results(results_path)[task]["teste"]["llm_zero_shot"]
    assert stored["n"] == 30 and stored["falhas"] == 0
    assert stored["modelo"] == "Claude Haiku 5.5 (zero-shot)"
    assert stored["chamadas"] == 2  # 30 avaliações em lotes de 25
    assert stored["custo_total_usd"] == pytest.approx(2 * (1000 * 0.10 + 200 * 0.50) / 1e6)
    assert stored["hiperparametros"]["esforco"] == "low"
    if task == TASK_SENTIMENT:
        assert stored["f1_macro"] == pytest.approx(1.0)
    else:
        # "sem_reclamacao" junto com "atraso" é descartado pela regra do guia.
        per_class = stored["por_classe"]
        assert per_class["atraso"]["f1"] == 1.0 and per_class["defeito_qualidade"]["f1"] == 1.0
        assert per_class["sem_reclamacao"]["suporte"] == 0
    log = pd.read_csv(tmp_path / "llm_execucoes.csv")
    assert list(log["conjunto"]) == ["validacao", "teste"]


def test_teste_nao_pode_ser_medido_de_novo_com_outra_configuracao(isolated):
    tmp_path, _ = isolated
    frame = _frame(TASK_SENTIMENT)
    client = FakeClient(TASK_SENTIMENT, _sentiment_answer)
    kwargs = dict(client=client, frame=frame, cache_dir=tmp_path / "cache")
    llm.run_task(TASK_SENTIMENT, include_test=True, effort="low", **kwargs)
    # Mesma configuração: permitido (vem do cache, nenhuma chamada nova).
    calls = len(client.calls)
    llm.run_task(TASK_SENTIMENT, include_test=True, effort="low", **kwargs)
    assert len(client.calls) == calls
    # Outra configuração no teste: bloqueado; na validação, permitido.
    with pytest.raises(SystemExit, match="já foi medido"):
        llm.run_task(TASK_SENTIMENT, include_test=True, effort="medium", **kwargs)
    llm.run_task(TASK_SENTIMENT, include_test=False, effort="medium", **kwargs)
    llm.run_task(TASK_SENTIMENT, include_test=True, effort="medium", force=True, **kwargs)
    log = pd.read_csv(tmp_path / "llm_execucoes.csv")
    assert log["teste_forcado"].sum() == 1


def test_sem_chamadas_pendentes_nao_exige_chave(isolated, monkeypatch):
    tmp_path, _ = isolated
    frame = _frame(TASK_PROBLEMS)
    llm.run_task(
        TASK_PROBLEMS,
        include_test=False,
        client=FakeClient(TASK_PROBLEMS, _problem_answer),
        frame=frame,
        cache_dir=tmp_path / "cache",
    )

    def no_key():
        raise AssertionError("não deveria pedir a chave")

    monkeypatch.setattr(llm, "make_client", no_key)
    result = llm.run_task(
        TASK_PROBLEMS, include_test=False, frame=frame, cache_dir=tmp_path / "cache"
    )
    assert np.isfinite(result["validacao"]["f1_macro"])


def test_sdk_de_verdade_contra_servidor_local(monkeypatch):
    """Usa o pacote anthropic real (sem internet): confere o corpo enviado e a leitura."""
    pytest.importorskip("anthropic")
    import http.server

    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (nome exigido pelo http.server)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received["path"] = self.path
            received["body"] = json.loads(body)
            received["has_key"] = "chave-falsa" in str(self.headers)
            answer = {"respostas": [{"id": 1, "sentimento": "negativa"}]}
            payload = {
                "id": "msg_teste",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [
                    {"type": "thinking", "thinking": "...", "signature": "x"},
                    {"type": "text", "text": json.dumps(answer)},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 900,
                    "output_tokens_details": {"thinking_tokens": 12},
                },
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key="chave-falsa", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        )
        classifier = llm.Classifier(client, llm.RunConfig(TASK_SENTIMENT))
        result = classifier.classify(["r1"], ["veio quebrado"])
    finally:
        server.shutdown()

    assert result.predictions == ["negativa"] and result.failed_ids == []
    assert result.usage["cache_read_input_tokens"] == 900
    assert result.usage["thinking_tokens"] == 12
    body = received["body"]
    assert received["path"] == "/v1/messages" and received["has_key"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"]["required"] == ["respostas"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body and "thinking" not in body
