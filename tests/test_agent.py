"""Testes do agente de relatórios, sem rede: a API é simulada."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from reviews import agent, llm, sellers
from reviews.annotation import LABELS

from .test_sellers import _items, _products, _reviews, _sellers

K = len(LABELS)
REF = {"avaliacoes": 1500, "pct": {label: 10.0 for label in LABELS}}


# ---------------------------------------------------------------------------
# Verificador de números
# ---------------------------------------------------------------------------


def test_extrai_numeros_ignorando_ids_e_marcadores():
    text = "1. Vendedor 4a3ca931 tem 27,9% de negativas (845 avaliações) em 2017-03.\n2. Nota 3.52"
    tokens = [t for t, _, _ in agent.extract_numbers(text)]
    assert tokens == ["27,9", "845", "2017", "03", "3.52"]
    assert agent.extract_numbers("27,9")[0][1:] == (27.9, 1)


def test_numeros_aceitos_e_rejeitados():
    outputs = [
        {"pct": 27.94, "diferenca": -2.9, "nota": 3.524, "meses": [{"mes": "2017-03"}]},
        {"exemplos": [{"texto": "não coube o notebook de 15,6'"}]},
    ]
    ok = "Negativas: 27,9% (2,9 pontos abaixo), nota 3,52 em março de 2017; citação: 15,6'."
    assert agent.unsupported_numbers(ok, outputs) == ([], 5)
    bad = "Negativas: 28% e 31,5% em 2018."
    missing, total = agent.unsupported_numbers(bad, outputs)
    assert missing == ["31,5", "2018"] and total == 3  # 28 = arredondamento de 27,94


def test_citacoes_conferidas_palavra_por_palavra():
    outputs = [
        {"categorias": [{"categoria": "produto_diferente", "nome": "Produto errado ou diferente"}]},
        {
            "exemplos": [
                {"texto": "Comprei 3 cadeiras, todas vieram faltando peças. Não consegui trocar."}
            ]
        },
    ]
    ok = (
        'Em "produto diferente" (curto, ignorado) e "Produto errado ou diferente", um cliente '
        'disse: "Comprei 3 cadeiras, todas vieram faltando peças... Não consegui trocar." e '
        "“todas vieram faltando peças”."
    )
    assert agent.unsupported_quotes(ok, outputs) == []
    bad = 'Um cliente disse: "Comprei 3 cadeiras e nenhuma chegou montada."'
    assert agent.unsupported_quotes(bad, outputs) == [
        "Comprei 3 cadeiras e nenhuma chegou montada."
    ]


# ---------------------------------------------------------------------------
# Cliente falso: classificador (saída JSON) e agente (ferramentas)
# ---------------------------------------------------------------------------


def _usage(inp=1000, out=200):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens_details=SimpleNamespace(thinking_tokens=0),
    )


def _tool_use(tid, name, tool_input=None):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=tool_input or {})


def _text(text):
    return SimpleNamespace(type="text", text=text)


class FakeClient:
    """Responde às chamadas do classificador e segue um roteiro para o agente."""

    def __init__(self, script):
        self.script = list(script)
        self.agent_calls = []
        self.classifier_calls = 0
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        if "tools" not in kwargs:  # classificador da Fase 5
            with self.lock:
                self.classifier_calls += 1
            items = json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])
            answers = [{"id": item["id"], "categorias": ["atraso"]} for item in items]
            return SimpleNamespace(
                content=[_text(json.dumps({"respostas": answers}))],
                stop_reason="end_turn",
                usage=_usage(),
            )
        # cópia rasa: o laço continua acrescentando mensagens à mesma lista
        self.agent_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        stop_reason, content = self.script.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


@pytest.fixture
def df():
    return sellers.build_seller_reviews(_reviews(), _items(), _products(), _sellers())


def _tools(df):
    neg = df[(df["seller_id"] == "A") & (df["sentimento"] == "negativa")].reset_index(drop=True)
    matrix = np.zeros((len(neg), K), dtype=int)
    matrix[:, 0] = 1
    return agent.SellerTools(df, "A", neg, matrix, REF)


GOOD = "## Resumo\n- 50% das avaliações são negativas, com nota média 3."


def test_laco_com_ferramentas_paralelas_erro_e_reescrita(df):
    tools = _tools(df)
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor"), _tool_use("t2", "entregas")]),
        ("tool_use", [_tool_use("t3", "exemplos_avaliacoes", {"categoria": "inexistente"})]),
        ("end_turn", [_text("## Resumo\n- 77% das avaliações são negativas.")]),
        ("end_turn", [_text("Aqui está o relatório:\n\n" + GOOD)]),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, tools)

    assert result.body == GOOD and result.unsupported == [] and result.revisions == 1
    assert result.calls == 4 and result.numbers_checked == 2
    assert result.cost_usd == pytest.approx(4 * (1000 * 0.10 + 200 * 0.50) / 1e6)

    first = client.agent_calls[0]
    assert first["model"] == "claude-haiku-5-5"
    assert first["output_config"] == {"effort": "medium"}
    assert "temperature" not in first and first["tools"] == agent.TOOLS

    # 2ª chamada: resposta anterior devolvida inteira + resultados das duas ferramentas
    second = client.agent_calls[1]["messages"]
    assert second[1] == {"role": "assistant", "content": script[0][1]}
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert json.loads(results[0]["content"])["avaliacoes_com_comentario"] == 4

    error = client.agent_calls[2]["messages"][4]["content"][0]
    assert error["is_error"] is True and "desconhecida" in error["content"]

    revision = client.agent_calls[3]["messages"][-1]["content"]
    assert "77" in revision and "Reescreva" in revision
    assert [c["ferramenta"] for c in tools.calls] == ["resumo_vendedor", "entregas"]


def test_citacao_inventada_pede_reescrita(df):
    invented = [_text('## Resumo\n- Um cliente disse: "o produto chegou quebrado e sem nota".')]
    fixed = [_text(GOOD)]
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor")]),
        ("end_turn", invented),
        ("end_turn", fixed),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, _tools(df))
    assert result.revisions == 1 and result.unsupported_quotes == []
    assert "trechos entre aspas" in client.agent_calls[2]["messages"][-1]["content"]
    assert result.revision_reasons and "chegou quebrado" in result.revision_reasons[0]


def test_numeros_sem_fonte_ficam_marcados_depois_da_reescrita(df):
    bad = [_text("## Resumo\n- 99% negativas.")]
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor")]),
        ("end_turn", bad),
        ("end_turn", bad),
    ]
    result = agent.run_agent(FakeClient(script), _tools(df))
    assert result.unsupported == ["99"] and result.revisions == 1
    note = agent.methodology(0.0, result, 2)
    assert "⚠️ Sem correspondência nos dados: números 99" in note


def test_paradas_inesperadas_e_limites(df, monkeypatch):
    with pytest.raises(RuntimeError, match="max_tokens"):
        agent.run_agent(FakeClient([("max_tokens", [_text("...")])]), _tools(df))
    with pytest.raises(llm.SpendLimitReached):
        agent.run_agent(FakeClient([("end_turn", [_text(GOOD)])]), _tools(df), limit_usd=1e-6)
    monkeypatch.setattr(agent, "MAX_CALLS", 2)
    loop = [("tool_use", [_tool_use(f"t{i}", "resumo_vendedor")]) for i in range(3)]
    with pytest.raises(RuntimeError, match="2 chamadas"):
        agent.run_agent(FakeClient(loop), _tools(df))


def test_relatorio_completo_ponta_a_ponta(df, tmp_path, monkeypatch):
    monkeypatch.setattr(sellers, "marketplace_problem_reference", lambda: REF)
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor"), _tool_use("t2", "problemas_relatados")]),
        ("end_turn", [_text(GOOD)]),
    ]
    client = FakeClient(script)
    path = agent.generate(
        "A", df, client=client, output_dir=tmp_path / "rel", cache_dir=tmp_path / "cache"
    )
    assert client.classifier_calls == 1  # 2 negativas do vendedor A, um lote
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Relatório de avaliações · vendedor `A`")
    assert GOOD in text and "## Como este relatório foi feito" in text
    assert "todos conferem" in text and "\r\n" not in text
    audit = json.loads((tmp_path / "rel" / "vendedor_A_ferramentas.json").read_text("utf-8"))
    assert audit["numeros_sem_fonte"] == [] and len(audit["ferramentas"]) == 2
    assert audit["motivos_das_reescritas"] == []
    problems = audit["ferramentas"][1]["saida"]
    assert problems["categorias"][0]["categoria"] == "atraso"

    # De novo: a classificação vem do cache (nenhuma chamada nova ao classificador).
    client2 = FakeClient(
        [("tool_use", [_tool_use("t1", "resumo_vendedor")]), ("end_turn", [_text(GOOD)])]
    )
    agent.generate(
        "A", df, client=client2, output_dir=tmp_path / "rel", cache_dir=tmp_path / "cache"
    )
    assert client2.classifier_calls == 0


def test_sdk_de_verdade_devolve_o_raciocinio_ao_modelo(df):
    """Com o pacote anthropic real (servidor local): o bloco de raciocínio volta intacto."""
    anthropic = pytest.importorskip("anthropic")
    import http.server

    bodies = []
    replies = [
        {
            "content": [
                {"type": "thinking", "thinking": "", "signature": "assinatura-xyz"},
                {"type": "tool_use", "id": "toolu_1", "name": "resumo_vendedor", "input": {}},
            ],
            "stop_reason": "tool_use",
        },
        {"content": [{"type": "text", "text": GOOD}], "stop_reason": "end_turn"},
    ]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = replies[len(bodies) - 1]
            payload = {
                "id": f"msg_{len(bodies)}",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
                **reply,
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(
            api_key="chave-falsa", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        )
        result = agent.run_agent(client, _tools(df))
    finally:
        server.shutdown()

    assert result.body == GOOD and result.unsupported == []
    echoed = bodies[1]["messages"][1]
    assert echoed["role"] == "assistant"
    assert echoed["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "assinatura-xyz",
    }
    assert echoed["content"][1]["type"] == "tool_use" and echoed["content"][1]["id"] == "toolu_1"
    result_block = bodies[1]["messages"][2]["content"][0]
    assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "toolu_1"
    assert bodies[0]["tools"][5]["input_schema"]["properties"]["categoria"]["enum"] == list(LABELS)
