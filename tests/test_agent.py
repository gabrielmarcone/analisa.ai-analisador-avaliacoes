"""Testes do agente de relatórios, sem rede: a API é simulada."""

import json
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reviews import agent, llm, sellers
from reviews.annotation import LABELS

from .test_sellers import seller_data

K = len(LABELS)
REF = {"avaliacoes": 1500, "pct": {label: 10.0 for label in LABELS}, "descricao": "ref. de teste"}


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
    # 28 não vale como arredondamento de 27,94: um inteiro tem que existir como inteiro.
    assert missing == ["28", "31,5", "2018"] and total == 3
    # Os números das regras do próprio prompt valem sem estar nas ferramentas.
    assert agent.unsupported_numbers("Nos últimos 3 meses, 2 pontos acima.", outputs) == ([], 2)


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


def _problems_output(counts):
    rows = [
        {"categoria": c, "nome": c.replace("_", " ").title(), "avaliacoes": n}
        for c, n in counts.items()
    ]
    for row in rows:
        row["posicao"] = 1 + sum(o["avaliacoes"] > row["avaliacoes"] for o in rows)
    return {"avaliacoes_negativas_classificadas": 100, "categorias": rows}


def test_ranking_de_problemas_conferido():
    outputs = [
        _problems_output(
            {"nao_recebido": 86, "atendimento": 54, "produto_diferente": 52, "incompleto": 54}
        )
    ]
    # Erro real visto num relatório: o 2º lugar é atendimento (empatado com incompleto).
    wrong = "Os problemas mais citados são não recebimento (36,4%) e produto errado ou diferente."
    found = agent.wrong_rankings(wrong, outputs)
    assert len(found) == 1 and "Nao Recebido" in found[0]
    ok = (
        "O problema mais citado é não recebido. "
        "Os mais citados são não recebimento e atendimento. "
        "Os mais citados são não recebido e pedido incompleto (empatado em 2º). "
        "Produto errado ou diferente está acima da referência."
    )
    assert agent.wrong_rankings(ok, outputs) == []
    assert agent.wrong_rankings("Produto errado ou diferente é o principal problema.", outputs)
    assert agent.wrong_rankings("O mais citado é X.", []) == []  # sem a ferramenta: não julga


def test_ranking_com_ordinais_e_com_numero_de_mais_citados():
    # Contagem real do vendedor 4a3ca931.
    counts = {
        "nao_recebido": 86,
        "atendimento": 54,
        "produto_diferente": 52,
        "defeito_qualidade": 50,
        "atraso": 39,
        "incompleto": 15,
        "outro": 15,
        "sem_reclamacao": 7,
    }
    outputs = [_problems_output(counts)]
    # Falso positivo real: o ordinal estava certo, mas era julgado como "o mais citado".
    ordinal = (
        '**Conferir separação e expedição.** "Produto errado ou diferente" é o terceiro '
        "mais citado e está 4,5 pontos percentuais acima da referência."
    )
    assert agent.wrong_rankings(ordinal, outputs) == []
    ok = (
        "Os três problemas mais citados são não recebido, atendimento e produto errado. "
        "Não recebido é o mais citado; atendimento é o segundo mais citado. "
        "Pedido incompleto é o sexto mais citado. Outro problema é o 7º mais citado."
    )  # incompleto e outro empatam em 6º: podem ser 6º ou 7º
    assert agent.wrong_rankings(ok, outputs) == []
    # Erro real do primeiro rascunho: pula o 2º (atendimento).
    draft = (
        '- O principal problema citado é "não recebido", seguido de "produto errado ou '
        'diferente" e "defeito ou má qualidade".'
    )
    assert "os 3 mais citados são" in agent.wrong_rankings(draft, outputs)[0]
    assert agent.wrong_rankings("Defeito é o segundo mais citado.", outputs) == [
        "«Defeito é o segundo mais citado.» (pela contagem, Defeito Qualidade é o 4º)"
    ]
    assert agent.wrong_rankings("Outro problema é o oitavo mais citado.", outputs)
    two = agent.wrong_rankings("Entre os dois mais citados está produto errado.", outputs)
    assert "os 2 mais citados são: Nao Recebido, Atendimento" in two[0]
    one = agent.wrong_rankings("Atendimento é o problema mais citado.", outputs)
    assert "o mais citado é: Nao Recebido)" in one[0]


def test_recomendacao_apoiada_so_em_problema_secundario_e_marcada():
    # Tabela real do vendedor 6560211a (posição, diferença para a referência).
    table = {
        "defeito_qualidade": (1, 20.0),
        "atendimento": (2, -3.2),
        "nao_recebido": (3, -13.7),
        "outro": (4, 7.6),
        "produto_diferente": (5, -1.8),
        "atraso": (6, -8.2),
        "incompleto": (7, -7.4),
        "sem_reclamacao": (8, 3.3),
    }
    rows = [
        {
            "categoria": c,
            "nome": c.replace("_", " ").title(),
            "posicao": pos,
            "diferenca_pontos_percentuais": diff,
        }
        for c, (pos, diff) in table.items()
    ]
    outputs = [{"avaliacoes_negativas_classificadas": 229, "categorias": rows}]
    report = (
        "## Resumo\n- Produto errado ou diferente está perto da referência.\n\n"
        "## Recomendações\n\n"
        "1. **Revisar defeitos antes do envio.** Defeito aparece em 37,1% das negativas.\n"
        "2. **Investigar as negativas com pedido no prazo.** A diferença é de 4,4 pontos.\n"
        "3. **Revisar a descrição dos anúncios.** Produto errado ou diferente aparece em "
        "15,7%, perto da referência, mas merece atenção.\n"
        '4. **Analisar "outro problema".** Está 7,6 pontos acima da referência.\n'
        "5. **Manter os prazos.** Seus atrasos estão abaixo do marketplace.\n"
    )
    weak = agent.weak_recommendations(report, outputs)
    assert weak == [
        "«Revisar a descrição dos anúncios.» "
        "(Produto Diferente: 5º mais citado, -1,8 pp da referência)"
    ]
    # Junto com um problema que tem apoio, passa; fora da seção, não é julgado.
    ok = report.replace("Produto errado ou diferente aparece", "Produto errado e defeito aparecem")
    assert agent.weak_recommendations(ok, outputs) == []
    assert agent.weak_recommendations(report.split("## Recomendações")[0], outputs) == []
    assert agent.weak_recommendations(report, []) == []


def test_aspas_que_so_nomeiam_categoria_nao_sao_citacao():
    outputs = [{"exemplos": [{"texto": "Esse produto veio com defeito"}]}]
    text = (
        'Em "defeito ou má qualidade" e "atendimento, cancelamento ou reembolso", um cliente '
        'disse: "Esse produto veio com defeito".'
    )
    assert agent.unsupported_quotes(text, outputs) == []
    assert agent.unsupported_quotes('"Esse produto veio sem defeito nenhum"', outputs)


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

    def __init__(self, script, bad_texts=()):
        self.script = list(script)
        self.bad_texts = set(bad_texts)  # textos que o classificador "não consegue" classificar
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
            if any(item["texto"] in self.bad_texts for item in items):
                answers = answers[:-1]  # resposta fora do formato: falta um id
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
def data():
    return seller_data()


def _tools(data):
    neg = sellers.commented_negatives(data.reviews, "A")
    matrix = np.zeros((len(neg), K), dtype=int)
    matrix[:, 0] = 1
    return agent.SellerTools(data, "A", neg, matrix, REF)


GOOD = "## Resumo\n- 50% das avaliações são negativas, com nota média 3."


def _required_tools(prefix="r"):
    return [_tool_use(f"{prefix}{i}", name) for i, name in enumerate(agent.REQUIRED_TOOLS)]


def _classification(tools):
    return agent.Classification(tools.negatives, tools.matrix, agent.Cost(0.0, 0.0))


def test_laco_com_ferramentas_paralelas_erro_e_reescrita(data):
    tools = _tools(data)
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor"), _tool_use("t2", "entregas")]),
        ("tool_use", [_tool_use("t3", "exemplos_avaliacoes", {"categoria": "inexistente"})]),
        ("end_turn", [_text("## Resumo\n- 77% das avaliações são negativas.")]),
        ("end_turn", [_text("Aqui está o relatório:\n\n" + GOOD)]),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, tools)

    assert result.body == GOOD and result.findings.unsupported == [] and result.revisions == 1
    assert result.calls == 4 and result.findings.numbers_checked == 2
    # Não consultou o que o relatório precisa: fica registrado depois da única reescrita.
    assert result.findings.missing_tools == [
        "comparar_com_marketplace",
        "evolucao_mensal",
        "problemas_relatados",
    ]
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
    summary = json.loads(results[0]["content"])
    assert summary["avaliacoes"] == 6 and summary["avaliacoes_com_comentario"] == 4
    delivery = json.loads(results[1]["content"])
    assert delivery["atraso_nos_pedidos"]["vendedor"]["pct_com_atraso"] == 50.0

    error = client.agent_calls[2]["messages"][4]["content"][0]
    assert error["is_error"] is True and "desconhecida" in error["content"]

    revision = client.agent_calls[3]["messages"][-1]["content"]
    assert "77" in revision and "Reescreva" in revision
    assert [c["ferramenta"] for c in tools.calls] == ["resumo_vendedor", "entregas"]


def test_citacao_inventada_pede_reescrita(data):
    invented = [_text('## Resumo\n- Um cliente disse: "o produto chegou quebrado e sem nota".')]
    fixed = [_text(GOOD)]
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor")]),
        ("end_turn", invented),
        ("end_turn", fixed),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, _tools(data))
    assert result.revisions == 1 and result.findings.unsupported_quotes == []
    assert "trechos entre aspas" in client.agent_calls[2]["messages"][-1]["content"]
    assert any("chegou quebrado" in reason for reason in result.revision_reasons)


def test_recomendacao_sem_apoio_pede_reescrita_e_fica_registrada(data):
    tools = _tools(data)
    tools.matrix[:] = 0
    tools.matrix[0, [0, 4, 5]] = 1  # atraso, defeito e atendimento
    tools.matrix[1, 1] = 1  # não recebido; o resto empata em 5º, 10 pontos abaixo da ref.
    weak = "## Recomendações\n\n1. **Reduzir pedidos incompletos.** Vale conferir."
    script = [
        ("tool_use", [_tool_use("t1", "problemas_relatados")]),
        ("end_turn", [_text(weak)]),
        ("end_turn", [_text(weak)]),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, tools)
    assert result.revisions == 1 and len(result.findings.weak_recommendations) == 1
    assert "Reduzir pedidos incompletos." in result.findings.weak_recommendations[0]
    assert "recomendações se apoiam" in client.agent_calls[2]["messages"][-1]["content"]
    note = agent.methodology(_classification(tools), result)
    assert "⚠️ Não conferem: " in note and "recomendações «Reduzir" in note


def test_numeros_sem_fonte_ficam_marcados_depois_da_reescrita(data):
    bad = [_text("## Resumo\n- 99% negativas.")]
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor")]),
        ("end_turn", bad),
        ("end_turn", bad),
    ]
    tools = _tools(data)
    result = agent.run_agent(FakeClient(script), tools)
    assert result.findings.unsupported == ["99"] and result.revisions == 1
    note = agent.methodology(_classification(tools), result)
    assert "⚠️ Não conferem: ferramentas não consultadas" in note and "números 99" in note
    assert "Tudo confere" not in note
    assert "custou" not in note  # sem custo da referência, sem a frase


def test_paradas_inesperadas_e_limites(data, monkeypatch):
    with pytest.raises(RuntimeError, match="max_tokens"):
        agent.run_agent(FakeClient([("max_tokens", [_text("...")])]), _tools(data))
    with pytest.raises(llm.SpendLimitReached):
        agent.run_agent(FakeClient([("end_turn", [_text(GOOD)])]), _tools(data), limit_usd=1e-6)
    monkeypatch.setattr(agent, "MAX_CALLS", 2)
    loop = [("tool_use", [_tool_use(f"t{i}", "resumo_vendedor")]) for i in range(3)]
    with pytest.raises(RuntimeError, match="2 chamadas"):
        agent.run_agent(FakeClient(loop), _tools(data))


def test_relatorio_completo_ponta_a_ponta(data, tmp_path, monkeypatch):
    script = [("tool_use", _required_tools()), ("end_turn", [_text(GOOD)])]
    client = FakeClient(script)
    reference = (REF, agent.Cost(0.0805, 0.0))
    path = agent.generate(
        "A",
        data,
        client=client,
        output_dir=tmp_path / "rel",
        cache_dir=tmp_path / "cache",
        reference=reference,
    )
    assert client.classifier_calls == 1  # 2 negativas com comentário do vendedor A, um lote
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Relatório de avaliações · vendedor `A`")
    assert GOOD in text and "## Como este relatório foi feito" in text
    assert "Tudo confere" in text and "\r\n" not in text
    assert "As 2 do vendedor foram classificadas" in text
    assert "aplicado às 1500 avaliações" in text
    assert "A referência do marketplace custou US$ 0,0805 uma vez" in text
    audit = json.loads((tmp_path / "rel" / "vendedor_A_ferramentas.json").read_text("utf-8"))
    assert audit["numeros_sem_fonte"] == [] and len(audit["ferramentas"]) == 5
    assert audit["motivos_das_reescritas"] == [] and audit["ferramentas_nao_consultadas"] == []
    assert audit["negativas_sem_classificacao"] == []
    problems = audit["ferramentas"][4]["saida"]
    assert problems["categorias"][0]["categoria"] == "atraso"
    cost = audit["custo_usd"]
    assert cost["classificacao"] == pytest.approx(0.0002) and cost["agente"] > 0
    assert cost["total"] == pytest.approx(cost["classificacao"] + cost["agente"])
    assert cost["gasto_nesta_execucao"] == pytest.approx(cost["total"])  # nada no cache ainda

    # De novo: a classificação vem do cache (nenhuma chamada nova ao classificador).
    client2 = FakeClient([("tool_use", _required_tools()), ("end_turn", [_text(GOOD)])])
    agent.generate(
        "A",
        data,
        client=client2,
        output_dir=tmp_path / "rel",
        cache_dir=tmp_path / "cache",
        reference=reference,
    )
    assert client2.classifier_calls == 0
    again = json.loads((tmp_path / "rel" / "vendedor_A_ferramentas.json").read_text("utf-8"))
    assert again["custo_usd"]["classificacao"] == pytest.approx(0.0002)  # custo do zero
    assert again["custo_usd"]["gasto_nesta_execucao"] == pytest.approx(again["custo_usd"]["agente"])


def test_negativa_que_o_classificador_nao_classifica_fica_de_fora(data, tmp_path):
    """Antes, a falha virava "outro problema" e inflava essa categoria sem aviso."""
    negatives = sellers.commented_negatives(data.reviews, "A")
    bad = negatives["texto"].iloc[0]
    client = FakeClient([], bad_texts={bad})
    result = agent.classify_negatives(
        data.reviews, "A", client, cache_dir=tmp_path, progress=lambda _: None
    )
    assert result.unclassified == (negatives["review_id"].iloc[0],)
    assert list(result.negatives["review_id"]) == [negatives["review_id"].iloc[1]]
    assert result.matrix.shape == (1, K) and result.matrix[0, LABELS.index("outro")] == 0
    out = sellers.problems(result.negatives, result.matrix, REF, len(result.unclassified))
    assert out["avaliacoes_negativas_classificadas"] == 1
    assert out["negativas_sem_classificacao"] == 1
    assert "outro" not in [r["categoria"] for r in out["categorias"] if r["avaliacoes"]]

    tools = agent.SellerTools(data, "A", result.negatives, result.matrix, REF, 1)
    note = agent.methodology(result, agent.AgentResult("", 1, {}, 0))
    assert "Das 2 do vendedor, 1 foram classificadas (1 ficaram sem classificação" in note
    assert tools.run("problemas_relatados", {})["negativas_sem_classificacao"] == 1
    # Sem cache e sem poder chamar a API (conferência), a classificação não roda.
    with pytest.raises(SystemExit, match="não estão no cache"):
        agent.classify_negatives(data.reviews, "A", cache_dir=tmp_path / "vazio", offline=True)


def test_tabela_de_problemas_conferida_linha_a_linha():
    rows = [
        ("nao_recebido", "Não recebido", 1, 86, 36.4, 35.5, 0.9),
        ("atendimento", "Atendimento, cancelamento ou reembolso", 2, 54, 22.9, 27.7, -4.8),
        ("produto_diferente", "Produto errado ou diferente", 3, 52, 22.0, 17.5, 4.5),
    ]
    outputs = [
        {
            "avaliacoes_negativas_classificadas": 236,
            "categorias": [
                {
                    "categoria": c,
                    "nome": nome,
                    "posicao": pos,
                    "avaliacoes": n,
                    "pct_das_negativas_com_comentario": pct,
                    "pct_referencia_marketplace": ref,
                    "diferenca_pontos_percentuais": diff,
                }
                for c, nome, pos, n, pct, ref, diff in rows
            ],
        }
    ]
    table = (
        "## Principais problemas\n\n"
        "| Problema | Avaliações | % | Referência | Diferença |\n|---|---|---|---|---|\n"
        "| Não recebido | 86 | 36,4% | 35,5% | +0,9 p.p. |\n"
        "| Atendimento, cancelamento ou reembolso | 54 | 22,9% | 27,7% | −4,8 p.p. |\n"
        "| Produto errado ou diferente | 52 | 22,0% | 17,5% | +4,5 p.p. |\n\n## Entregas\n| 99 |"
    )
    assert agent.wrong_table_rows(table, outputs) == []
    swapped = table.replace("| 54 | 22,9%", "| 52 | 22,9%")
    assert agent.wrong_table_rows(swapped, outputs)[0].startswith("«Atendimento")
    sign = table.replace("−4,8 p.p.", "+4,8 p.p.")
    assert "+4,8" in agent.wrong_table_rows(sign, outputs)[0]
    lines = table.split("\n")
    lines[4], lines[6] = lines[6], lines[4]
    assert agent.wrong_table_rows("\n".join(lines), outputs) == [
        "a tabela não segue a ordem do mais citado para o menos citado"
    ]


def test_citacao_atribuida_ao_tipo_de_problema_e_a_nota_certos():
    negatives = pd.DataFrame(
        {
            "texto": ["Comprei e até hoje não chegou nada", "Veio quebrado, péssima qualidade"],
            "review_score": [1, 2],
        }
    )
    matrix = np.zeros((2, K), dtype=int)
    matrix[0, LABELS.index("nao_recebido")] = 1
    matrix[1, LABELS.index("defeito_qualidade")] = 1
    ok = (
        '- Não recebido: "Comprei e até hoje não chegou nada" (nota 1)\n'
        '- **Defeito ou má qualidade:** *"Veio quebrado, péssima qualidade"* (nota 2)\n'
        'Um cliente disse: "Veio quebrado, péssima qualidade".'  # linha sem tipo: não julga
    )
    assert agent.misattributed_quotes(ok, negatives, matrix) == []
    wrong_type = '- Atraso na entrega: "Comprei e até hoje não chegou nada"'
    assert agent.misattributed_quotes(wrong_type, negatives, matrix) == [
        "«Comprei e até hoje não chegou nada» (a avaliação é de: Não recebido)"
    ]
    wrong_note = '- Não recebido: "Comprei e até hoje não chegou nada" (nota 2)'
    assert (
        "nota 2; a avaliação tem nota 1"
        in agent.misattributed_quotes(wrong_note, negatives, matrix)[0]
    )


def test_relatorio_sem_as_ferramentas_obrigatorias_pede_reescrita(data):
    script = [
        ("tool_use", [_tool_use("t1", "resumo_vendedor")]),
        ("end_turn", [_text(GOOD)]),
        ("tool_use", _required_tools()),
        ("end_turn", [_text(GOOD)]),
    ]
    client = FakeClient(script)
    result = agent.run_agent(client, _tools(data))
    assert "consulte também estas ferramentas" in client.agent_calls[2]["messages"][-1]["content"]
    assert result.revisions == 1 and result.findings.flags() == []


def test_conferir_relatorio_publicado_sem_api(data, tmp_path):
    reference = (REF, agent.Cost(0.0805, 0.0))
    path = agent.generate(
        "A",
        data,
        client=FakeClient([("tool_use", _required_tools()), ("end_turn", [_text(GOOD)])]),
        output_dir=tmp_path / "rel",
        cache_dir=tmp_path / "cache",
        reference=reference,
    )
    original = path.read_text(encoding="utf-8")
    check = agent.verify_report(path, data, reference, tmp_path / "cache")
    assert check.problems == [] and check.findings.numbers_checked == 2

    # Atualizar reescreve só a metodologia e a auditoria; o texto do agente fica intacto.
    audit_path = tmp_path / "rel" / "vendedor_A_ferramentas.json"
    path.write_text(original.replace("Tudo confere.", "texto antigo."), encoding="utf-8")
    agent.verify_report(path, data, reference, tmp_path / "cache", update=True)
    assert path.read_text(encoding="utf-8") == original
    assert json.loads(audit_path.read_text("utf-8"))["conferido_em"]

    # Um número trocado no texto é pego, sem chamar a API.
    path.write_text(original.replace("50% das", "60% das"), encoding="utf-8")
    assert agent.verify_report(path, data, reference, tmp_path / "cache").findings.unsupported == [
        "60"
    ]

    # Se os dados mudaram, as ferramentas não batem com o registrado e nada é reescrito.
    path.write_text(original, encoding="utf-8")
    audit = json.loads(audit_path.read_text("utf-8"))
    audit["ferramentas"][0]["saida"]["avaliacoes"] = 999
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    tampered = path.read_text(encoding="utf-8").replace("Tudo confere.", "x.")
    path.write_text(tampered, encoding="utf-8")
    check = agent.verify_report(path, data, reference, tmp_path / "cache", update=True)
    assert check.tool_mismatches == ["resumo_vendedor"] and check.problems
    assert path.read_text(encoding="utf-8") == tampered


def test_referencia_usa_o_mesmo_classificador_e_o_cache(tmp_path):
    rows = []
    for split, n in (("treino", 30), ("validacao", 10), ("teste", 20)):
        for i in range(n):
            row = {"review_id": f"{split}-{i}", "texto": f"texto {split} {i}", "split": split}
            row.update({label: 0 for label in LABELS})
            row["atraso"] = 1
            rows.append(row)
    frame = pd.DataFrame(rows)
    # A trava de gasto vale para a soma das partes: 2 lotes do treino (US$ 0,0004) já
    # consomem quase tudo, e o primeiro lote da validação passa do limite.
    with pytest.raises(llm.SpendLimitReached):
        agent.build_reference(
            FakeClient([]),
            frame,
            limit_usd=0.0005,
            cache_dir=tmp_path / "trava",
            progress=lambda _: None,
        )

    client = FakeClient([])  # o cliente falso responde "atraso" para tudo
    reference, cost = agent.build_reference(
        client, frame, cache_dir=tmp_path, progress=lambda _: None
    )
    assert reference["avaliacoes"] == 60 and reference["pct"]["atraso"] == 100.0
    assert "mesmo classificador" in reference["descricao"]
    assert cost.total_usd == pytest.approx(4 * 0.0002) and cost.new_usd == cost.total_usd
    assert client.classifier_calls == 4  # lotes de 25: 2 (treino) + 1 + 1
    again = FakeClient([])
    _, cost_again = agent.build_reference(again, frame, cache_dir=tmp_path, progress=lambda _: None)
    assert again.classifier_calls == 0  # tudo do cache
    assert cost_again.total_usd == pytest.approx(cost.total_usd) and cost_again.new_usd == 0
    assert str(cost_again) == "US$ 0.0008 (gasto agora: US$ 0.0000)"


def test_sdk_de_verdade_devolve_o_raciocinio_ao_modelo(data, monkeypatch):
    """Com o pacote anthropic real (servidor local): o bloco de raciocínio volta intacto."""
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(agent, "REQUIRED_TOOLS", ("resumo_vendedor",))  # o foco é o SDK
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
        result = agent.run_agent(client, _tools(data))
    finally:
        server.shutdown()

    assert result.body == GOOD and result.findings.unsupported == []
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
