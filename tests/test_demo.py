"""Testes da demo (Fase 7), sem rede.

As regras de previsão são testadas com logits escolhidos à mão, então rodam em
qualquer ambiente. Os testes que carregam modelos de verdade usam BERTs minúsculos
criados aqui mesmo e só rodam onde há PyTorch e Transformers.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reviews import bert, demo
from reviews import demo_ui as ui
from reviews.annotation import CATEGORIES, LABELS
from reviews.baseline import TASK_PROBLEMS, TASK_SENTIMENT, apply_rules
from reviews.data import build_text, clean_text
from reviews.evaluate import fmt_br, load_results, save_result

K = len(LABELS)
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Texto e regras de previsão (sem PyTorch)
# ---------------------------------------------------------------------------


def test_texto_igual_ao_da_base():
    title, comment = "  Atrasou!  ", "não chegou\r\nno prazo   "
    assert demo.prepare_text(title, comment) == build_text(clean_text(title), clean_text(comment))
    assert demo.prepare_text(None, "Ótimo") == "Ótimo"
    assert demo.prepare_text("Péssimo", "") == "Péssimo"
    with pytest.raises(ValueError, match="Escreva"):
        demo.prepare_text("  ", None)
    with pytest.raises(ValueError, match="1000"):
        demo.prepare_text("", "a" * (demo.MAX_COMMENT_CHARS + 1))


class FakeClassifier(demo.Classifier):
    """Mesmo código de previsão, com logits escolhidos à mão no lugar dos modelos."""

    def __init__(self, sentiment, problems):
        self._s, self._p = np.asarray(sentiment, float), np.asarray(problems, float)

    def sentiment_logits(self, texts):
        return self._s[: len(texts)]

    def problem_logits(self, texts):
        return self._p[: len(texts)]


def test_previsao_segue_as_regras_da_avaliacao():
    # Linha 0: negativa, atraso e defeito acima de 50%. Linha 1: negativa, nenhum acima
    # de 50% (fica o mais provável). Linha 2: positiva (problemas não se aplicam).
    sentiment = [[2.0, -1.0], [0.5, 0.0], [-3.0, 3.0]]
    problems = np.full((3, K), -4.0)
    problems[0, [0, 4]] = 2.0
    problems[1, LABELS.index("atendimento")] = -0.2
    problems[0, LABELS.index("sem_reclamacao")] = 1.0  # nunca junto com outro tipo
    clf = FakeClassifier(sentiment, problems)
    out = clf.predict(["a", "b", "c"])

    assert [p.sentiment for p in out] == ["negativa", "negativa", "positiva"]
    assert out[0].problems == ["atraso", "defeito_qualidade"]
    assert out[1].problems == ["atendimento"]
    assert out[2].problems is None and out[2].problem_probabilities is None
    for p in out:
        assert sum(p.sentiment_probabilities.values()) == pytest.approx(1.0)
    probs = 1 / (1 + np.exp(-problems))
    expected = apply_rules(probs)
    for i in (0, 1):
        assert out[i].problems == [LABELS[j] for j in range(K) if expected[i, j]]
        assert out[i].problem_probabilities["atraso"] == pytest.approx(probs[i, 0])
    # Mesma resposta que a avaliação daria a esses logits.
    assert [p.sentiment for p in out] == list(
        bert.logits_to_predictions(TASK_SENTIMENT, np.asarray(sentiment))
    )


# ---------------------------------------------------------------------------
# A leitura em HTML (sem PyTorch)
# ---------------------------------------------------------------------------


def _prediction(sentiment="negativa", neg=0.97, probs=None, text="texto"):
    probs = probs if probs is not None else {k: 0.05 for k in LABELS}
    chosen = apply_rules(np.array([[probs[k] for k in LABELS]]))[0]
    negative = sentiment == "negativa"
    return demo.Prediction(
        text=text,
        sentiment=sentiment,
        sentiment_probabilities={"negativa": neg, "positiva": 1 - neg},
        problems=[k for k, c in zip(LABELS, chosen, strict=True) if c] if negative else None,
        problem_probabilities=probs if negative else None,
    )


def _rows(page: str, weak: bool) -> list[str]:
    marker = '<div class="lab-medida fraca">' if weak else '<div class="lab-medida">'
    return [chunk.split("</span>")[0] for chunk in page.split(marker)[1:]]


def _names(page: str, weak: bool) -> list[str]:
    return [row.split('lab-medida-nome">')[1] for row in _rows(page, weak)]


def test_porcentagens_sem_fingir_certeza():
    assert ui.pct(0.9977) == "&gt;99%" and ui.pct(0.995) == "&gt;99%"
    assert ui.pct(0.994) == "99%" and ui.pct(0.845) == "85%" and ui.pct(0.5) == "50%"
    assert ui.pct(0.004) == "&lt;1%" and ui.pct(0.005) == "1%"


def test_leitura_mostra_os_tipos_detectados_em_ordem():
    probs = {k: 0.05 for k in LABELS}
    probs |= {"nao_recebido": 0.975, "atraso": 0.84, "atendimento": 0.68, "defeito_qualidade": 0.3}
    page = ui.reading_html(_prediction(probs=probs))
    assert "Avaliação negativa com 3 tipos de problema" in page
    # Mais provável primeiro; nomes inteiros, mesmo os que têm vírgula.
    assert _names(page, weak=False) == [
        CATEGORIES["nao_recebido"],
        CATEGORIES["atraso"],
        CATEGORIES["atendimento"],
    ]
    assert len(_rows(page, weak=True)) == len(LABELS) - 3
    assert "Ver os outros 5 tipos" in page
    assert "Confiança baixa" not in page and "Negativa" in page and "97%" in page

    one = ui.reading_html(_prediction(probs={**{k: 0.05 for k in LABELS}, "incompleto": 0.92}))
    assert "Avaliação negativa com 1 tipo de problema" in one


def test_leitura_avisa_quando_a_confianca_e_baixa():
    probs = {k: 0.04 for k in LABELS} | {"outro": 0.30, "defeito_qualidade": 0.25}
    prediction = _prediction(neg=0.94, probs=probs)
    assert prediction.problems == ["outro"]  # a regra da avaliação fica com o mais provável
    page = ui.reading_html(prediction)
    assert ui.low_confidence(prediction)
    assert "Confiança baixa" in page and "Tipo mais provável" in page
    assert "sem tipo claro" in page and _names(page, weak=False) == [CATEGORIES["outro"]]


def test_leitura_positiva_nao_mostra_tipos():
    page = ui.reading_html(_prediction(sentiment="positiva", neg=0.0023))
    assert "Avaliação positiva" in page and "&gt;99%" in page
    assert "lab-medida" not in page and "Confiança baixa" not in page


def test_texto_digitado_nao_vira_html():
    attack = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    page = ui.reading_html(_prediction(text=attack))
    assert "<script>" not in page and "<img" not in page
    assert "&lt;script&gt;" in page
    assert "<script>" not in ui.reading_error_html("<script>x</script>")
    assert "<script>" not in ui.specimen_html(_prediction(text="<script>x</script>"))


def test_exemplos_e_leitura_da_tela_inicial():
    names = [name for name, _, _ in demo.EXAMPLES]
    assert len(set(names)) == len(names) and 0 <= demo.SPECIMEN < len(demo.EXAMPLES)
    assert all(demo.prepare_text(t, c) for _, t, c in demo.EXAMPLES)
    assert set(ui.DESCRIPTIONS) == set(LABELS)
    types = ui.types_html()
    assert all(CATEGORIES[k] in types for k in LABELS)


def test_conferencia_compara_com_o_registrado(tmp_path):
    frames = {
        TASK_SENTIMENT: pd.DataFrame(
            {"texto": ["ruim", "bom"], "sentimento": ["negativa", "positiva"]}
        ),
        TASK_PROBLEMS: pd.DataFrame({"texto": ["atrasou"], **{c: [0] for c in LABELS}}),
    }
    frames[TASK_PROBLEMS]["atraso"] = 1
    rows = max(2, len(demo.EXAMPLES))  # o teste e os exemplos passam pelo mesmo classificador
    problems = np.full((rows, K), -4.0)
    problems[:, 0] = 3.0
    sentiment = [[2.0, -2.0], [-2.0, 2.0]] + [[2.0, -2.0]] * (rows - 2)
    clf = FakeClassifier(sentiment, problems)
    results = tmp_path / "resultados.json"
    save_result(TASK_SENTIMENT, "teste", "bertimbau", {"f1_macro": 1.0, "n": 2}, path=results)
    save_result(TASK_PROBLEMS, "teste", "bertimbau", {"f1_macro": 0.125, "n": 1}, path=results)
    output = tmp_path / "conferencia.json"
    report = demo.check_models(clf, results, output, frames)
    assert report["confere"] is True
    assert report["tarefas"][TASK_SENTIMENT]["diferenca"] == 0.0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["exemplos"]) == len(demo.EXAMPLES)
    assert saved["exemplos"][0]["exemplo"] == demo.EXAMPLES[0][0]
    assert saved["exemplos"][0]["texto"] == demo.prepare_text(*demo.EXAMPLES[0][1:])

    save_result(TASK_SENTIMENT, "teste", "bertimbau", {"f1_macro": 0.9, "n": 2}, path=results)
    assert demo.check_models(clf, results, output, frames)["confere"] is False


def test_abas_de_relatorios_e_resultados():
    choices = demo.report_choices()
    assert list(choices) == [
        "4a3ca931 · cama, mesa e banho",
        "6560211a · relógios e presentes",
        "cc419e06 · beleza e saúde",
    ]
    text = demo.results_markdown()
    for name in (
        "TF-IDF + Regressão Logística",
        "BERTimbau ajustado",
        "Claude Haiku 5.5 (zero-shot)",
    ):
        assert text.count(f"| {name} |") == 2  # uma linha em cada tarefa
    assert "0,881" in text and "0,947" in text


def test_numeros_da_tela_inicial_vem_dos_arquivos():
    results = load_results()
    page = ui.results_section_html(results)
    sentiment = results["sentimento"]["teste"]["bertimbau"]
    problems = results["problemas"]["teste"]["bertimbau"]
    accuracy = ui.percent(sentiment["acuracia"] * 100)
    assert accuracy == "95,6%" and accuracy in page
    for value in (sentiment["f1_macro"], problems["f1_macro"]):
        assert fmt_br(value, 3) in page
    assert fmt_br(results["problemas"]["teste"]["llm_zero_shot"]["f1_macro"], 3) in page
    # Melhor e pior tipo, e o "sem reclamação" (F1 0 em poucos casos) dito com clareza.
    assert "0,89 em atraso na entrega" in page and "0,49 em outro problema" in page
    support = problems["por_classe"]["sem_reclamacao"]["suporte"]
    assert f"({support} casos no teste)" in page
    assert ui.results_section_html({}) == ""


def test_numeros_dos_relatorios_batem_com_o_texto():
    for label, path in demo.report_choices().items():
        summary = ui.report_summary(path.with_name(f"{path.stem}_ferramentas.json"))
        assert summary["confere"] is True and summary["numeros_conferidos"] > 0
        assert label.startswith(summary["codigo"])
        text = path.read_text(encoding="utf-8")
        # Cada número do topo aparece no próprio relatório (que passou pela conferência).
        assert f"{summary['avaliacoes']} avaliações" in text
        for value in (summary["nota"], summary["nota_marketplace"]):
            assert fmt_br(value, 2) in text
        for value in (summary["negativas"], summary["negativas_marketplace"], summary["atraso"]):
            assert ui.percent(value) in text
        head, readings, body = demo.report_parts(path)
        assert head.startswith("## Relatório de avaliações") and body.startswith("### Resumo")
        assert number_in(readings, summary["avaliacoes"]) and "Tudo confere" in readings
    broken = dict(summary, confere=False)
    assert "Tudo confere" not in ui.report_readings_html(broken)
    assert demo.report_parts(None)[0] == "_Nenhum relatório._"


def number_in(page: str, n: int) -> bool:
    return ui.number(n) in page


# ---------------------------------------------------------------------------
# Pasta do Space e publicação
# ---------------------------------------------------------------------------


def test_pasta_do_space_tem_so_o_necessario(tmp_path):
    files = demo.build_space(tmp_path / "space", user="fulano")
    expected = {"app.py", "requirements.txt", "README.md", "reports/resultados.json"}
    expected |= {f"src/reviews/{name}" for name in demo.SPACE_MODULES}
    expected |= {
        f"reports/relatorios/{p.name}" for p in (ROOT / "reports" / "relatorios").glob("vendedor_*")
    }
    assert set(files) == expected
    assert not any(part in f for f in files for part in (".env", "data/", "models/", "annotations"))
    space = tmp_path / "space"
    assert 'Classifier(user="fulano")' in (space / "app.py").read_text(encoding="utf-8")
    readme = (space / "README.md").read_text(encoding="utf-8")
    yaml = pytest.importorskip("yaml")
    meta = yaml.safe_load(readme.split("---")[1])
    assert meta["sdk"] == "gradio" and meta["sdk_version"] == demo.GRADIO_VERSION
    assert meta["app_file"] == "app.py" and meta["license"] == "cc-by-nc-sa-4.0"
    assert meta["models"] == [
        "fulano/bertimbau-olist-sentimento",
        "fulano/bertimbau-olist-problemas",
    ]
    assert len(meta["short_description"]) <= 60
    reqs = (space / "requirements.txt").read_text(encoding="utf-8")
    assert (
        f"transformers=={demo.TRANSFORMERS_VERSION}" in reqs and "gradio" not in reqs.split("#")[-1]
    )

    # A demo importa só módulos que vão para o Space (Python isolado, sem o repositório).
    code = (
        "import sys; sys.path.insert(0, 'src'); import reviews.demo, reviews.bert; "
        "import json; print(json.dumps(sorted(m.split('.')[1] for m in sys.modules "
        "if m.startswith('reviews.'))))"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=space, capture_output=True, text=True, check=True
    )
    imported = set(json.loads(run.stdout))
    assert imported <= {name.removesuffix(".py") for name in demo.SPACE_MODULES}


def test_publicar_exige_token_e_envia_a_pasta(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("dotenv")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(demo, "PROJECT_ROOT", tmp_path)  # sem ler o .env de verdade
    (tmp_path / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        demo.publish("fulano", staging=tmp_path / "s")

    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def create_repo(self, repo, **kwargs):
            calls["create"] = (repo, kwargs)

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            calls["files"] = sorted(p.name for p in Path(kwargs["folder_path"]).rglob("*.py"))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_teste_falso")
    url = demo.publish("fulano", staging=tmp_path / "s")
    assert url == "https://huggingface.co/spaces/fulano/analisa-ai"
    repo, kwargs = calls["create"]
    assert repo == "fulano/analisa-ai" and kwargs["repo_type"] == "space"
    assert kwargs["space_sdk"] == "gradio" and kwargs["exist_ok"] is True
    assert calls["upload"]["repo_type"] == "space" and calls["upload"]["repo_id"] == repo
    assert "app.py" in calls["files"] and "demo.py" in calls["files"]
    with pytest.raises(ValueError):
        demo.publish("fulano/outro", staging=tmp_path / "s")


# ---------------------------------------------------------------------------
# Com modelos de verdade (BERTs minúsculos), onde há PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_models(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "chegou", "atrasou", "quebrado"]
    if int(transformers.__version__.split(".")[0]) >= 5:
        tokenizer = transformers.BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    else:  # pragma: no cover - Transformers 4
        (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=160,
    )
    base = tmp_path / "base"
    transformers.set_seed(0)
    transformers.BertModel(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    for task, variable in (
        (TASK_SENTIMENT, "DEMO_MODELO_SENTIMENTO"),
        (TASK_PROBLEMS, "DEMO_MODELO_PROBLEMAS"),
    ):
        tok, model = bert.load_model(task, str(base))
        folder = tmp_path / task
        model.save_pretrained(folder)
        tok.save_pretrained(folder)
        monkeypatch.setenv(variable, str(folder))
    return tmp_path


def test_classificador_carrega_modelos_e_responde(tiny_models):
    clf = demo.Classifier(user="ignorado")  # as variáveis DEMO_MODELO_* têm prioridade
    assert clf.sentiment[1].config.id2label[0] == "negativa"
    assert clf.problems[1].config.problem_type == "multi_label_classification"
    one = clf.classify("Atrasou", "chegou quebrado")
    batch = clf.predict([one.text, "chegou"])
    assert batch[0].sentiment == one.sentiment
    assert batch[0].sentiment_probabilities == pytest.approx(one.sentiment_probabilities, abs=1e-5)
    long_text = "chegou " * 600  # passa do limite de 128 tokens: o tokenizador corta
    assert clf.predict([long_text])[0].sentiment in ("negativa", "positiva")


def test_app_responde_pela_api_do_gradio(tiny_models):
    gradio_client = pytest.importorskip("gradio_client")
    clf = demo.Classifier()
    app = demo.build_app(clf)
    demo.launch(
        app, server_name="127.0.0.1", server_port=None, prevent_thread_lock=True, quiet=True
    )
    try:
        client = gradio_client.Client(app.local_url, verbose=False)
        # Só a classificação fica na API; os eventos da tela são internos.
        assert list(client.view_api(print_info=False, return_format="dict")["named_endpoints"]) == [
            "/classificar"
        ]
        out = client.predict("Atrasou", "chegou quebrado", api_name="/classificar")
        assert out == clf.classify("Atrasou", "chegou quebrado").to_json()
        assert out["texto"] == "Atrasou. chegou quebrado"
        with pytest.raises(Exception, match="Escreva"):
            client.predict("", "   ", api_name="/classificar")
    finally:
        app.close()


def test_space_montado_roda_sozinho(tiny_models, tmp_path):
    """A pasta do Space, sem o repositório, monta a demo com os modelos indicados."""
    pytest.importorskip("gradio")
    space = tmp_path / "space"
    demo.build_space(space, user="fulano")
    code = (
        "import sys; sys.path.insert(0, 'src'); from reviews.demo import Classifier, build_app; "
        "app = build_app(Classifier(user='fulano')); "
        "from reviews import demo_ui; assert 'lab-leitura' in demo_ui.css(); demo_ui.theme(); "
        "from reviews.demo import report_choices; print(len(report_choices()))"
    )
    env = {**os.environ, "PYTHONPATH": ""}
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=space, capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().splitlines()[-1] == "3"
