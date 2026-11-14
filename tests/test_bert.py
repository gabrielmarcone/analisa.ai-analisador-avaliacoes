"""Testes do módulo do BERTimbau.

As funções puras rodam em qualquer ambiente. O teste de ponta a ponta treina um
BERT minúsculo criado aqui mesmo (sem baixar nada) e só roda se PyTorch, Transformers e
Accelerate estiverem instalados (``requirements-bert.txt``); sem eles, é pulado.
"""

import json
from functools import partial

import numpy as np
import pandas as pd
import pytest

from reviews import bert
from reviews.annotation import LABELS, NO_COMPLAINT
from reviews.baseline import TASK_PROBLEMS, TASK_SENTIMENT
from reviews.evaluate import load_results, save_result

K = len(LABELS)
SEM = list(LABELS).index(NO_COMPLAINT)


def test_modulo_importa_sem_torch():
    # O import no topo do arquivo já prova; aqui conferimos as constantes básicas.
    assert bert.BASE_MODEL == "neuralmind/bert-base-portuguese-cased"
    assert set(bert.CONFIGS) == {TASK_SENTIMENT, TASK_PROBLEMS}


def test_task_labels():
    assert bert.task_labels(TASK_SENTIMENT) == ["negativa", "positiva"]
    assert bert.task_labels(TASK_PROBLEMS) == list(LABELS)
    with pytest.raises(ValueError, match="desconhecida"):
        bert.task_labels("outra")


def test_encode_targets():
    frame = pd.DataFrame({"sentimento": ["positiva", "negativa", "positiva"]})
    assert bert.encode_targets(TASK_SENTIMENT, frame).tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="fora da tarefa"):
        bert.encode_targets(TASK_SENTIMENT, pd.DataFrame({"sentimento": ["neutra"]}))

    problems = pd.DataFrame([[1] + [0] * (K - 1)], columns=list(LABELS))
    targets = bert.encode_targets(TASK_PROBLEMS, problems)
    assert targets.dtype == np.float32 and targets.shape == (1, K)


def test_logits_sentimento():
    pred = bert.logits_to_predictions(TASK_SENTIMENT, np.array([[2.0, -1.0], [0.1, 0.3]]))
    assert pred.tolist() == ["negativa", "positiva"]
    with pytest.raises(ValueError, match="n×2"):
        bert.logits_to_predictions(TASK_SENTIMENT, np.zeros((1, 3)))


def test_logits_problemas_seguem_as_regras_do_guia():
    logits = np.full((3, K), -4.0)
    logits[0, 0] = 3.0  # atraso acima do limiar
    logits[1, 2] = -0.5  # ninguém passa de 0,5: fica o mais provável
    logits[2, 0] = 2.0
    logits[2, SEM] = 3.0  # sem_reclamacao não pode acompanhar outro
    pred = bert.logits_to_predictions(TASK_PROBLEMS, logits)
    assert pred[0].tolist() == [1] + [0] * (K - 1)
    assert pred[1, 2] == 1 and pred[1].sum() == 1
    assert pred[2, 0] == 1 and pred[2, SEM] == 0


def test_warmup_steps():
    # 100 exemplos, lote 32 → 4 passos/época × 3 épocas = 12 passos; 10% → 1
    assert bert.warmup_steps(100, 32, 3, 0.1) == 1
    assert bert.warmup_steps(750, 16, 12, 0.1) == 56  # 47 passos × 12 = 564


def test_repo_id():
    assert bert.repo_id("gabriel", "problemas") == "gabriel/bertimbau-olist-problemas"
    for bad in ("", "a/b", " gabriel"):
        with pytest.raises(ValueError):
            bert.repo_id(bad, "sentimento")


def test_model_card_com_resultados(tmp_path):
    path = tmp_path / "resultados.json"
    por_classe = {
        label: {"precisao": 0.5, "revocacao": 0.5, "f1": 0.5, "suporte": 3} for label in LABELS
    }
    for key, f1 in (("baseline_tfidf", 0.68), ("bertimbau", 0.74)):
        save_result(
            "problemas", "teste", key, {"f1_macro": f1, "n": 500, "por_classe": por_classe}, path
        )
    card = bert.model_card("problemas", load_results(path), "gabriel/bertimbau-olist-problemas")
    assert card.startswith("---\nlanguage: pt\nlicense: cc-by-nc-sa-4.0\n")
    assert "| BERTimbau ajustado | teste | 0,740 | 500 |" in card
    assert "| Baseline TF-IDF | teste | 0,680 | 500 |" in card
    assert "| Atraso na entrega | `atraso` | 0,500 | 3 |" in card
    assert 'repo = "gabriel/bertimbau-olist-problemas"' in card
    assert "https://huggingface.co/gabriel/bertimbau-olist-sentimento" in card
    assert "se nenhuma passa, entra a mais provável" in card
    assert "não comercial" in card
    # Sem LLM nem latência registrados, não há parágrafo de comparação.
    assert "No mesmo teste" not in card and "ms por avaliação" not in card

    llm = {"f1_macro": 0.88, "n": 500, "modelo": "Claude Haiku 5.5 (zero-shot)"}
    save_result("problemas", "teste", "llm_zero_shot", llm, path)
    card = bert.model_card("problemas", load_results(path), "gabriel/bertimbau-olist-problemas")
    assert "| Claude Haiku 5.5 (zero-shot) | teste | 0,880 | 500 |" in card
    assert "chega a F1 macro 0,880; este modelo, a 0,740." in card


def test_model_cards_com_os_resultados_do_projeto():
    """Os cards publicados saem do resultados.json versionado; nada é digitado à mão."""
    results = load_results(bert.REPORTS_DIR / "resultados.json")
    if not results:
        pytest.skip("resultados.json ausente")
    for task in (TASK_SENTIMENT, TASK_PROBLEMS):
        card = bert.model_card(task, results, f"gmarconems/bertimbau-olist-{task}")
        test = results[task]["teste"]
        ours = bert.fmt_br(test["bertimbau"]["f1_macro"])
        assert f"| BERTimbau ajustado | teste | {ours} |" in card
        assert f"| {test['llm_zero_shot']['modelo']} | teste |" in card
        assert "a validação escolheu a época" in card
        assert "{" not in card.split("```")[0]  # nenhum campo do modelo ficou sem preencher


def test_model_card_sem_resultados():
    card = bert.model_card("sentimento", {}, "u/bertimbau-olist-sentimento")
    assert "Sem resultados registrados" in card


def test_publicar_exige_token(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(bert, "ENV_FILE", tmp_path / ".env")  # não lê o .env de quem roda
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        bert.publish("gabriel")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        bert.update_cards("gabriel")


def test_token_vem_do_env(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_do_arquivo\n", encoding="utf-8")
    monkeypatch.setattr(bert, "ENV_FILE", env)
    assert bert._hf_token() == "hf_do_arquivo"


# ---------------------------------------------------------------------------
# Ponta a ponta com um BERT minúsculo (só onde há PyTorch)
# ---------------------------------------------------------------------------

WORDS_GOOD = ["chegou", "rapido", "perfeito", "recomendo", "otimo", "lindo"]
WORDS_BAD = ["atrasou", "quebrado", "defeito", "pessimo", "demorou", "rachado"]


@pytest.fixture
def tiny_bert(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS_GOOD, *WORDS_BAD, "e", "veio"]
    folder = tmp_path / "tiny-bert"
    folder.mkdir()
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    # Transformers 5 recebe o vocabulário como dicionário; o 4.x, como arquivo.
    if int(transformers.__version__.split(".")[0]) >= 5:
        tokenizer = transformers.BertTokenizerFast(vocab={word: i for i, word in enumerate(vocab)})
    else:
        tokenizer = transformers.BertTokenizerFast(vocab_file=str(folder / "vocab.txt"))
    assert tokenizer.convert_tokens_to_ids("perfeito") != tokenizer.unk_token_id
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        initializer_range=0.2,  # o padrão (0,02) é pequeno demais para um modelo tão estreito
    )
    transformers.set_seed(0)
    transformers.BertModel(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


def _sentences(rng, words, n):
    return [" ".join(rng.choice(words, 5)) for _ in range(n)]


def _frame(task):
    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("treino", 40), ("validacao", 12), ("teste", 12)):
        good, bad = _sentences(rng, WORDS_GOOD, n), _sentences(rng, WORDS_BAD, n)
        for i, text in enumerate(good + bad):
            row = {"review_id": f"{split}-{i}", "texto": text, "split": split}
            if task == TASK_SENTIMENT:
                row["sentimento"] = "positiva" if i < n else "negativa"
            else:
                for label in LABELS:
                    row[label] = 0
                row[NO_COMPLAINT if i < n else "defeito_qualidade"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("task", [TASK_SENTIMENT, TASK_PROBLEMS])
def test_ponta_a_ponta_com_bert_minusculo(task, tiny_bert, tmp_path, monkeypatch):
    # O Trainer do Transformers exige o accelerate (requirements-bert.txt). A demo só
    # faz previsão e não precisa dele, então sem ele este teste é pulado, não quebra.
    pytest.importorskip("accelerate")
    results_path = tmp_path / "resultados.json"
    monkeypatch.setattr(bert, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(bert, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(bert, "save_result", partial(save_result, path=results_path))
    monkeypatch.setattr(bert, "write_results_markdown", lambda: None)

    config = bert.TrainConfig(epochs=8, learning_rate=5e-3, batch_size=8, patience=8)
    evaluated = bert.run_task(
        task, include_test=True, base_model=tiny_bert, config=config, frame=_frame(task)
    )

    assert set(evaluated) == {"validacao", "teste"}
    # Separação trivial: o modelo precisa acertar as duas classes presentes.
    # (Na tarefa de problemas a F1 macro inclui 6 categorias sem exemplos, por isso
    # olhamos a F1 de cada classe que existe nos dados.)
    per_class = evaluated["teste"]["por_classe"]
    present = [label for label, values in per_class.items() if values["suporte"] > 0]
    assert len(present) == 2
    assert all(per_class[label]["f1"] > 0.9 for label in present), per_class
    assert evaluated["teste"]["n"] == 24

    saved = bert.model_dir(task)
    assert (saved / "config.json").is_file() and (saved / "tokenizer.json").is_file()
    assert not (saved / "checkpoints").exists()  # checkpoints apagados após o treino

    history = pd.read_csv(tmp_path / "reports" / f"bertimbau_{task}_historico.csv")
    assert list(history.columns) == ["epoca", "f1_macro_validacao"] and len(history) >= 1

    stored = load_results(results_path)[task]["teste"]["bertimbau"]
    assert stored["modelo"] == "BERTimbau ajustado"
    assert stored["latencia_ms_por_avaliacao"] > 0
    assert stored["hiperparametros"]["modelo_base"] == tiny_bert

    # O modelo salvo recarrega e prevê com os rótulos certos.
    tokenizer, model = bert.load_model(task, str(saved))
    assert model.config.id2label[0] == bert.task_labels(task)[0]
    logits = bert.predict_logits(
        model,
        tokenizer,
        ["chegou rapido perfeito recomendo otimo", "atrasou quebrado defeito pessimo rachado"],
    )
    pred = bert.logits_to_predictions(task, logits)
    if task == TASK_SENTIMENT:
        assert pred.tolist() == ["positiva", "negativa"]
    else:
        assert pred[0, SEM] == 1 and pred[1, list(LABELS).index("defeito_qualidade")] == 1


def test_publicar_envia_pasta_e_model_card(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", token))

        def create_repo(self, repo, **kwargs):
            calls.append(("create_repo", repo, kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setattr(bert, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(bert, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(bert, "load_results", lambda: {})
    folder = bert.model_dir("sentimento")
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")

    urls = bert.publish("gabriel", tasks=["sentimento"])

    assert urls == ["https://huggingface.co/gabriel/bertimbau-olist-sentimento"]
    assert (folder / "README.md").read_text(encoding="utf-8").startswith("---\nlanguage: pt")
    create = next(c for c in calls if c[0] == "create_repo")
    assert create[1] == "gabriel/bertimbau-olist-sentimento" and create[2]["exist_ok"] is True
    upload = next(c for c in calls if c[0] == "upload_folder")[1]
    assert upload["folder_path"] == str(folder) and "checkpoints/*" in upload["ignore_patterns"]

    # Sem modelo treinado, não publica nada.
    with pytest.raises(SystemExit, match="Treine antes"):
        bert.publish("gabriel", tasks=["problemas"])


def test_atualizar_cards_envia_so_o_readme(tmp_path, monkeypatch, capsys):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", token))

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(bert, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(bert, "load_results", lambda: {})

    # Simulação: mostra os cards e não precisa de token nem chama o Hub.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert bert.update_cards("gabriel", dry_run=True) == []
    out = capsys.readouterr().out
    assert "gabriel/bertimbau-olist-sentimento · README.md" in out
    assert "gabriel/bertimbau-olist-problemas · README.md" in out
    assert calls == []

    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    urls = bert.update_cards("gabriel")
    assert urls == [
        "https://huggingface.co/gabriel/bertimbau-olist-sentimento",
        "https://huggingface.co/gabriel/bertimbau-olist-problemas",
    ]
    assert "hf_fake" not in capsys.readouterr().out  # o token nunca é impresso
    uploads = [c[1] for c in calls if c[0] == "upload_file"]
    assert [u["repo_id"] for u in uploads] == [
        "gabriel/bertimbau-olist-sentimento",
        "gabriel/bertimbau-olist-problemas",
    ]
    for upload in uploads:  # só o README.md; nenhum peso é reenviado
        assert upload["path_in_repo"] == "README.md" and upload["repo_type"] == "model"
        assert upload["path_or_fileobj"].decode("utf-8").startswith("---\nlanguage: pt")


# ---------------------------------------------------------------------------
# O código do "Como usar" dos cards
# ---------------------------------------------------------------------------


def _card_code(task: str, repo: str) -> str:
    card = bert.model_card(task, {}, repo)
    return card.split("```python\n", 1)[1].split("```", 1)[0]


def test_exemplos_do_card_batem_com_a_conferencia_da_demo():
    """As saídas mostradas no card são as do modelo publicado, gravadas pela demo."""
    path = bert.REPORTS_DIR / "demo_conferencia.json"
    if not path.is_file():
        pytest.skip("demo_conferencia.json ausente")
    examples = {e["texto"]: e for e in json.loads(path.read_text(encoding="utf-8"))["exemplos"]}

    sentiment = _card_code(TASK_SENTIMENT, "u/r")
    text = "Ótimo. Chegou antes do prazo e a qualidade é excelente. Recomendo!"
    shown = examples[text]
    assert f'"{text}"' in sentiment and shown["sentimento"] == "positiva"
    assert f"'score': {shown['prob_sentimento']['positiva']:.3f}" in sentiment

    problems = _card_code(TASK_PROBLEMS, "u/r")
    broken = "Veio quebrado. A tampa veio rachada e o aparelho não liga. Quero devolver."
    assert f"# {examples[broken]['problemas']!r}" in problems
    vague = examples["ruim"]
    assert f"# {vague['problemas']!r}" in problems
    assert max(vague["prob_problemas"].values()) < 0.5  # o exemplo ilustra a regra 2
    best = max(vague["prob_problemas"], key=vague["prob_problemas"].get)
    assert best == vague["problemas"][0]
    assert f"com {vague['prob_problemas'][best]:.2f}".replace(".", ",") in problems


@pytest.fixture
def saved_classifiers(tiny_bert, tmp_path):
    """Modelos de classificação minúsculos, salvos como os publicados no Hub."""
    folders = {}
    for task in (TASK_SENTIMENT, TASK_PROBLEMS):
        tokenizer, model = bert.load_model(task, tiny_bert)
        folder = tmp_path / f"publicado-{task}"
        model.save_pretrained(folder)
        tokenizer.save_pretrained(folder)
        # Barras normais: o caminho entra como texto no código do card, e as barras
        # invertidas do Windows (C:\Users\...) virariam escapes do Python.
        folders[task] = folder.as_posix()
    return folders


def test_codigo_do_card_de_sentimento_roda(saved_classifiers):
    code = _card_code(TASK_SENTIMENT, saved_classifiers[TASK_SENTIMENT])
    namespace = {}
    exec(code, namespace)  # o código do card, como está
    output = namespace["classifier"]("chegou rapido", truncation=True, max_length=128)
    assert len(output) == 1 and output[0]["label"] in {"negativa", "positiva"}


def test_codigo_do_card_de_problemas_segue_as_regras_do_projeto(saved_classifiers):
    import torch

    code = _card_code(TASK_PROBLEMS, saved_classifiers[TASK_PROBLEMS])
    namespace = {}
    exec(code, namespace)  # carrega o modelo salvo e roda os exemplos do card
    tipos_de_problema = namespace["tipos_de_problema"]
    labels, prob = tipos_de_problema("veio quebrado")
    assert list(prob) == list(LABELS) and labels

    # Com probabilidades escolhidas, o código do card dá o mesmo que o apply_rules.
    real_model = namespace["model"]
    cases = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1],  # dois tipos passam
            [0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.4, 0.2],  # nenhum passa: fica "outro"
            [0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.9],  # "sem reclamação" com outro: sai
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],  # só "sem reclamação"
            [0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # exatamente no limiar: os dois entram
        ]
    )

    class Output:
        def __init__(self, logits):
            self.logits = logits

    class FixedModel:
        config = real_model.config

        def __init__(self, probabilities):
            p = torch.from_numpy(np.asarray(probabilities, dtype=np.float64)).unsqueeze(0)
            self.logits = torch.log(p) - torch.log1p(-p)  # sigmoide inversa

        def __call__(self, **kwargs):
            return Output(self.logits)

    expected = bert.apply_rules(cases)
    for row, want in zip(cases, expected, strict=True):
        namespace["model"] = FixedModel(row)
        got, _ = tipos_de_problema("veio quebrado")
        assert got == [label for label, flag in zip(LABELS, want, strict=True) if flag], row
