"""Demo em Gradio (Fase 7): classifica avaliações com os modelos publicados no Hub.

Páginas (abas):
- **Início:** o que a ferramenta faz, com uma leitura de exemplo feita pelo modelo na hora
  e os números de ``reports/resultados.json`` e das auditorias dos relatórios.
- **Analisar:** sentimento (BERTimbau) e, se a avaliação for negativa, os tipos de
  problema (BERTimbau), com as probabilidades. O texto passa pela mesma limpeza da base,
  a mesma tokenização e as mesmas regras da avaliação (``bert.predict_logits`` e
  ``bert.logits_to_predictions``), então a demo dá o que foi medido no teste.
- **Relatórios:** os relatórios de vendedores da Fase 6, prontos (nenhuma chamada de API).
- **Resultados:** a comparação dos três modelos no teste.

A API ``/classificar`` devolve a previsão em JSON. Nada aqui chama API paga: o LLM aparece
só nos resultados. A demo não guarda o que é digitado. O visual (tema, CSS e HTML) fica
em ``demo_ui.py`` e ``demo.css``.

A demo roda só no computador (127.0.0.1, sem link público): desde 2026 o Hugging Face
cobra para hospedar Spaces em Gradio, e o projeto não usa serviço pago. O README mostra a
demo em prints e num vídeo curto.

Uso:
    python -m reviews.demo                     # abre em http://127.0.0.1:7860
    python -m reviews.demo conferir            # mede os modelos do Hub no teste (sem API)

PyTorch, Transformers e Gradio são importados só onde são usados.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from reviews import bert
from reviews import demo_ui as ui
from reviews.annotation import LABELS as PROBLEM_LABELS
from reviews.baseline import TASK_PROBLEMS, TASK_SENTIMENT
from reviews.data import build_text, clean_text
from reviews.evaluate import comparison_markdown, fmt_br, load_results
from reviews.labels import NEGATIVE
from reviews.paths import REPORTS_DIR, RESULTS_FILE
from reviews.splits import TEST

HF_USER = "gmarconems"
GITHUB_URL = "https://github.com/gabrielmarcone/analisa.ai-analisador-avaliacoes"
MAX_TITLE_CHARS = 100
MAX_COMMENT_CHARS = 1000
REPORTS_SUBDIR = "relatorios"
CHECK_FILE = REPORTS_DIR / "demo_conferencia.json"
CHECK_TOLERANCE = 0.002  # diferença de F1 aceita entre a medição original (GPU) e a da demo

# Exemplos inventados (não saem da base, para não expor nenhum cliente real):
# (nome do botão, título, comentário). O "ruim" mostra o caso de confiança baixa.
EXAMPLES = [
    ("Elogio", "Ótimo", "Chegou antes do prazo e a qualidade é excelente. Recomendo!"),
    ("Não chegou", "", "Comprei há um mês e até hoje não recebi. Ninguém responde meus e-mails."),
    (
        "Veio quebrado",
        "Veio quebrado",
        "A tampa veio rachada e o aparelho não liga. Quero devolver.",
    ),
    ("Faltou item", "", "Pedi duas cadeiras e só chegou uma."),
    (
        "Diferente do anúncio",
        "Diferente do anúncio",
        "A cor é outra e o tamanho é menor do que o da foto.",
    ),
    ("Só “ruim”", "", "ruim"),
]
SPECIMEN = 1  # o exemplo lido na tela inicial
SCROLL_TOP = "() => { window.scrollTo({ top: 0 }); }"
# No celular a leitura fica abaixo do formulário: depois de analisar, a tela desce até ela.
SCROLL_TO_READING = (
    "() => { if (window.innerWidth < 760) "
    "document.getElementById('lab-leitura')?.scrollIntoView({ behavior: 'smooth' }); }"
)


def model_source(task: str, user: str = HF_USER) -> str:
    """Repositório do Hub (ou pasta local, pelas variáveis DEMO_MODELO_*) de cada tarefa."""
    variable = {TASK_SENTIMENT: "DEMO_MODELO_SENTIMENTO", TASK_PROBLEMS: "DEMO_MODELO_PROBLEMAS"}
    return os.environ.get(variable[task]) or bert.repo_id(user, task)


def prepare_text(title: str | None, comment: str | None) -> str:
    """Mesmo texto que os modelos viram no treino: título + comentário, limpos."""
    title, comment = clean_text(title), clean_text(comment)
    if len(title) > MAX_TITLE_CHARS or len(comment) > MAX_COMMENT_CHARS:
        raise ValueError(
            f"Use até {MAX_TITLE_CHARS} caracteres no título e {MAX_COMMENT_CHARS} no comentário."
        )
    text = build_text(title, comment)
    if not text:
        raise ValueError("Escreva o comentário (ou o título) da avaliação.")
    return text


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class Prediction:
    text: str
    sentiment: str
    sentiment_probabilities: dict[str, float]
    problems: list[str] | None  # None quando a avaliação é positiva
    problem_probabilities: dict[str, float] | None

    def to_json(self) -> dict:
        return {
            "texto": self.text,
            "sentimento": self.sentiment,
            "prob_sentimento": self.sentiment_probabilities,
            "problemas": self.problems,
            "prob_problemas": self.problem_probabilities,
        }


class Classifier:
    """Os dois BERTimbau publicados, com as regras de previsão da avaliação."""

    def __init__(self, user: str = HF_USER):
        self.sentiment = bert.load_model(TASK_SENTIMENT, model_source(TASK_SENTIMENT, user))
        self.problems = bert.load_model(TASK_PROBLEMS, model_source(TASK_PROBLEMS, user))

    def sentiment_logits(self, texts: Sequence[str]) -> np.ndarray:
        tokenizer, model = self.sentiment
        return bert.predict_logits(model, tokenizer, texts)

    def problem_logits(self, texts: Sequence[str]) -> np.ndarray:
        tokenizer, model = self.problems
        return bert.predict_logits(model, tokenizer, texts)

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        texts = list(texts)
        s_logits = self.sentiment_logits(texts)
        s_labels = bert.logits_to_predictions(TASK_SENTIMENT, s_logits)
        s_probs = _softmax(np.asarray(s_logits, dtype=np.float64))
        p_logits = self.problem_logits(texts)
        p_matrix = bert.logits_to_predictions(TASK_PROBLEMS, p_logits)
        p_probs = _sigmoid(np.asarray(p_logits, dtype=np.float64))
        sentiments = bert.task_labels(TASK_SENTIMENT)
        out = []
        for i, text in enumerate(texts):
            negative = s_labels[i] == NEGATIVE
            out.append(
                Prediction(
                    text=text,
                    sentiment=str(s_labels[i]),
                    sentiment_probabilities={
                        label: float(s_probs[i, j]) for j, label in enumerate(sentiments)
                    },
                    # O modelo de problemas só viu negativas: numa positiva, não se aplica.
                    problems=(
                        [label for j, label in enumerate(PROBLEM_LABELS) if p_matrix[i, j]]
                        if negative
                        else None
                    ),
                    problem_probabilities=(
                        {label: float(p_probs[i, j]) for j, label in enumerate(PROBLEM_LABELS)}
                        if negative
                        else None
                    ),
                )
            )
        return out

    def classify(self, title: str | None, comment: str | None) -> Prediction:
        return self.predict([prepare_text(title, comment)])[0]


# ---------------------------------------------------------------------------
# Conteúdo das abas
# ---------------------------------------------------------------------------


def report_choices(reports_dir: Path = REPORTS_DIR / REPORTS_SUBDIR) -> dict[str, Path]:
    """Rótulo da lista ("4a3ca931 · cama, mesa e banho") → arquivo do relatório."""
    choices = {}
    for path in sorted(Path(reports_dir).glob("vendedor_*.md")):
        summary = ui.report_summary(path.with_name(f"{path.stem}_ferramentas.json"))
        category = summary["categoria"] if summary else ""
        label = path.stem.removeprefix("vendedor_") + (f" · {category}" if category else "")
        choices[label] = path
    return choices


def results_markdown(results_file: Path = RESULTS_FILE) -> str:
    results = load_results(results_file)
    if not results:
        return "_Sem resultados registrados._"
    parts = [
        "Medidos uma única vez no conjunto de teste, com as mesmas funções de avaliação "
        "para os três modelos. A demo usa os BERTimbau; o LLM (Claude Haiku 5.5) só "
        "aparece aqui, porque cada chamada a ele tem custo."
    ]
    for task, title in ((TASK_PROBLEMS, "Tipos de problema"), (TASK_SENTIMENT, "Sentimento")):
        parts.append(f"### {title} · teste\n\n{comparison_markdown(results, task, TEST)}")
    return "\n\n".join(parts)


def _demote_headings(markdown: str) -> str:
    """Um nível abaixo (# vira ##): na página, o título principal é o da aba."""
    return "\n".join(f"#{line}" if line.startswith("#") else line for line in markdown.splitlines())


def report_parts(path: Path | None) -> tuple[str, str, str]:
    """Cabeçalho do relatório, números principais (da auditoria) e corpo, para a tela."""
    if path is None or not Path(path).is_file():
        return "_Nenhum relatório._", "", ""
    head, body = ui.split_report(Path(path).read_text(encoding="utf-8"))
    summary = ui.report_summary(Path(path).with_name(f"{Path(path).stem}_ferramentas.json"))
    return _demote_headings(head), ui.report_readings_html(summary), _demote_headings(body)


def launch(app, **kwargs):
    """Abre a demo com a identidade visual (tema, CSS e fontes) e o rodapé enxuto."""
    return app.launch(
        theme=ui.theme(),
        css=ui.css(),
        head=ui.HEAD,
        footer_links=["api", "settings"],
        **kwargs,
    )


def build_app(
    classifier: Classifier | None = None,
    reports_dir: Path = REPORTS_DIR / REPORTS_SUBDIR,
    results_file: Path = RESULTS_FILE,
):
    import gradio as gr

    classifier = classifier or Classifier()
    reports = report_choices(reports_dir)
    summaries = [
        ui.report_summary(p.with_name(f"{p.stem}_ferramentas.json")) for p in reports.values()
    ]
    _, specimen_title, specimen_comment = EXAMPLES[SPECIMEN]
    specimen = classifier.classify(specimen_title, specimen_comment)

    def run(title: str, comment: str) -> str:
        try:
            prediction = classifier.classify(title, comment)
        except ValueError as error:
            # O painel troca o resultado anterior pelo aviso: não sobra resposta de outro texto.
            return ui.reading_error_html(str(error))
        return ui.reading_html(prediction)

    def classificar(titulo: str, comentario: str) -> dict:
        """Classifica uma avaliação: sentimento e, se negativa, os tipos de problema."""
        try:
            return classifier.classify(titulo, comentario).to_json()
        except ValueError as error:
            raise gr.Error(str(error)) from None

    def show_report(label: str):
        return report_parts(reports.get(label))

    def fill(title: str, comment: str):
        return lambda: (title, comment)

    def go(tab: str):
        return lambda: gr.Tabs(selected=tab)

    hidden = {"api_visibility": "private"}
    with gr.Blocks(title="analisa.ai · avaliações de e-commerce", analytics_enabled=False) as app:
        gr.HTML(ui.header_html())
        with gr.Tabs(selected="inicio", elem_classes="lab-nav") as tabs:
            with gr.Tab("Início", id="inicio"):
                with gr.Row(equal_height=False, elem_classes="lab-heroi-linha"):
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML(ui.hero_html())
                        with gr.Row(elem_classes="lab-acoes"):
                            to_analyse = gr.Button(
                                "Analisar uma avaliação →", variant="primary", scale=0
                            )
                            to_report = gr.Button(
                                "Ver um relatório de vendedor", variant="secondary", scale=0
                            )
                        gr.HTML(ui.privacy_html())
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML(ui.specimen_html(specimen))
                gr.HTML(ui.types_html())
                gr.HTML(ui.steps_html())
                gr.HTML(ui.reports_section_html(summaries))
                with gr.Row(elem_classes="lab-acoes lab-acoes-secao"):
                    to_reports = gr.Button("Abrir os relatórios →", variant="secondary", scale=0)
                gr.HTML(ui.results_section_html(load_results(results_file)))
            with gr.Tab("Analisar", id="analisar"):
                gr.HTML(
                    ui.page_header_html(
                        "Analisar",
                        "Analise uma avaliação",
                        "Cole o título e o comentário do jeito que o cliente escreveu, "
                        "ou escolha um exemplo.",
                    )
                )
                with gr.Row(equal_height=False, elem_classes="lab-analisar"):
                    with gr.Column(scale=2, min_width=300, elem_classes="lab-form"):
                        title = gr.Textbox(
                            label="Título (opcional)",
                            max_length=MAX_TITLE_CHARS,
                            info=f"Até {MAX_TITLE_CHARS} caracteres.",
                        )
                        comment = gr.Textbox(
                            label="Comentário",
                            lines=5,
                            max_length=MAX_COMMENT_CHARS,
                            info=f"Até {ui.number(MAX_COMMENT_CHARS)} caracteres.",
                        )
                        button = gr.Button("Analisar avaliação", variant="primary")
                        gr.HTML('<p class="lab-rotulo lab-ex-rotulo">Exemplos inventados</p>')
                        with gr.Row(elem_classes="lab-exemplos"):
                            examples = [
                                gr.Button(name, variant="secondary", size="sm", scale=0)
                                for name, _, _ in EXAMPLES
                            ]
                    with gr.Column(scale=3, min_width=300):
                        reading = gr.HTML(ui.reading_empty_html(), elem_id="lab-leitura")
            with gr.Tab("Relatórios", id="relatorios"):
                gr.HTML(
                    ui.page_header_html(
                        "Relatórios de vendedor",
                        "Sua loja comparada com o marketplace",
                        "Escritos por um agente (Claude Haiku 5.5) que só consulta números "
                        "calculados em pandas, com conferência automática dos números, das "
                        "citações e do ranking. Estão prontos: abrir um relatório não chama API.",
                    )
                )
                first = next(iter(reports), None)
                choice = gr.Dropdown(
                    choices=list(reports), value=first, label="Vendedor", elem_classes="lab-escolha"
                )
                head, readings, body = report_parts(reports.get(first))
                report_head = gr.Markdown(head, elem_classes="lab-md lab-relatorio-cab")
                report_readings = gr.HTML(readings)
                report_body = gr.Markdown(body, elem_classes="lab-md lab-relatorio")
            with gr.Tab("Resultados", id="resultados"):
                gr.HTML(
                    ui.page_header_html(
                        "Resultados",
                        "Os três modelos no teste",
                        "TF-IDF com Regressão Logística, BERTimbau ajustado e LLM sem treino, "
                        "medidos com as mesmas funções de avaliação.",
                    )
                )
                gr.Markdown(results_markdown(results_file), elem_classes="lab-md lab-resultados")
        gr.HTML(
            ui.footer_html(
                GITHUB_URL,
                bert.repo_id(HF_USER, TASK_SENTIMENT),
                bert.repo_id(HF_USER, TASK_PROBLEMS),
            )
        )

        for trigger in (button.click, title.submit):
            trigger(run, [title, comment], reading, **hidden).then(
                None, js=SCROLL_TO_READING, **hidden
            )
        for example, (_, ex_title, ex_comment) in zip(examples, EXAMPLES, strict=True):
            example.click(fill(ex_title, ex_comment), None, [title, comment], **hidden).then(
                run, [title, comment], reading, **hidden
            ).then(None, js=SCROLL_TO_READING, **hidden)
        choice.change(show_report, choice, [report_head, report_readings, report_body], **hidden)
        to_analyse.click(go("analisar"), None, tabs, js=SCROLL_TOP, **hidden)
        to_report.click(go("relatorios"), None, tabs, js=SCROLL_TOP, **hidden)
        to_reports.click(go("relatorios"), None, tabs, js=SCROLL_TOP, **hidden)
        gr.api(classificar, api_name="classificar")
    # Uma previsão por vez: o PyTorch já usa todos os núcleos da CPU numa previsão.
    app.queue(max_size=20, default_concurrency_limit=1)
    return app


# ---------------------------------------------------------------------------
# Conferência dos modelos do Hub no teste
# ---------------------------------------------------------------------------


def check_models(
    classifier: Classifier | None = None,
    results_file: Path = RESULTS_FILE,
    output: Path = CHECK_FILE,
    frames: dict | None = None,
) -> dict:
    """Mede os modelos que a demo carrega no conjunto de teste e compara com o registrado.

    Não grava nada em ``resultados.json`` (o teste já foi medido); só confere que os
    modelos publicados, com o caminho de código da demo, dão o mesmo F1. Também grava as
    previsões dos exemplos da demo, a referência para conferir o que a tela mostra.
    """
    from reviews.baseline import problem_frame, sentiment_frame, split_frames

    classifier = classifier or Classifier()
    if frames is None:
        frames = {
            TASK_SENTIMENT: split_frames(sentiment_frame())[TEST],
            TASK_PROBLEMS: split_frames(problem_frame())[TEST],
        }
    recorded = load_results(results_file)
    report = {"verificado_em": datetime.now(UTC).isoformat(timespec="seconds"), "tarefas": {}}
    ok = True
    for task, logits_of in (
        (TASK_SENTIMENT, classifier.sentiment_logits),
        (TASK_PROBLEMS, classifier.problem_logits),
    ):
        frame = frames[task]
        predictions = bert.logits_to_predictions(task, logits_of(frame["texto"].tolist()))
        metrics = bert.score(task, bert.truth_of(task, frame), predictions)
        before = recorded.get(task, {}).get(TEST, {}).get(bert.MODEL_KEY, {})
        diff = metrics["f1_macro"] - before["f1_macro"] if before else None
        passed = diff is not None and abs(diff) <= CHECK_TOLERANCE
        ok = ok and passed
        report["tarefas"][task] = {
            "avaliacoes": int(len(frame)),
            "f1_macro_registrado": before.get("f1_macro"),
            "f1_macro_demo": metrics["f1_macro"],
            "diferenca": diff,
            "f1_por_classe_demo": {k: v["f1"] for k, v in metrics["por_classe"].items()},
            "confere": passed,
        }
        shown = "sem registro" if diff is None else f"diferença {diff:+.4f}"
        print(
            f"{task}: F1 macro {fmt_br(metrics['f1_macro'], 4)} na demo, "
            f"{fmt_br(before['f1_macro'], 4) if before else '—'} registrado ({shown}) · "
            f"{'confere' if passed else 'NÃO CONFERE'}"
        )
    texts = [prepare_text(title, comment) for _, title, comment in EXAMPLES]
    report["exemplos"] = [
        {"exemplo": name, **p.to_json()}
        for (name, _, _), p in zip(EXAMPLES, classifier.predict(texts), strict=True)
    ]
    report["confere"] = ok
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Conferência salva em {output}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo do analisador de avaliações.")
    sub = parser.add_subparsers(dest="comando")
    run = sub.add_parser("abrir", help="abre a demo no seu computador (padrão)")
    run.add_argument("--porta", type=int, default=7860)
    sub.add_parser("conferir", help="mede os modelos do Hub no teste e salva a conferência")
    args = parser.parse_args(argv)

    if args.comando == "conferir":
        report = check_models()
        if not report["confere"]:
            raise SystemExit("Os modelos do Hub não deram o F1 registrado; veja acima.")
    else:
        port = getattr(args, "porta", 7860)
        # Só no seu computador: nenhum link público é criado.
        launch(build_app(), server_name="127.0.0.1", server_port=port, share=False)


if __name__ == "__main__":
    main()
