"""App local de anotação (Gradio).

Uso:
    python -m reviews.annotate_app              # 1ª rodada
    python -m reviews.annotate_app --rodada 2   # reanotação para medir concordância

O app só escuta em 127.0.0.1 (seu computador) e nunca cria link público.
Cada avaliação é salva no CSV assim que você clica em "Salvar".
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from reviews.annotation import (
    CATEGORIES,
    LABELS,
    build_agreement_queue,
    clean_note,
    load_labels,
    load_queue,
    save_labels,
    save_queue,
    selection_of,
    upsert_label,
)
from reviews.paths import (
    AGREEMENT_LABELS_FILE,
    AGREEMENT_QUEUE_FILE,
    LABELS_FILE,
    PROCESSED_REVIEWS_FILE,
    QUEUE_FILE,
)

STAGE_NAMES = {
    "piloto": "Piloto (treino)",
    "teste": "Teste",
    "validacao": "Validação",
    "treino": "Treino",
}


@dataclass
class AnnotationSession:
    """Estado da anotação, separado da interface para poder ser testado."""

    queue: pd.DataFrame
    texts: dict[str, str]
    labels: pd.DataFrame
    labels_path: Path
    stage_label: str | None = None
    _done: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        missing = set(self.queue["review_id"]) - set(self.texts)
        if missing:
            raise ValueError(
                f"{len(missing)} avaliações da fila não estão na base processada. "
                "Rode `python -m reviews.prepare` de novo."
            )
        self._done = set(self.labels["review_id"])

    @property
    def size(self) -> int:
        return len(self.queue)

    def refresh(self) -> None:
        """Relê os rótulos do disco.

        Assim, correções feitas no arquivo com o app aberto não são apagadas
        pelo próximo "Salvar", e a tela sempre mostra o que está gravado.
        """
        self.labels = load_labels(self.labels_path)
        self._done = set(self.labels["review_id"])

    def first_pending(self, start: int = 0) -> int:
        """Índice da próxima avaliação sem rótulo a partir de `start` (ou o fim da fila)."""
        ids = self.queue["review_id"].tolist()
        for idx in range(max(start, 0), len(ids)):
            if ids[idx] not in self._done:
                return idx
        for idx in range(0, min(start, len(ids))):
            if ids[idx] not in self._done:
                return idx
        return len(ids)

    def view(self, idx: int) -> dict:
        """Tudo que a tela precisa mostrar para a avaliação na posição `idx`."""
        self.refresh()
        done = len(self._done & set(self.queue["review_id"]))
        header = f"**{done} de {self.size} anotadas**"
        if idx >= self.size:
            return {
                "idx": self.size,
                "header": header + " · fila concluída 🎉",
                "text": "Todas as avaliações desta fila foram anotadas.",
                "selected": [],
                "revisar": False,
                "note": "",
            }

        item = self.queue.iloc[idx]
        review_id = item["review_id"]
        stage = self.stage_label or STAGE_NAMES.get(item["etapa"], item["etapa"])
        header += f" · item {idx + 1} · etapa: {stage}"
        selected, revisar, note = [], False, ""
        existing = self.labels[self.labels["review_id"] == review_id]
        if not existing.empty:
            row = existing.iloc[0]
            selected, revisar, note = selection_of(row), bool(row["revisar"]), row["observacao"]
            header += " · ✔ já anotada"
        return {
            "idx": idx,
            "header": header,
            "text": self.texts[review_id],
            "selected": selected,
            "revisar": revisar,
            "note": str(note),
        }

    def save(self, idx: int, selected: list[str], revisar: bool, note: str) -> int:
        """Salva os rótulos de `idx` em disco e devolve o índice da próxima pendente."""
        if not 0 <= idx < self.size:
            raise ValueError("Não há avaliação nesta posição.")
        review_id = self.queue.iloc[idx]["review_id"]
        self.refresh()
        updated = upsert_label(self.labels, review_id, selected, revisar, clean_note(note))
        save_labels(updated, self.labels_path)
        self.labels = updated
        self._done.add(review_id)
        return self.first_pending(idx + 1)


def load_texts(review_ids: pd.Series, path: Path = PROCESSED_REVIEWS_FILE) -> dict[str, str]:
    if not Path(path).is_file():
        raise SystemExit("Base processada não encontrada. Rode `python -m reviews.prepare`.")
    reviews = pd.read_parquet(path, columns=["review_id", "texto"])
    reviews = reviews[reviews["review_id"].isin(set(review_ids))]
    return dict(zip(reviews["review_id"], reviews["texto"], strict=True))


def build_session(rodada: int) -> AnnotationSession:
    queue = load_queue(QUEUE_FILE)
    if rodada == 1:
        return AnnotationSession(
            queue, load_texts(queue["review_id"]), load_labels(LABELS_FILE), LABELS_FILE
        )

    if not AGREEMENT_QUEUE_FILE.is_file():
        save_queue(build_agreement_queue(queue, load_labels(LABELS_FILE)), AGREEMENT_QUEUE_FILE)
        print(f"Fila de concordância criada em {AGREEMENT_QUEUE_FILE}.")
    agreement_queue = load_queue(AGREEMENT_QUEUE_FILE)
    return AnnotationSession(
        agreement_queue,
        load_texts(agreement_queue["review_id"]),
        load_labels(AGREEMENT_LABELS_FILE),
        AGREEMENT_LABELS_FILE,
        stage_label="Concordância (2ª rodada)",
    )


def build_app(session: AnnotationSession):
    import gradio as gr

    choices = [(CATEGORIES[label], label) for label in LABELS]

    def show(idx: int, status: str = ""):
        v = session.view(idx)
        return v["idx"], v["header"], v["text"], v["selected"], v["revisar"], v["note"], status

    def on_save(idx, selected, revisar, note):
        try:
            next_idx = session.save(int(idx), selected or [], bool(revisar), note or "")
        except (ValueError, PermissionError) as err:
            v = session.view(int(idx))
            return idx, v["header"], v["text"], selected, revisar, note, f"⚠️ {err}"
        return show(next_idx, "Salvo.")

    def on_previous(idx):
        return show(max(int(idx) - 1, 0))

    def on_next_pending(idx):
        return show(session.first_pending(int(idx) + 1))

    def on_goto(idx, item):
        if item is None or not 1 <= int(item) <= session.size:
            return (*show(int(idx))[:-1], f"⚠️ Digite um número de 1 a {session.size}.")
        return show(int(item) - 1)

    with gr.Blocks(title="Anotação de avaliações", analytics_enabled=False) as app:
        gr.Markdown("## Anotação de avaliações negativas")
        idx_state = gr.State(0)
        header = gr.Markdown()
        text = gr.Textbox(label="Avaliação", lines=6, max_lines=20, interactive=False)
        checks = gr.CheckboxGroup(choices=choices, label="Problemas relatados (marque todos)")
        with gr.Row():
            revisar = gr.Checkbox(label="Revisar depois (estou em dúvida)")
            note = gr.Textbox(
                label="Observação (opcional, sem dados pessoais)",
                max_length=300,
                lines=1,
            )
        with gr.Row():
            prev_btn = gr.Button("← Anterior")
            next_btn = gr.Button("Próxima pendente →")
            save_btn = gr.Button("Salvar e seguir", variant="primary")
        with gr.Row():
            goto_item = gr.Number(label="Ir para o item nº", precision=0, minimum=1, scale=3)
            goto_btn = gr.Button("Ir", scale=1)
        status = gr.Markdown()
        with gr.Accordion("Resumo do guia", open=False):
            gr.Markdown(GUIDE_SUMMARY)

        outputs = [idx_state, header, text, checks, revisar, note, status]
        save_btn.click(on_save, [idx_state, checks, revisar, note], outputs)
        prev_btn.click(on_previous, [idx_state], outputs)
        next_btn.click(on_next_pending, [idx_state], outputs)
        goto_btn.click(on_goto, [idx_state, goto_item], outputs)
        app.load(lambda: show(session.first_pending(0)), None, outputs)
    return app


GUIDE_SUMMARY = """
- **Anote o que o texto diz.** Não use nota nem dados de entrega.
- **Atraso** só com prazo, data, demora ou duração. "Até hoje não chegou" é só **Não recebido**.
- Chegou **parte** do pedido? É só **Pedido incompleto** (nota fiscal sozinha não conta).
- Não é o anunciado (ou é falsificado)? **Produto errado**. É o certo mas ruim? **Defeito**.
- **Atendimento**: reclamação do processo ou pedido de cancelamento, troca, devolução ou reembolso.
- Falta de estoque após a compra: **Atendimento** + **Não recebido**.
- Retirar nos Correios: já pegou → **Outro**; ainda não pegou → **Não recebido** + **Outro**.
- Culpar os Correios pelo atraso não é **Outro**.
- Reclamação genérica ("Péssimo", "Ruim") → **Outro**.
- "Consta como entregue, mas não chegou" → **Não recebido** (não é Outro).
- Embalagem ruim → **Outro** (junto com Defeito se o produto chegou danificado).
- Nota fiscal só como prova não conta. "Não funciona" é **Defeito**; incompatível é **Outro**.
- **Outro** é o último recurso: confira antes se outra categoria cobre o problema.
- **Sem reclamação** nunca vai junto com outro rótulo.
- Em dúvida? Marque **Revisar depois** e siga. O guia completo está em `docs/guia_anotacao.md`.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Abre o app de anotação no navegador.")
    parser.add_argument("--rodada", type=int, choices=(1, 2), default=1)
    parser.add_argument("--porta", type=int, default=7860)
    args = parser.parse_args(argv)

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    session = build_session(args.rodada)
    app = build_app(session)
    app.launch(server_name="127.0.0.1", server_port=args.porta, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
