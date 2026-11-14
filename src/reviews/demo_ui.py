"""Identidade visual da demo (direção "Laboratório") e o HTML de cada parte da tela.

Tudo aqui só monta texto: as funções recebem previsões e números já calculados e
devolvem HTML. Os números da tela inicial vêm de ``reports/resultados.json`` e das
auditorias dos relatórios, nunca digitados à mão. Todo texto que vem de fora (o que a
pessoa digita, as citações dos relatórios) passa por ``html.escape``.

Cores, fontes e componentes estão em ``demo.css``; o desenho de referência está no
Claude Design ("analisa.ai · identidade Laboratório").
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from reviews.annotation import CATEGORIES, NO_COMPLAINT
from reviews.baseline import TASK_PROBLEMS, TASK_SENTIMENT, THRESHOLD
from reviews.evaluate import fmt_br
from reviews.labels import NEGATIVE
from reviews.splits import TEST

CSS_FILE = Path(__file__).with_name("demo.css")
FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800"
    "&family=Atkinson+Hyperlegible+Mono:wght@400;500"
    "&family=Atkinson+Hyperlegible+Next:wght@400;500;700&display=swap"
)
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    f'<link rel="stylesheet" href="{FONTS_URL}">'
)
FONT_TEXT = ["Atkinson Hyperlegible Next", "Segoe UI", "Arial", "sans-serif"]
FONT_MONO = ["Atkinson Hyperlegible Mono", "ui-monospace", "Consolas", "monospace"]

SENTIMENT_LABELS = {
    "negativa": ("Negativa", "nota 1 ou 2"),
    "positiva": ("Positiva", "nota 4 ou 5"),
}

# O que cada tipo quer dizer, em linguagem de vendedor (resumo do docs/guia_anotacao.md).
DESCRIPTIONS = {
    "atraso": "O cliente cita prazo, data ou demora.",
    "nao_recebido": "Nada do pedido chegou, ou consta como entregue e não chegou.",
    "incompleto": "Chegou só parte dos itens, ou faltou peça, acessório ou quantidade.",
    "produto_diferente": "Outro produto, ou cor, tamanho, modelo ou marca diferente do anúncio.",
    "defeito_qualidade": "Veio quebrado, danificado, não funciona ou tem qualidade ruim.",
    "atendimento": (
        "A loja não responde ou cancelou, ou o cliente pediu troca, devolução ou reembolso."
    ),
    "outro": "Frete caro, retirada nos Correios, nota fiscal, embalagem ou reclamação genérica.",
    NO_COMPLAINT: "Nenhum problema descrito. Nunca vem junto com outro tipo.",
}

# Nomes das categorias de produto da Olist como aparecem para o vendedor.
PRODUCT_CATEGORIES = {
    "cama mesa banho": "cama, mesa e banho",
    "relogios presentes": "relógios e presentes",
    "beleza saude": "beleza e saúde",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def pct(p: float) -> str:
    """Probabilidade para a tela: inteiro, sem fingir certeza (>99% em vez de 100%)."""
    if p >= 0.995:
        return "&gt;99%"
    if p < 0.005:
        return "&lt;1%"
    return f"{math.floor(p * 100 + 0.5)}%"


def number(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def percent(value: float, decimals: int = 1) -> str:
    return f"{fmt_br(value, decimals)}%"


def product_category(name: str) -> str:
    return PRODUCT_CATEGORIES.get(name, name)


# ---------------------------------------------------------------------------
# Peças da leitura
# ---------------------------------------------------------------------------


def _bar_width(p: float) -> str:
    return f"{max(0.6, round(p * 100, 1))}%"


def measure_row(name: str, p: float, strong: bool) -> str:
    weak = "" if strong else " fraca"
    return (
        f'<div class="lab-medida{weak}"><span class="lab-medida-nome">{esc(name)}</span>'
        f'<span class="lab-trilho" aria-hidden="true"><span class="lab-barra{weak}" '
        f'style="width: {_bar_width(p)}"></span><span class="lab-limiar"></span></span>'
        f'<span class="lab-valor">{pct(p)}</span></div>'
    )


def ruler(positive_probability: float) -> str:
    left = f"{round(positive_probability * 100, 1)}%"
    ticks = "".join(
        f'<span class="lab-tick{" meio" if x == 50 else ""}" style="left: {x}%"></span>'
        for x in (0, 25, 50, 75, 100)
    )
    return (
        f'<div class="lab-regua" aria-hidden="true">{ticks}'
        f'<span class="lab-marcador" style="left: {left}"></span></div>'
        '<div class="lab-regua-pontas" aria-hidden="true"><span>Negativa</span>'
        "<span>Positiva</span></div>"
    )


def _sentiment_block(prediction) -> str:
    name, grade = SENTIMENT_LABELS[prediction.sentiment]
    p = prediction.sentiment_probabilities[prediction.sentiment]
    positive = prediction.sentiment_probabilities.get("positiva", 1.0 - p)
    return (
        '<div class="lab-bloco">'
        '<div class="lab-linha-cab"><span class="lab-rotulo">Sentimento</span>'
        f'<span><strong>{name}</strong> <span class="lab-sub">({grade})</span> '
        f'<span class="lab-leitura-num">{pct(p)}</span></span></div>'
        f"{ruler(positive)}</div>"
    )


def _sorted_problems(prediction) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    probs = prediction.problem_probabilities
    chosen = sorted(((k, probs[k]) for k in prediction.problems), key=lambda kv: -kv[1])
    others = sorted(
        ((k, v) for k, v in probs.items() if k not in prediction.problems), key=lambda kv: -kv[1]
    )
    return chosen, others


def low_confidence(prediction) -> bool:
    """Nenhum tipo passou do limite: a regra da avaliação ficou com o mais provável."""
    probs = prediction.problem_probabilities
    return bool(probs) and max(probs.values()) < THRESHOLD


def verdict(prediction) -> tuple[str, str]:
    if prediction.sentiment != NEGATIVE or prediction.problems is None:
        return (
            "Avaliação positiva",
            "Tipos de problema não se aplicam: o modelo foi treinado só com avaliações negativas.",
        )
    if low_confidence(prediction):
        return "Avaliação negativa, sem tipo claro", "Nenhum tipo de problema passou de 50%."
    n = len(prediction.problems)
    if n == 1:
        return "Avaliação negativa com 1 tipo de problema", "Ele passou de 50% de probabilidade."
    return (
        f"Avaliação negativa com {n} tipos de problema",
        "Cada um passou de 50% de probabilidade.",
    )


HOW_IT_READS = (
    '<details class="lab-detalhes"><summary>Como a leitura é feita</summary><ul>'
    "<li>Um BERTimbau diz se a avaliação é negativa (nota 1 ou 2) ou positiva (nota 4 ou 5).</li>"
    "<li>Se for negativa, outro BERTimbau dá a probabilidade de cada tipo. Entra cada tipo com "
    "pelo menos 50%; se nenhum chega lá, entra o mais provável; “sem reclamação” nunca vem "
    "junto com outro tipo.</li>"
    "<li>Os modelos erram: use a leitura como triagem, não como veredito. Os números do teste "
    "estão em Resultados.</li></ul></details>"
)


def _panel(body: str) -> str:
    return (
        '<section class="lab-painel" aria-label="Leitura">'
        '<div class="lab-painel-cab"><span class="lab-rotulo">Leitura</span>'
        '<span class="lab-rotulo">BERTimbau · base Olist</span></div>'
        f"{body}</section>"
    )


def reading_html(prediction) -> str:
    """O resultado de uma avaliação: veredito, sentimento e tipos de problema."""
    title, subtitle = verdict(prediction)
    parts = [
        f'<div class="lab-bloco lab-veredito" role="status"><h2 class="lab-tit">{esc(title)}</h2>'
        f"<p>{esc(subtitle)}</p></div>",
        _sentiment_block(prediction),
    ]
    if prediction.problems is not None:
        chosen, others = _sorted_problems(prediction)
        low = low_confidence(prediction)
        heading = "Tipo mais provável" if low else "Tipos de problema detectados"
        rows = "".join(measure_row(CATEGORIES[k], p, True) for k, p in chosen)
        notice = (
            '<div class="lab-aviso"><span class="lab-rotulo">Confiança baixa</span>'
            "<p>Nenhum tipo passou de 50%. Pela regra da avaliação, aparece o mais provável. "
            "Leia a avaliação antes de agir.</p></div>"
            if low
            else ""
        )
        rest = "".join(measure_row(CATEGORIES[k], p, False) for k, p in others)
        parts.append(
            '<div class="lab-bloco">'
            f'<div class="lab-linha-cab"><span class="lab-rotulo">{heading}</span>'
            '<span class="lab-rotulo">┆ limite de 50%</span></div>'
            f"{rows}{notice}"
            f'<details class="lab-detalhes"><summary>Ver os outros {len(others)} tipos</summary>'
            f"{rest}</details></div>"
        )
    parts.append(
        f'<div class="lab-bloco lab-rodape"><p class="lab-analisado">Texto analisado: '
        f"“{esc(prediction.text)}”</p>{HOW_IT_READS}</div>"
    )
    return _panel("".join(parts))


def reading_empty_html() -> str:
    return _panel(
        '<div class="lab-bloco lab-veredito"><h2 class="lab-tit">A leitura aparece aqui</h2>'
        "<p>Escreva uma avaliação e clique em Analisar avaliação, ou escolha um exemplo.</p>"
        f'</div><div class="lab-bloco lab-rodape">{HOW_IT_READS}</div>'
    )


def reading_error_html(message: str) -> str:
    return _panel(
        '<div class="lab-bloco lab-veredito" role="alert">'
        '<h2 class="lab-tit">Não deu para analisar</h2>'
        f"<p>{esc(message)}</p></div>"
    )


def specimen_html(prediction) -> str:
    """A leitura de exemplo da tela inicial: saída real do modelo, sem ajuste."""
    rows = ""
    if prediction.problems is not None:
        chosen, _ = _sorted_problems(prediction)
        rows = (
            '<div class="lab-linha-cab lab-amostra-tipos"><span class="lab-rotulo">'
            'Tipos de problema</span><span class="lab-rotulo">┆ limite de 50%</span></div>'
            + "".join(measure_row(CATEGORIES[k], p, True) for k, p in chosen)
        )
    name, _ = SENTIMENT_LABELS[prediction.sentiment]
    p = prediction.sentiment_probabilities[prediction.sentiment]
    return (
        '<figure class="lab-amostra">'
        '<div class="lab-painel-cab"><span class="lab-rotulo">Leitura de exemplo</span>'
        '<span class="lab-rotulo">Saída real do modelo</span></div>'
        f'<div class="lab-papel"><blockquote class="lab-tit">“{esc(prediction.text)}”</blockquote>'
        "</div>"
        '<div class="lab-amostra-corpo"><div class="lab-linha-cab">'
        '<span class="lab-rotulo">Sentimento</span>'
        f'<span class="lab-leitura-num">{name} · {pct(p)}</span></div>{rows}</div>'
        "<figcaption>Avaliação inventada. Probabilidades do modelo publicado, sem ajuste."
        "</figcaption></figure>"
    )


# ---------------------------------------------------------------------------
# Cabeçalho, tela inicial e rodapé
# ---------------------------------------------------------------------------


def header_html() -> str:
    return (
        '<header class="lab-cab">'
        '<span class="lab-marca" aria-label="analisa.ai">analisa<span>.ai</span></span>'
        '<span class="lab-rotulo">Demo · base pública da Olist · 2016–2018</span></header>'
    )


def page_header_html(label: str, title: str, intro: str) -> str:
    return (
        f'<div class="lab-pagina-cab"><p class="lab-rotulo">{esc(label)}</p>'
        f'<h1 class="lab-tit">{esc(title)}</h1><p class="lab-intro">{esc(intro)}</p></div>'
    )


def hero_html() -> str:
    return (
        '<div class="lab-heroi"><p class="lab-rotulo">Para vendedores de e-commerce</p>'
        '<h1 class="lab-tit">Descubra do que seus clientes estão reclamando.</h1>'
        '<p class="lab-intro">Cole uma avaliação e o analisa.ai diz se ela é negativa e aponta '
        "o problema: atraso, pedido que não chegou, item faltando, defeito, produto diferente "
        "do anúncio ou atendimento.</p></div>"
    )


def privacy_html() -> str:
    return '<p class="lab-nota">Os textos que você digita não são guardados.</p>'


def _section_head(number_label: str, title: str, intro: str = "") -> str:
    side = f'<p class="lab-intro">{intro}</p>' if intro else ""
    return (
        f'<div class="lab-secao-cab"><div><p class="lab-rotulo">{number_label}</p>'
        f'<h2 class="lab-tit">{title}</h2></div>{side}</div>'
    )


def types_html() -> str:
    cells = "".join(
        f'<div class="lab-celula"><span class="lab-num">{i:02d}</span>'
        f"<h3>{esc(CATEGORIES[k])}</h3><p>{esc(DESCRIPTIONS[k])}</p></div>"
        for i, k in enumerate(CATEGORIES, start=1)
    )
    return (
        '<section class="lab-secao">'
        + _section_head(
            "01 · O que ele lê",
            "Oito tipos de problema",
            "Definidos num guia de anotação e marcados à mão em 1.500 avaliações negativas. "
            "Uma avaliação pode ter mais de um tipo.",
        )
        + f'<div class="lab-celulas">{cells}</div></section>'
    )


def steps_html() -> str:
    steps = (
        ("Você cola a avaliação", "Título e comentário, do jeito que o cliente escreveu."),
        (
            "Dois modelos leem o texto",
            "Um diz se a avaliação é negativa. Se for, o outro dá a probabilidade de cada tipo "
            "de problema.",
        ),
        (
            "Você vê a leitura",
            "Entra cada tipo com pelo menos 50%. Se nenhum chega lá, aparece o mais provável, "
            "com aviso de confiança baixa.",
        ),
    )
    items = "".join(
        f'<li><span class="lab-num lab-passo">{i}</span><h3>{t}</h3><p>{d}</p></li>'
        for i, (t, d) in enumerate(steps, start=1)
    )
    return (
        '<section class="lab-secao">'
        + _section_head("02 · Como funciona", "Três passos, uma leitura")
        + f'<ol class="lab-passos">{items}</ol></section>'
    )


def report_summary(audit_file: Path) -> dict | None:
    """Os números que o agente consultou, tirados da auditoria do relatório."""
    audit_file = Path(audit_file)
    if not audit_file.is_file():
        return None
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    out = {f"saida_{c['ferramenta']}": c["saida"] for c in reversed(audit["ferramentas"])}
    summary = out.get("saida_resumo_vendedor", {})
    market = out.get("saida_comparar_com_marketplace", {})
    delays = out.get("saida_entregas", {}).get("atraso_nos_pedidos", {})
    problems = out.get("saida_problemas_relatados", {}).get("categorias", [])
    top_category = (summary.get("categorias_principais") or [{}])[0].get("categoria", "")
    top_problem = min(problems, key=lambda c: c["posicao"]) if problems else None
    checks = (
        "numeros_sem_fonte",
        "citacoes_sem_fonte",
        "citacoes_mal_atribuidas",
        "rankings_errados",
        "recomendacoes_sem_apoio",
        "tabela_de_problemas",
        "ferramentas_nao_consultadas",
    )
    return {
        "codigo": str(audit.get("vendedor", ""))[:8],
        "categoria": product_category(top_category),
        "avaliacoes": summary.get("avaliacoes"),
        "primeiro_mes": summary.get("primeiro_mes"),
        "ultimo_mes": summary.get("ultimo_mes"),
        "nota": market.get("nota_media_vendedor"),
        "nota_marketplace": market.get("nota_media_marketplace"),
        "negativas": market.get("pct_negativas_vendedor"),
        "negativas_marketplace": market.get("pct_negativas_marketplace"),
        "atraso": delays.get("vendedor", {}).get("pct_com_atraso"),
        "atraso_marketplace": delays.get("marketplace", {}).get("pct_com_atraso"),
        "problema": top_problem["nome"] if top_problem else None,
        "problema_pct": top_problem["pct_das_negativas_com_comentario"] if top_problem else None,
        "numeros_conferidos": audit.get("numeros_conferidos"),
        "confere": all(not audit.get(k) for k in checks),
    }


def _month(value: str | None) -> str:
    months = "jan fev mar abr mai jun jul ago set out nov dez".split()
    if not value or len(value) != 7:
        return esc(value or "")
    year, month = value.split("-")
    return f"{months[int(month) - 1]}/{year}"


def report_readings_html(s: dict | None) -> str:
    if not s:
        return ""

    def cell(label: str, value: str, note: str) -> str:
        return (
            f'<div class="lab-medicao"><span class="lab-rotulo">{label}</span>'
            f'<span class="lab-medicao-num">{value}</span><span class="lab-sub">{note}</span></div>'
        )

    cells = [
        cell(
            "Avaliações",
            number(s["avaliacoes"]),
            f"{_month(s['primeiro_mes'])} a {_month(s['ultimo_mes'])}",
        ),
        cell("Nota média", fmt_br(s["nota"], 2), f"marketplace {fmt_br(s['nota_marketplace'], 2)}"),
        cell(
            "Negativas",
            percent(s["negativas"]),
            f"marketplace {percent(s['negativas_marketplace'])}",
        ),
        cell(
            "Pedidos com atraso",
            percent(s["atraso"]),
            f"marketplace {percent(s['atraso_marketplace'])}",
        ),
    ]
    if s["confere"]:
        check = (
            f"Conferência automática: os {esc(s['numeros_conferidos'])} números do texto "
            "existem nos cálculos, a tabela bate linha a linha e as citações são literais. "
            "Tudo confere."
        )
    else:
        check = "A conferência automática apontou divergências neste relatório; veja o final."
    return f'<div class="lab-medicoes">{"".join(cells)}</div><p class="lab-nota">{check}</p>'


def split_report(markdown: str) -> tuple[str, str]:
    """Separa o cabeçalho do relatório (título e data) do corpo (a partir do Resumo)."""
    marker = "\n## "
    i = markdown.find(marker)
    if i < 0:
        return markdown, ""
    return markdown[:i].strip(), markdown[i + 1 :].strip()


def reports_section_html(summaries: list[dict]) -> str:
    summaries = [s for s in summaries if s]
    if not summaries:
        return ""
    rows = "".join(
        f'<tr><td class="lab-num">{esc(s["codigo"])}</td><td>{esc(s["categoria"].capitalize())}'
        f'</td><td class="n">{number(s["avaliacoes"])}</td><td class="n">'
        f"{percent(s['negativas'])}</td><td>{esc(s['problema'] or '—')} "
        f'<span class="lab-sub lab-num">{percent(s["problema_pct"])}</span></td></tr>'
        for s in summaries
    )
    market = summaries[0]["negativas_marketplace"]
    return (
        '<section class="lab-secao lab-continua">'
        + _section_head(
            "03 · Relatórios de vendedor",
            "Sua loja comparada com o marketplace",
            f"Para {len(summaries)} vendedores reais da base pública, um agente escreveu um "
            "relatório com nota, negativas, atrasos e os problemas mais citados. Cada número "
            "do texto é conferido automaticamente.",
        )
        + '<div class="lab-tabela"><table><caption>No marketplace, '
        f"{percent(market)} das avaliações são negativas</caption><thead><tr>"
        '<th scope="col">Vendedor</th><th scope="col">Categoria principal</th>'
        '<th scope="col" class="n">Avaliações</th><th scope="col" class="n">Negativas</th>'
        '<th scope="col">Problema mais citado</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def results_section_html(results: dict) -> str:
    """Quanto dá para confiar, com os números de resultados.json (teste, medido uma vez)."""
    sentiment = results.get(TASK_SENTIMENT, {}).get(TEST, {}).get("bertimbau")
    problems = results.get(TASK_PROBLEMS, {}).get(TEST, {}).get("bertimbau")
    llm = results.get(TASK_PROBLEMS, {}).get(TEST, {}).get("llm_zero_shot")
    if not sentiment or not problems:
        return ""
    per_class = {k: v for k, v in problems["por_classe"].items() if k != NO_COMPLAINT}
    best = max(per_class, key=lambda k: per_class[k]["f1"])
    worst = min(per_class, key=lambda k: per_class[k]["f1"])
    no_complaint = problems["por_classe"].get(NO_COMPLAINT)
    tail = (
        f" Em “sem reclamação” ({no_complaint['suporte']} casos no teste), "
        f"o F1 foi {fmt_br(no_complaint['f1'], 2)}."
        if no_complaint
        else ""
    )
    cells = [
        (
            percent(sentiment["acuracia"] * 100),
            "de acerto no sentimento",
            f"{number(sentiment['n'])} avaliações de teste. F1 macro de "
            f"{fmt_br(sentiment['f1_macro'], 3)}.",
            "",
        ),
        (
            fmt_br(problems["f1_macro"], 3),
            "F1 macro nos tipos de problema",
            f"{number(problems['n'])} avaliações marcadas à mão, numa escala de 0 a 1. Vai de "
            f"{fmt_br(per_class[best]['f1'], 2)} em {CATEGORIES[best].lower()} a "
            f"{fmt_br(per_class[worst]['f1'], 2)} em {CATEGORIES[worst].lower()}.{tail}",
            "",
        ),
    ]
    if llm:
        cells.append(
            (
                fmt_br(llm["f1_macro"], 3),
                "F1 do Claude Haiku 5.5 nos tipos",
                "O melhor modelo nessa tarefa, usado nos relatórios. A demo usa o BERTimbau, "
                "que não tem custo por avaliação.",
                " lab-secundario",
            )
        )
    body = "".join(
        f'<div class="lab-medicao-grande"><span class="lab-medicao-num{extra}">{value}</span>'
        f'<strong>{label}</strong><span class="lab-sub">{note}</span></div>'
        for value, label, note, extra in cells
    )
    return (
        '<section class="lab-secao">'
        + _section_head(
            "04 · Resultados",
            "Quanto dá para confiar",
            "Medido uma única vez, em avaliações que os modelos nunca viram. Os modelos erram: "
            "use a leitura como triagem, não como veredito.",
        )
        + f'<div class="lab-grandes">{body}</div></section>'
    )


def footer_html(github_url: str, sentiment_repo: str, problems_repo: str) -> str:
    return (
        '<footer class="lab-rodape-site"><div>'
        '<span class="lab-marca" aria-hidden="true">analisa<span>.ai</span></span>'
        "<p>Projeto de portfólio de Gabriel Marcone. Os textos digitados aqui não são "
        "guardados.</p></div><ul>"
        f'<li><a href="{esc(github_url)}" target="_blank" rel="noopener">'
        "Código, metodologia e decisões no GitHub</a></li>"
        f'<li><a href="https://huggingface.co/{esc(sentiment_repo)}" target="_blank" '
        'rel="noopener">Modelo de sentimento</a></li>'
        f'<li><a href="https://huggingface.co/{esc(problems_repo)}" target="_blank" '
        'rel="noopener">Modelo de tipos de problema</a></li></ul>'
        '<p class="lab-nota">Dados da Olist sob CC BY-NC-SA 4.0. Os modelos e os relatórios '
        "seguem a mesma licença, para uso não comercial.</p></footer>"
    )


# ---------------------------------------------------------------------------
# Tema do Gradio
# ---------------------------------------------------------------------------

LIGHT = {
    "fundo": "#F3F5F6",
    "sup": "#FFFFFF",
    "tinta": "#0F1419",
    "tinta2": "#44505C",
    "tinta3": "#5F6B77",
    "linha": "#D3D9DE",
    "dest": "#2D46D6",
    "sobre_dest": "#FFFFFF",
}
DARK = {
    "fundo": "#0C1014",
    "sup": "#141A20",
    "tinta": "#E6EAEE",
    "tinta2": "#A9B4BE",
    "tinta3": "#8C98A4",
    "linha": "#2A343E",
    "dest": "#8EA0FF",
    "sobre_dest": "#0C1014",
}


def theme():
    """Tema do Gradio com as cores e fontes da identidade (claro e escuro)."""
    import gradio as gr

    values = {}
    for suffix, c in (("", LIGHT), ("_dark", DARK)):
        values.update(
            {
                f"body_background_fill{suffix}": c["fundo"],
                f"body_text_color{suffix}": c["tinta"],
                f"body_text_color_subdued{suffix}": c["tinta2"],
                f"background_fill_primary{suffix}": c["sup"],
                f"background_fill_secondary{suffix}": c["fundo"],
                f"border_color_primary{suffix}": c["linha"],
                f"border_color_accent{suffix}": c["dest"],
                f"color_accent{suffix}": c["dest"],
                f"color_accent_soft{suffix}": c["fundo"],
                f"link_text_color{suffix}": c["dest"],
                f"link_text_color_hover{suffix}": c["tinta"],
                f"link_text_color_visited{suffix}": c["dest"],
                f"block_background_fill{suffix}": c["sup"],
                f"block_border_color{suffix}": c["linha"],
                f"block_label_text_color{suffix}": c["tinta"],
                f"block_title_text_color{suffix}": c["tinta"],
                f"block_info_text_color{suffix}": c["tinta2"],
                f"input_background_fill{suffix}": c["sup"],
                f"input_background_fill_focus{suffix}": c["sup"],
                f"input_border_color{suffix}": c["tinta3"],
                f"input_border_color_focus{suffix}": c["dest"],
                f"input_placeholder_color{suffix}": c["tinta3"],
                f"button_primary_background_fill{suffix}": c["dest"],
                f"button_primary_background_fill_hover{suffix}": c["tinta"],
                f"button_primary_text_color{suffix}": c["sobre_dest"],
                f"button_primary_text_color_hover{suffix}": c["fundo"],
                f"button_primary_border_color{suffix}": c["dest"],
                f"button_primary_border_color_hover{suffix}": c["tinta"],
                f"button_secondary_background_fill{suffix}": c["sup"],
                f"button_secondary_background_fill_hover{suffix}": c["sup"],
                f"button_secondary_text_color{suffix}": c["tinta"],
                f"button_secondary_text_color_hover{suffix}": c["tinta"],
                f"button_secondary_border_color{suffix}": c["linha"],
                f"button_secondary_border_color_hover{suffix}": c["tinta3"],
                f"loader_color{suffix}": c["dest"],
                f"panel_background_fill{suffix}": c["sup"],
                f"panel_border_color{suffix}": c["linha"],
                f"table_border_color{suffix}": c["linha"],
                f"table_even_background_fill{suffix}": c["sup"],
                f"table_odd_background_fill{suffix}": c["sup"],
                f"code_background_fill{suffix}": c["fundo"],
            }
        )
    values.update(
        {
            "block_border_width": "1px",
            "input_border_width": "1px",
            "input_border_width_dark": "1px",
            "button_border_width": "1px",
            "button_border_width_dark": "1px",
            "block_radius": "2px",
            "block_shadow": "none",
            "block_shadow_dark": "none",
            "input_radius": "2px",
            "input_shadow": "none",
            "input_shadow_dark": "none",
            "input_shadow_focus": "none",
            "input_shadow_focus_dark": "none",
            "button_large_radius": "2px",
            "button_small_radius": "2px",
            "button_medium_radius": "2px",
            "button_primary_shadow": "none",
            "button_primary_shadow_hover": "none",
            "button_secondary_shadow": "none",
            "button_secondary_shadow_hover": "none",
            "container_radius": "2px",
            "embed_radius": "2px",
            "layout_gap": "24px",
        }
    )
    base = gr.themes.Base(font=FONT_TEXT, font_mono=FONT_MONO)
    accepted = {k: v for k, v in values.items() if hasattr(base, k)}
    return base.set(**accepted)


def css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")
