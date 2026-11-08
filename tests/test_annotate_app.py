import pandas as pd
import pytest

from reviews.annotate_app import AnnotationSession
from reviews.annotation import empty_labels, load_labels


@pytest.fixture
def session(tmp_path) -> AnnotationSession:
    queue = pd.DataFrame(
        {
            "ordem": [1, 2, 3],
            "review_id": ["a", "b", "c"],
            "split": ["treino", "teste", "teste"],
            "etapa": ["piloto", "teste", "teste"],
        }
    )
    texts = {"a": "não chegou", "b": "veio quebrado", "c": "<b>html</b> **markdown**"}
    return AnnotationSession(queue, texts, empty_labels(), tmp_path / "rotulos.csv")


def test_fluxo_salva_em_disco_e_avanca(session):
    assert session.first_pending() == 0
    nxt = session.save(0, ["nao_recebido"], False, "")
    assert nxt == 1
    assert load_labels(session.labels_path)["review_id"].tolist() == ["a"]

    view = session.view(0)
    assert view["selected"] == ["nao_recebido"] and "já anotada" in view["header"]
    assert "1 de 3" in session.view(1)["header"]


def test_erro_de_validacao_nao_grava(session):
    with pytest.raises(ValueError):
        session.save(0, [], False, "")
    assert not session.labels_path.exists()


def test_texto_e_mostrado_sem_alteracao(session):
    # O texto vai para um Textbox (não renderiza HTML/Markdown), então é passado cru.
    assert session.view(2)["text"] == "<b>html</b> **markdown**"


def test_pula_anotadas_e_conclui(session):
    session.save(1, ["defeito_qualidade"], False, "")
    assert session.first_pending(1) == 2
    session.save(2, ["outro"], True, "dúvida")
    assert session.first_pending(0) == 0  # volta ao início e acha a pendente
    session.save(0, ["atraso"], False, "")
    assert session.first_pending(0) == 3
    assert "concluída" in session.view(3)["header"]


def test_retoma_de_onde_parou(session):
    session.save(0, ["atraso"], False, "")
    resumed = AnnotationSession(
        session.queue, session.texts, load_labels(session.labels_path), session.labels_path
    )
    assert resumed.first_pending() == 1


def test_texto_faltando_gera_erro_claro(tmp_path):
    queue = pd.DataFrame({"ordem": [1], "review_id": ["x"], "split": ["teste"], "etapa": ["teste"]})
    with pytest.raises(ValueError, match="prepare"):
        AnnotationSession(queue, {}, empty_labels(), tmp_path / "r.csv")


def test_interface_monta_sem_erros(session):
    from reviews.annotate_app import build_app

    app = build_app(session)
    assert app is not None


def test_correcao_externa_nao_e_apagada(session):
    from reviews.annotation import save_labels, upsert_label

    session.save(0, ["atraso"], False, "")
    # Alguém corrige o item "a" direto no arquivo com o app aberto...
    fixed = upsert_label(load_labels(session.labels_path), "a", ["nao_recebido"])
    save_labels(fixed, session.labels_path)
    # ...e o app mostra a correção e não a apaga ao salvar outro item.
    assert session.view(0)["selected"] == ["nao_recebido"]
    session.save(1, ["defeito_qualidade"], False, "")
    saved = load_labels(session.labels_path).set_index("review_id")
    assert saved.loc["a", "nao_recebido"] == 1 and saved.loc["a", "atraso"] == 0
    assert saved.loc["b", "defeito_qualidade"] == 1
