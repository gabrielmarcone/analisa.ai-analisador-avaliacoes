import pytest

from reviews import download as dl


def test_missing_files(tmp_path):
    assert dl.missing_files(tmp_path) == list(dl.REQUIRED_FILES)
    for name in dl.REQUIRED_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert dl.missing_files(tmp_path) == []


def test_arquivo_vazio_conta_como_faltando(tmp_path):
    for name in dl.REQUIRED_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / dl.REQUIRED_FILES[0]).write_text("", encoding="utf-8")
    assert dl.missing_files(tmp_path) == [dl.REQUIRED_FILES[0]]


def test_comando_e_lista_sem_shell(tmp_path):
    cmd = dl.build_command("kaggle", tmp_path)
    assert cmd == ["kaggle", "datasets", "download", dl.DATASET, "-p", str(tmp_path), "--unzip"]
    assert dl.build_command("kaggle", tmp_path, force=True)[-1] == "--force"


def test_nao_baixa_se_ja_existe(tmp_path, monkeypatch):
    for name in dl.REQUIRED_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")

    def fail(*args, **kwargs):
        raise AssertionError("não deveria chamar o Kaggle")

    monkeypatch.setattr(dl.subprocess, "run", fail)
    dl.download(tmp_path)


def test_erro_claro_sem_kaggle(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="Kaggle CLI não encontrado"):
        dl.download(tmp_path)


def test_erro_claro_se_download_falha(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.shutil, "which", lambda _: "kaggle")

    class Result:
        returncode = 1

    monkeypatch.setattr(dl.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(SystemExit, match="kaggle auth login"):
        dl.download(tmp_path)
