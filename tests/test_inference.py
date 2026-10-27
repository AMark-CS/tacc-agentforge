import pytest

from inference.client import ModelClient


@pytest.mark.skip(reason="Task 4: not implemented")
def test_chat_returns_result():
    ModelClient("http://localhost:8000/v1").chat([{"role": "user", "content": "hi"}])


def test_fts5_probe():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "check_transport", pathlib.Path("deploy/tacc/check_transport.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert isinstance(mod.fts5_available(), bool)
