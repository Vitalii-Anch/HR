"""(a) App import/startup smoke test. Must pass with no ANTHROPIC_API_KEY set."""
import os


def test_import_app_module():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    import app.main  # noqa: F401

    assert app.main.app is not None


def test_config_settings_load_without_key():
    from app.config import Settings

    s = Settings()
    assert s.anthropic_api_key is None or isinstance(s.anthropic_api_key, str)
    assert s.embedding_model_name == "all-MiniLM-L6-v2"


def test_fastapi_health_and_chat_smoke():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["mcp_connected"] is True
        assert body["index_loaded"] is True

        r = client.post("/chat", json={"message": "What is the PTO carryover limit?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert isinstance(body["citations"], list)
        assert isinstance(body["tool_trace"], list)
