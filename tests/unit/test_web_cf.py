from interfaces.web_cf import _resolve_frame


def test_builds_full_url_when_base_set(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://zh.example.workers.dev")
    assert (
        _resolve_frame("data/processed/ep01_5.0s.jpg")
        == "https://zh.example.workers.dev/frames/ep01_5.0s.jpg"
    )


def test_strips_trailing_slash_on_base(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://zh.example.workers.dev/")
    assert (
        _resolve_frame("data/processed/ep01_5.0s.jpg")
        == "https://zh.example.workers.dev/frames/ep01_5.0s.jpg"
    )


def test_relative_path_when_base_unset(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert _resolve_frame("data/processed/ep01_5.0s.jpg") == "/frames/ep01_5.0s.jpg"


def test_none_for_empty(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert _resolve_frame("") is None
    assert _resolve_frame(None) is None
