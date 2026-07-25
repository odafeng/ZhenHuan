import pytest
from core.reranker import _read_frame_bytes


def test_reads_local_file_when_no_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    f = tmp_path / "ep01_5.0s.jpg"
    f.write_bytes(b"local-bytes")
    assert _read_frame_bytes(str(f)) == b"local-bytes"


def test_returns_none_for_missing_local_file(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert _read_frame_bytes("data/processed/does_not_exist.jpg") is None


def test_fetches_from_base_url_when_set(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://zh.example.workers.dev")
    calls = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"remote-bytes"

    def fake_urlopen(url, timeout=None):
        calls["url"] = url
        return FakeResp()

    monkeypatch.setattr("core.reranker.urllib.request.urlopen", fake_urlopen)
    result = _read_frame_bytes("data/processed/ep01_5.0s.jpg")
    assert result == b"remote-bytes"
    assert calls["url"] == "https://zh.example.workers.dev/frames/ep01_5.0s.jpg"


def test_returns_none_when_remote_fetch_fails(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://zh.example.workers.dev")

    def boom(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("core.reranker.urllib.request.urlopen", boom)
    assert _read_frame_bytes("data/processed/ep01_5.0s.jpg") is None
