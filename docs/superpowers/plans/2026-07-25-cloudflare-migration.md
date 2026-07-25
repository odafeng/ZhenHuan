# Cloudflare 遷移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ZhenHuan 從已下線的 Render 部署搬到 Cloudflare（Worker + Containers + R2），沿用既有 Python + Gradio 程式碼。

**Architecture:** Worker 負責路由 —— `/frames/*` 以 R2 binding 供圖並快取，其餘轉發到跑 Gradio 的 Container。索引檔（`segments.pkl` + `humor_pool.pkl`，約 40 MB）打包進 image；60,874 張原圖（6.94 GB）放 R2。Stage 2 vision rerank 於查詢時從 R2 抓候選圖。

**Tech Stack:** Python 3.11、Gradio 6、Anthropic SDK、Cloudflare Workers（TypeScript）、`@cloudflare/containers`、R2、wrangler、rclone。

## Global Constraints

- Container runtime `requirements.txt` 只裝 3 個套件：`gradio`、`anthropic`、`python-dotenv`。不得加入 `faiss`、`torch`、`sentence-transformers`、`numpy`、`opencc`。
- `pyproject.toml` 維持完整依賴（測試用），與 `requirements.txt` 職責分離，不互相同步。
- reranker 的改動是外科式：只動 frame bytes 讀取，不重寫選圖邏輯。
- 不加任何新 Python 依賴 —— 從 R2 抓圖用 stdlib `urllib`。
- 圖片 URL、vision 讀圖共用單一環境變數 `PUBLIC_BASE_URL`；未設定時回退本機行為，既有 CLI／測試不受影響。
- frames 用本機原圖（6.94 GB、平均 120 KB/張），不用壓縮版。
- R2 bucket 維持 private，只透過 Worker 供圖，不啟用 r2.dev。
- Worker 轉發到 Container 一律用 `container.fetch()`，不可用 `containerFetch()`。
- 既有 25 個測試必須維持綠燈。
- 每完成一個 task 就 commit。commit message 用 Conventional Commits（英文）。

---

## File Structure

**新增：**
- `interfaces/web_cf.py` — Cloudflare 版 Gradio entry（由 `web_render.py` 改寫，移除下載邏輯）
- `worker/index.ts` — Worker 路由：`/frames/*` → R2 + Cache；其餘 → Container
- `wrangler.jsonc` — Worker + Container + R2 binding 設定
- `scripts/verify_r2_upload.py` — 上傳後驗證 R2 物件數與可讀性
- `tests/unit/test_web_cf.py` — `_resolve_frame` 單元測試
- `tests/unit/test_read_frame_bytes.py` — `_read_frame_bytes` 單元測試

**修改：**
- `core/reranker.py` — 新增 `_read_frame_bytes()`，`_final_select_with_vision` 改用之
- `requirements.txt` — 砍到 3 個套件
- `Dockerfile` — entrypoint 改 `web_cf`、port 對齊
- `.dockerignore` — 納入 `data/cache/index/*.pkl`

**產生（非程式碼）：**
- `data/cache/index/humor_pool.pkl` — 執行 `build_humor_pool.py` 產出

---

## Task 1: 建 humor pool（擋門步驟，需 API 費用）

**Files:**
- Run: `scripts/build_humor_pool.py`（既有，不改）
- Produce: `data/cache/index/humor_pool.pkl`

**Interfaces:**
- Consumes: `data/cache/index/segments.pkl`（既有）、`ANTHROPIC_API_KEY`
- Produces: `data/cache/index/humor_pool.pkl` — `list[Segment]`，約 3–5K 筆，供 `reranker.load_segments` 載入

> **Note:** 此 task 不寫程式碼，是一次性資料產生。成本約 $0.55。它擋住 Task 8（打包進 image）與部署，故排第一。

- [ ] **Step 1: 確認前置檔案與金鑰存在**

Run:
```bash
ls -la data/cache/index/segments.pkl && grep -q ANTHROPIC_API_KEY .env && echo "key present"
```
Expected: 顯示 `segments.pkl`（約 40 MB）且印出 `key present`。

- [ ] **Step 2: 執行建池腳本**

Run:
```bash
set -a && source .env && set +a && .venv/bin/python scripts/build_humor_pool.py
```
Expected: 逐 chunk 印出 picks，最後印出 `Total humor pool: N segments`（N 約 3000–5000）與 `Saved to data/cache/index/humor_pool.pkl`。

- [ ] **Step 3: 驗證產物合理**

Run:
```bash
.venv/bin/python -c "
import pickle
pool = pickle.load(open('data/cache/index/humor_pool.pkl','rb'))
print(f'pool size: {len(pool)}')
assert 1000 <= len(pool) <= 15000, f'pool size {len(pool)} out of expected range'
with_frame = sum(1 for s in pool if s.frame_path)
print(f'with frame_path: {with_frame}/{len(pool)}')
assert with_frame > len(pool) * 0.8, 'too many pool segments lack frame_path'
print('OK')
"
```
Expected: 印出 pool size 與 frame 覆蓋率，最後 `OK`。

- [ ] **Step 4: 抽樣人工檢視品質**

Run:
```bash
.venv/bin/python -c "
import pickle, random
pool = pickle.load(open('data/cache/index/humor_pool.pkl','rb'))
random.seed(1)
for s in random.sample(pool, 15):
    print(f'{s.episode_id} @ {s.current_start:.0f}s  {s.text_current}')
"
```
Expected: 15 句台詞，多數應具「可當梗圖回應」的語感（強烈語氣／雙關／經典語錄）。若明顯多為敘事廢句則品質不佳 —— 停下來回報，勿繼續。

> **不 commit：** `humor_pool.pkl` 未進 git（`data/cache/` 在 `.gitignore`）。它由 Task 8 的 `.dockerignore` 例外納入 image。此 task 只產生檔案。

---

## Task 2: 精簡 container requirements

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: 只含 3 個套件的 `requirements.txt`，供 Task 9 的 `docker build` 使用

- [ ] **Step 1: 改寫 requirements.txt**

把整個檔案內容換成：
```
gradio>=6.0
anthropic>=0.40
python-dotenv>=1.0
```

- [ ] **Step 2: 確認被移除的套件確實非 runtime 依賴**

Run:
```bash
grep -rEn "^(import|from) (faiss|torch|numpy|sentence_transformers|opencc|huggingface_hub|PIL|openai)" interfaces/web_cf.py core/reranker.py schemas.py config.py 2>/dev/null; echo "exit: $?"
```
Expected: 無任何 match（`web_cf.py` 此時尚未建立，grep 對缺檔會略過）。實際擋關驗證在 Task 9 的 `docker build` + 起容器。

> **Note:** 這裡不跑 pytest —— pytest 用 `pyproject.toml` 那份完整依賴，驗證不到「只裝 3 個」是否夠。真正的驗證是 Task 9。

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: trim container requirements to runtime-only deps

LLM-only path imports only gradio, anthropic and python-dotenv.
faiss/torch/sentence-transformers were dead weight left over from
the pre-LLM architecture. pyproject.toml keeps the full set for
tests; the two lists serve different targets and are intentionally
not synced."
```

---

## Task 3: reranker 支援從 R2 讀 frame bytes

**Files:**
- Modify: `core/reranker.py`
- Test: `tests/unit/test_read_frame_bytes.py`

**Interfaces:**
- Produces: `core.reranker._read_frame_bytes(frame_path: str) -> bytes | None`
  - `PUBLIC_BASE_URL` 有設 → 從 `{PUBLIC_BASE_URL}/frames/{basename}` 以 urllib 抓，失敗回 None
  - 未設 → 讀本機磁碟，不存在回 None
- Consumes: 環境變數 `PUBLIC_BASE_URL`（選用）

- [ ] **Step 1: 寫失敗測試**

建立 `tests/unit/test_read_frame_bytes.py`：
```python
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
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/unit/test_read_frame_bytes.py -v`
Expected: FAIL —— `ImportError: cannot import name '_read_frame_bytes'`

- [ ] **Step 3: 實作 `_read_frame_bytes`**

在 `core/reranker.py`，把最上方的 import 區塊加上 `os` 與 `urllib.request`：
```python
import base64
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
```

在 `_parse_indices` 函式之後（class 定義之前）新增：
```python
def _read_frame_bytes(frame_path: str) -> bytes | None:
    """Read a frame's raw bytes for vision rerank.

    When PUBLIC_BASE_URL is set (Cloudflare container), frames live in R2
    and are fetched over HTTP via the Worker. Otherwise read from local
    disk (CLI, local dev, tests). Returns None on any failure so the
    caller can degrade that candidate to text-only.
    """
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if base:
        name = Path(frame_path).name
        try:
            with urllib.request.urlopen(f"{base}/frames/{name}", timeout=10) as resp:
                return resp.read()
        except Exception:
            return None
    p = Path(frame_path)
    if p.exists():
        return p.read_bytes()
    return None
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/unit/test_read_frame_bytes.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add core/reranker.py tests/unit/test_read_frame_bytes.py
git commit -m "feat(reranker): read vision frames from R2 when PUBLIC_BASE_URL set

Stage 2 vision rerank needs the candidate image bytes. After moving
frames to R2 they are no longer on the container's local disk, so add
_read_frame_bytes: fetch over HTTP when PUBLIC_BASE_URL is set, fall
back to local disk otherwise. stdlib urllib, no new dependency."
```

---

## Task 4: `_final_select_with_vision` 改用 `_read_frame_bytes`

**Files:**
- Modify: `core/reranker.py:256-300`（`_final_select_with_vision` 的圖片讀取段）

**Interfaces:**
- Consumes: `_read_frame_bytes`（Task 3）
- 行為不變：能取得 bytes 的候選帶圖送 Claude，取不到的降級為 text-only

- [ ] **Step 1: 替換磁碟讀取邏輯**

在 `core/reranker.py` 的 `_final_select_with_vision`，把這段：
```python
        valid_winners = []
        for i, s in enumerate(winners, 1):
            frame_path = s.frame_path
            if frame_path and Path(frame_path).exists():
                try:
                    img_data = base64.b64encode(Path(frame_path).read_bytes()).decode()
                    content_blocks.append({
                        "type": "text",
                        "text": f"\n--- 候選 {len(valid_winners) + 1} ---",
                    })
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data,
                        },
                    })
                    content_blocks.append({
                        "type": "text",
                        "text": f"台詞：「{s.text_current}」（{s.episode_id}）",
                    })
                    valid_winners.append(s)
                except Exception:
                    continue
            else:
```

換成：
```python
        valid_winners = []
        for i, s in enumerate(winners, 1):
            img_bytes = _read_frame_bytes(s.frame_path) if s.frame_path else None
            if img_bytes:
                img_data = base64.b64encode(img_bytes).decode()
                content_blocks.append({
                    "type": "text",
                    "text": f"\n--- 候選 {len(valid_winners) + 1} ---",
                })
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data,
                    },
                })
                content_blocks.append({
                    "type": "text",
                    "text": f"台詞：「{s.text_current}」（{s.episode_id}）",
                })
                valid_winners.append(s)
            else:
```

（下方 `else:` 區塊 —— 無圖候選的 text-only 處理 —— 維持不動。）

- [ ] **Step 2: 跑既有測試確認沒破壞**

Run: `.venv/bin/python -m pytest -q`
Expected: 全綠（Task 3 的 4 個 + 既有 25 個 = 29 passed）

- [ ] **Step 3: Commit**

```bash
git add core/reranker.py
git commit -m "refactor(reranker): use _read_frame_bytes in vision rerank

Vision candidate images now come through _read_frame_bytes, so Stage 2
works whether frames are on local disk (dev) or in R2 (container). The
no-image text-only fallback is unchanged."
```

---

## Task 5: `web_cf.py` — Cloudflare 版 Gradio entry

**Files:**
- Create: `interfaces/web_cf.py`
- Test: `tests/unit/test_web_cf.py`

**Interfaces:**
- Produces: `interfaces.web_cf._resolve_frame(frame_path: str | None) -> str | None`
  - 非空 frame_path → `{PUBLIC_BASE_URL}/frames/{basename}`；`PUBLIC_BASE_URL` 未設 → `/frames/{basename}`
  - 空／None → None
- Produces: module-level Gradio `app`，`__main__` 時啟動背景初始化並 launch
- Consumes: `core.reranker.LLMReranker`、環境變數 `PUBLIC_BASE_URL`、`PORT`

- [ ] **Step 1: 確保 venv 有 web_cf 的 import 依賴**

`web_cf.py` 在 module 載入時 `import gradio` 與 `from dotenv import load_dotenv`，但兩者都不在測試 venv（`.[dev]` 不含 gradio，`python-dotenv` 不在 `pyproject.toml`）。測試需要能 import `web_cf`，故先裝進 venv：
```bash
VIRTUAL_ENV=.venv uv pip install "gradio>=6.0" "python-dotenv>=1.0"
```
Expected: 安裝成功。這是測試支援依賴，與 container runtime 的 `requirements.txt` 各自獨立（職責分離）。

- [ ] **Step 2: 寫失敗測試**

建立 `tests/unit/test_web_cf.py`：
```python
import pytest
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
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `.venv/bin/python -m pytest tests/unit/test_web_cf.py -v`
Expected: FAIL —— import 失敗或 `_resolve_frame` 不存在

> import `interfaces.web_cf` 不得觸發模型載入或起執行緒 —— 見 Step 4，初始化只在 `__main__` 內啟動。

- [ ] **Step 4: 建立 `web_cf.py`**

建立 `interfaces/web_cf.py`，內容為：
```python
"""Gradio web UI for Cloudflare Containers deployment.

LLM-only mode. Index files (segments.pkl + humor_pool.pkl) are baked
into the container image; frames are served from R2 via the Worker.
Nothing is downloaded at runtime.
"""

import os
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

# --- Paths (baked into image) ---
INDEX_DIR = Path("data/cache/index")

# --- Global state ---
_ready = False
_status = "啟動中..."
_reranker = None


def _init():
    """Load the LLM reranker from baked-in index. No downloads."""
    global _ready, _status, _reranker
    try:
        _status = "載入中..."
        print("Loading LLM reranker...", flush=True)
        from core.reranker import LLMReranker
        _reranker = LLMReranker()
        _reranker.load_segments(INDEX_DIR.parent)
        _ready = True
        _status = "就緒！"
        print("Service ready!", flush=True)
    except Exception as e:
        _status = f"啟動失敗：{e}"
        import traceback
        traceback.print_exc()
        print(f"INIT FAILED: {e}", flush=True)


def _resolve_frame(frame_path: str | None) -> str | None:
    if not frame_path:
        return None
    name = Path(frame_path).name
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    return f"{base}/frames/{name}" if base else f"/frames/{name}"


def search(query: str, top_k: int = 5):
    if not query.strip():
        return []
    results_raw = _reranker.select(query.strip(), top_k=int(top_k))
    results = []
    for r in results_raw:
        seg = r.segment
        frame = _resolve_frame(seg.frame_path)
        if frame:
            caption = f"「{seg.text_current}」\n{seg.episode_id} | {seg.current_start:.0f}s"
            results.append((frame, caption))
    return results


def check_ready():
    if _ready:
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True, value=f"## 系統啟動中...\n\n{_status}\n\n請稍候。"), gr.update(visible=False)


# --- Gradio UI ---
MOBILE_CSS = """
@media (max-width: 768px) {
    .row { flex-direction: column !important; }
    .gallery .grid-wrap { grid-template-columns: repeat(2, 1fr) !important; }
}
"""

with gr.Blocks(title="甄嬛傳迷因產生器") as app:
    loading_box = gr.Markdown("## 系統啟動中...\n\n請稍候...", visible=True)

    with gr.Column(visible=False) as main_ui:
        gr.Markdown("# 甄嬛傳迷因產生器\n輸入一句話，找最好笑的甄嬛截圖回應")

        with gr.Row():
            query_input = gr.Textbox(
                label="你想說什麼？",
                placeholder="例：今天加班到好晚、發現他很大、你是在酸我嗎",
                scale=4,
            )
            top_k_input = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="結果數量", scale=1,
            )

        search_btn = gr.Button("找迷因！", variant="primary")

        gallery = gr.Gallery(
            label="搜尋結果",
            columns=3,
            height="auto",
            object_fit="contain",
        )

        search_btn.click(fn=search, inputs=[query_input, top_k_input], outputs=gallery)
        query_input.submit(fn=search, inputs=[query_input, top_k_input], outputs=gallery)

    timer = gr.Timer(2)
    timer.tick(fn=check_ready, outputs=[loading_box, main_ui])


if __name__ == "__main__":
    threading.Thread(target=_init, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(),
               css=MOBILE_CSS)
```

- [ ] **Step 5: 跑測試確認通過**

Run: `.venv/bin/python -m pytest tests/unit/test_web_cf.py -v`
Expected: 4 passed

- [ ] **Step 6: 跑完整測試確認沒破壞既有**

Run: `.venv/bin/python -m pytest -q`
Expected: 33 passed（25 既有 + 4 read_frame_bytes + 4 web_cf）

- [ ] **Step 7: Commit**

```bash
git add interfaces/web_cf.py tests/unit/test_web_cf.py
git commit -m "feat(web): add Cloudflare container entry web_cf

Loads reranker from baked-in index, no runtime downloads. _resolve_frame
returns a full R2/Worker image URL via PUBLIC_BASE_URL for the gallery.
Background init starts only under __main__ so the module imports cleanly
in tests."
```

---

## Task 6: Dockerfile + .dockerignore

**Files:**
- Modify: `Dockerfile`
- Modify: `.dockerignore`

**Interfaces:**
- Produces: image entrypoint `python -m interfaces.web_cf`，port 8080
- image 內含 `data/cache/index/segments.pkl` + `humor_pool.pkl`，不含 `data/processed`

- [ ] **Step 1: 改寫 Dockerfile**

把 `Dockerfile` 換成：
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# ffmpeg/build tools no longer needed — LLM-only path has no faiss/opencv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gradio listens on 8080 (matches wrangler container defaultPort)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "interfaces.web_cf"]
```

- [ ] **Step 2: 改寫 .dockerignore**

把 `.dockerignore` 換成：
```
data/videos/
data/processed/
data/subtitles/
data/cache/*
!data/cache/index
data/cache/index/*
!data/cache/index/segments.pkl
!data/cache/index/humor_pool.pkl
.git/
__pycache__/
*.pyc
.venv/
.env
docs/
tests/
```

- [ ] **Step 3: 驗證 build context 只納入該納入的**

Run:
```bash
docker build -t zhenhuan-cf . 2>&1 | tail -5 && \
docker run --rm zhenhuan-cf sh -c "ls -la data/cache/index/ && du -sh data/ 2>/dev/null"
```
Expected: `data/cache/index/` 內有 `segments.pkl` 與 `humor_pool.pkl`；`data/` 總大小為數十 MB 等級（**不含** 6.94 GB frames）。若 `data/` 是 GB 級，代表 `data/processed` 被誤納入 —— 停下修正。

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: container entry web_cf, bake index, exclude frames

Entry is now interfaces.web_cf on port 8080. .dockerignore is rewritten
to bake data/cache/index/*.pkl into the image while still excluding the
6.94 GB data/processed frames (served from R2), plus docs/ and tests/."
```

---

## Task 7: 本地驗證精簡後的 container 能起且能查詢（requirements 擋關驗證）

**Files:**
- 無（驗證 task，用 Task 6 的 image）

**Interfaces:**
- Consumes: Task 6 的 `zhenhuan-cf` image、`.env` 的 `ANTHROPIC_API_KEY`

> **Note:** 這是 Global Constraint「只裝 3 個套件仍能完成一次查詢」的真正驗證點。pytest 驗證不到（它用 pyproject 的完整依賴）。

- [ ] **Step 1: 起 container（不設 PUBLIC_BASE_URL，vision 走本機無圖 → text-only 降級，可接受）**

Run:
```bash
set -a && source .env && set +a && \
docker run -d --name zh-test -p 8080:8080 -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" zhenhuan-cf && \
sleep 5 && docker logs zh-test 2>&1 | tail -20
```
Expected: log 出現 `Loading LLM reranker...`、`Loaded humor pool: N candidates`、`Service ready!`，且**無** `ModuleNotFoundError`。若出現缺套件錯誤 → 該套件其實是 runtime 依賴，加回 `requirements.txt` 並回 Task 6 重 build。

- [ ] **Step 2: 打一次查詢驗證端到端**

Run:
```bash
curl -sS -X POST http://localhost:8080/gradio_api/call/search \
  -H 'Content-Type: application/json' \
  -d '{"data":["今天加班到好晚",5]}' | head -c 300; echo
```
Expected: 回傳含 event id 的 JSON（Gradio 的 async endpoint 兩段式回應），非連線錯誤。主要目的是確認 app 有起、endpoint 有回應。若無法確定 endpoint 格式，改以瀏覽器手動驗證：開 `http://localhost:8080`，等 loading 消失後查一句，確認有結果（圖片因未設 PUBLIC_BASE_URL 而無法載入是預期的，看得到 caption 即算通過）。

- [ ] **Step 3: 清理**

Run: `docker rm -f zh-test`
Expected: 容器移除。

> 此 task 不 commit（純驗證）。若全數通過，requirements 精簡確認安全。

---

## Task 8: 建 R2 bucket 並上傳 frames

**Files:**
- Create: `scripts/verify_r2_upload.py`

**Interfaces:**
- Produces: R2 bucket `zhenhuan-frames`，內含 60,874 個 `.jpg`（key 即檔名，如 `ep01_5.0s.jpg`）
- Consumes: 本機 `data/processed/*.jpg`

- [ ] **Step 1: 建 bucket**

Run:
```bash
npx wrangler r2 bucket create zhenhuan-frames
```
Expected: 建立成功訊息。若已存在，wrangler 會提示，可略過。

- [ ] **Step 2: 設定 rclone 指向 R2**

先取得 R2 API token（dashboard → R2 → Manage R2 API Tokens，需 Object Read & Write），然後：
```bash
rclone config create zhr2 s3 provider=Cloudflare \
  access_key_id=<R2_ACCESS_KEY_ID> \
  secret_access_key=<R2_SECRET_ACCESS_KEY> \
  endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com \
  acl=private
```
Expected: 建立名為 `zhr2` 的 remote，無錯誤。

- [ ] **Step 3: 上傳 frames（排除 .DS_Store）**

Run:
```bash
rclone copy data/processed zhr2:zhenhuan-frames \
  --include "*.jpg" --transfers 32 --checkers 32 --progress
```
Expected: 進度跑完，傳輸約 60,874 個檔、約 6.94 GB。

- [ ] **Step 4: 寫驗證腳本**

建立 `scripts/verify_r2_upload.py`：
```python
"""Verify R2 frame upload: compare object count and spot-check readability.

Usage:
    python scripts/verify_r2_upload.py
Reads local count from data/processed/*.jpg and R2 count via rclone.
"""

import subprocess
import sys
from pathlib import Path

REMOTE = "zhr2:zhenhuan-frames"


def local_count() -> int:
    return len(list(Path("data/processed").glob("*.jpg")))


def remote_count() -> int:
    out = subprocess.run(
        ["rclone", "size", REMOTE, "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    import json
    return json.loads(out)["count"]


def main() -> None:
    local = local_count()
    remote = remote_count()
    print(f"local jpgs:  {local}")
    print(f"remote objs: {remote}")
    if local != remote:
        print(f"MISMATCH: {local - remote} files missing on remote")
        sys.exit(1)
    print("Count OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 執行驗證**

Run: `.venv/bin/python scripts/verify_r2_upload.py`
Expected: `local jpgs` 與 `remote objs` 相等，印出 `Count OK`。不相等則重跑 Step 3 的 `rclone copy`（可續傳）。

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_r2_upload.py
git commit -m "feat(scripts): add R2 upload verification

Compares local data/processed jpg count against the R2 bucket object
count via rclone. Upload itself is done with rclone copy (parallel,
resumable), not a hand-rolled uploader."
```

---

## Task 9: Worker 路由 + wrangler 設定

**Files:**
- Create: `worker/index.ts`
- Create: `wrangler.jsonc`
- Create: `package.json`（wrangler 專案最小需求）

**Interfaces:**
- Consumes: R2 bucket `zhenhuan-frames`、`zhenhuan-cf` image（Dockerfile）
- Produces: 部署後的 Worker，`/frames/*` → R2、其餘 → Container

- [ ] **Step 1: 建立 package.json**

建立 `package.json`：
```json
{
  "name": "zhenhuan",
  "private": true,
  "devDependencies": {
    "wrangler": "^4.0.0",
    "@cloudflare/containers": "^0.2.0"
  }
}
```

- [ ] **Step 2: 安裝依賴**

Run: `npm install`
Expected: 產生 `node_modules` 與 `package-lock.json`，無錯誤。

- [ ] **Step 3: 建立 wrangler.jsonc**

建立 `wrangler.jsonc`（`<ACCOUNT_SUBDOMAIN>` 用 Step 8 取得的實際值回填；先留佔位）：
```jsonc
{
  "name": "zhenhuan",
  "main": "worker/index.ts",
  "compatibility_date": "2026-07-06",
  "containers": [
    {
      "class_name": "GradioContainer",
      "image": "./Dockerfile",
      "instance_type": "basic",
      "max_instances": 3
    }
  ],
  "durable_objects": {
    "bindings": [
      { "name": "GRADIO_CONTAINER", "class_name": "GradioContainer" }
    ]
  },
  "migrations": [
    { "tag": "v1", "new_sqlite_classes": ["GradioContainer"] }
  ],
  "r2_buckets": [
    { "binding": "FRAMES", "bucket_name": "zhenhuan-frames" }
  ]
}
```

- [ ] **Step 4: 建立 worker/index.ts**

建立 `worker/index.ts`：
```typescript
import { Container } from "@cloudflare/containers";

export class GradioContainer extends Container {
  defaultPort = 8080;
  sleepAfter = "30m";
  enableInternet = true; // needed for Anthropic API + R2 frame fetch in Stage 2
}

interface Env {
  GRADIO_CONTAINER: DurableObjectNamespace<GradioContainer>;
  FRAMES: R2Bucket;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    // /frames/* -> R2, with edge caching
    if (url.pathname.startsWith("/frames/")) {
      const key = url.pathname.slice("/frames/".length);
      if (!key) return new Response("Not found", { status: 404 });

      const cache = caches.default;
      const cached = await cache.match(request);
      if (cached) return cached;

      const object = await env.FRAMES.get(key);
      if (object === null) return new Response("Not found", { status: 404 });

      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("etag", object.httpEtag);
      headers.set("Content-Type", "image/jpeg");
      headers.set("Cache-Control", "public, max-age=31536000, immutable");

      const response = new Response(object.body, { headers });
      ctx.waitUntil(cache.put(request, response.clone()));
      return response;
    }

    // everything else -> the Gradio container (single shared instance)
    const container = env.GRADIO_CONTAINER.getByName("singleton");
    await container.startAndWaitForPorts();
    return container.fetch(request); // fetch(), NOT containerFetch() — SSE/WS
  },
};
```

- [ ] **Step 5: 本地跑 wrangler dev 驗證路由**

Run:
```bash
npx wrangler dev --remote 2>&1 | tee /tmp/wrangler-dev.log &
sleep 25
curl -sS -o /dev/null -w "frames route: %{http_code}\n" "http://localhost:8787/frames/ep01_5.0s.jpg"
```
Expected: `frames route: 200`（`--remote` 讓 R2 讀真實 bucket）。若 404，確認 Task 8 已上傳且 key 為 `ep01_5.0s.jpg`（無 `data/processed/` 前綴）。

- [ ] **Step 6: 停掉 dev**

Run: `pkill -f "wrangler dev" || true`

- [ ] **Step 7: 首次部署**

Run: `npx wrangler deploy`
Expected: 部署成功，輸出 Worker 網址 `https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev`。記下此網址。

- [ ] **Step 8: 回填 PUBLIC_BASE_URL 並重部署**

把 Step 7 得到的網址加入 `wrangler.jsonc` 的 container 設定（讓容器與 gallery 都能組出圖片 URL）：
```jsonc
  "containers": [
    {
      "class_name": "GradioContainer",
      "image": "./Dockerfile",
      "instance_type": "basic",
      "max_instances": 3,
      "envVars": {
        "PUBLIC_BASE_URL": "https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev"
      }
    }
  ],
```
然後：`npx wrangler deploy`
Expected: 重新部署成功。

- [ ] **Step 9: Commit**

```bash
git add worker/index.ts wrangler.jsonc package.json package-lock.json
git commit -m "feat(worker): route frames to R2, everything else to container

Worker serves /frames/* from the R2 binding with immutable edge
caching, and proxies all other paths to a single shared Gradio
container via fetch() (not containerFetch, so Gradio SSE survives).
PUBLIC_BASE_URL is injected into the container so both the gallery
and Stage 2 vision resolve image URLs."
```

---

## Task 10: 線上 smoke 驗證

**Files:**
- 無（驗證 task）

**Interfaces:**
- Consumes: 已部署的 Worker 網址

- [ ] **Step 1: 確認圖片路由線上可讀**

Run:
```bash
curl -sS -o /dev/null -w "%{http_code} %{content_type}\n" \
  "https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev/frames/ep01_5.0s.jpg"
```
Expected: `200 image/jpeg`

- [ ] **Step 2: 確認第二次請求命中快取**

Run:
```bash
curl -sS -o /dev/null -w "cf-cache: %{header_json}\n" \
  "https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev/frames/ep01_5.0s.jpg" 2>/dev/null | grep -io "cf-cache-status[^,]*" || \
curl -sSI "https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev/frames/ep01_5.0s.jpg" | grep -i cf-cache-status
```
Expected: 第二次出現 `cf-cache-status: HIT`（首次可能為 MISS）。

- [ ] **Step 3: 瀏覽器手動驗證完整流程**

用瀏覽器開 `https://zhenhuan.<ACCOUNT_SUBDOMAIN>.workers.dev`，等 loading 畫面消失（首次冷啟動 2–3s），輸入「今天加班到好晚」按「找迷因！」。
Expected:
- 搜尋結果有回來（代表 Gradio SSE 未被 Worker 阻斷 —— 這是最關鍵的一項）
- Gallery 顯示截圖（代表 `PUBLIC_BASE_URL` 正確、R2 供圖正常）
- 每張圖下方有台詞 + 集數時間

- [ ] **Step 4: 更新 TODO.md 標記部署完成**

在 `TODO.md` 的「Phase 8: 待辦」把「確認 Render 服務是否還在線」該行改為記錄已遷移至 Cloudflare，並勾選部署項。具體：
```markdown
## Phase 8: 已遷移至 Cloudflare（2026-07）
- [x] 從 Render 遷移到 Worker + Containers + R2（見 docs/adr/0001）
- [x] humor_pool.pkl 建立並打包進 image（每次查詢成本從 $0.51 降下）
- [x] frames 上傳 R2、Worker 代理供圖 + 邊緣快取
- [x] container requirements 精簡至 3 個套件
```

- [ ] **Step 5: Commit**

```bash
git add TODO.md
git commit -m "docs: mark Cloudflare migration deployed in TODO"
```

---

## 部署後備忘

- **監看冷啟動記憶體**：`basic` 是 1 GiB。若 `docker logs` / Cloudflare 觀測到 OOM，把 `wrangler.jsonc` 的 `instance_type` 改 `standard-1`（4 GiB）重部署。
- **humor pool 更新**：日後若重建 pool，重跑 Task 1 後需重新 `wrangler deploy`（pool 打包在 image 內）。
- **畫質退路**：若行動網路載入太慢，於 Worker 的 `/frames/*` 分支接 Cloudflare Image Transformations 動態縮圖，不動已上傳的原圖。
