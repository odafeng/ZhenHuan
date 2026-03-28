"""Gradio web UI for Render deployment.

LLM-only mode (no embedding model needed) — saves ~1.3GB RAM.
Downloads index + frames from HF Dataset to persistent disk on first run.
"""

import os
import tarfile
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", "/data"))
INDEX_DIR = PERSIST_DIR / "cache" / "index"
FRAMES_DIR = PERSIST_DIR / "processed"
HF_REPO = "odafeng/zhenhuan-data"

# --- Global state ---
_ready = False
_status = "啟動中..."
_reranker = None
_segments = None


def _download_and_init():
    """Download data + init LLM reranker (no embedding model needed)."""
    global _ready, _status, _reranker, _segments

    try:
        from huggingface_hub import hf_hub_download

        # 1. Download index (only need segments.pkl for LLM mode)
        seg_path = INDEX_DIR / "segments.pkl"
        print(f"Checking index at {seg_path}, exists={seg_path.exists()}", flush=True)
        if not seg_path.exists():
            _status = "下載索引中..."
            print("Downloading index...", flush=True)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(HF_REPO, "index.tar.gz", repo_type="dataset")
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(INDEX_DIR)
            print("Index ready", flush=True)
        else:
            print("Index already on disk, skipping download", flush=True)

        # 2. Download frames
        # Check if frames need upgrade: version file tracks quality
        version_file = FRAMES_DIR / ".version"
        current_version = version_file.read_text().strip() if version_file.exists() else ""
        target_version = "640px_q55_v2"
        frame_count = len(list(FRAMES_DIR.glob("*.jpg"))) if FRAMES_DIR.exists() else 0
        needs_download = frame_count < 1000 or current_version != target_version
        print(f"Checking frames at {FRAMES_DIR}, count={frame_count}, version={current_version or 'none'}", flush=True)
        if needs_download:
            _status = "下載截圖中..."
            print("Downloading frames_small.tar.gz...", flush=True)
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(HF_REPO, "frames_small.tar.gz", repo_type="dataset")
            print("  Download complete, extracting...", flush=True)
            _status = "解壓截圖中..."
            # Clear old frames first
            import shutil
            if FRAMES_DIR.exists():
                shutil.rmtree(FRAMES_DIR)
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(FRAMES_DIR)
            version_file.write_text(target_version)
            print("Frames ready", flush=True)
        else:
            print(f"Frames already on disk ({frame_count}), skipping download", flush=True)

        # 3. Load LLM reranker
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


# Start background init
threading.Thread(target=_download_and_init, daemon=True).start()


def _resolve_frame(frame_path: str | None) -> str | None:
    if not frame_path:
        return None
    p = Path(frame_path)
    if p.exists():
        return str(p.resolve())
    persist_path = FRAMES_DIR / p.name
    if persist_path.exists():
        return str(persist_path.resolve())
    return None


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
    return gr.update(visible=True, value=f"## 系統啟動中...\n\n{_status}\n\n首次啟動需下載資料，請稍候。"), gr.update(visible=False)


# --- Gradio UI ---
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
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(),
               allowed_paths=[str(FRAMES_DIR)])
