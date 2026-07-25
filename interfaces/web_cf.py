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
