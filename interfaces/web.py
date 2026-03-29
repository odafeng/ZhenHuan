"""Gradio web UI for ZhenHuan reaction image search."""

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from config import AppConfig
from core.embedder import LocalEmbedder
from core.searcher import Searcher
from schemas import SearchRequest
from services.retrieval_service import RetrievalService

# --- Project root (resolve relative paths) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Init (once at startup) ---
config = AppConfig()
config.ensure_dirs()

embedder = LocalEmbedder(config.local_model_name)
searcher = Searcher(config)
searcher.load(config.cache_dir / "index")

# LLM reranker (optional — only if API key is set)
reranker = None
try:
    from core.reranker import LLMReranker
    reranker = LLMReranker()
    reranker.load_segments(config.cache_dir)
    print("LLM mode ready")
except Exception as e:
    print(f"LLM mode unavailable ({e}), using embedding only")

retrieval_svc_llm = RetrievalService(config, embedder, searcher, reranker=reranker) if reranker else None
retrieval_svc_emb = RetrievalService(config, embedder, searcher)


def _resolve_frame(frame_path: str | None) -> str | None:
    """Resolve relative frame path to absolute."""
    if not frame_path:
        return None
    p = Path(frame_path)
    if p.exists():
        return str(p.resolve())
    # Try relative to project root
    p2 = PROJECT_ROOT / frame_path
    if p2.exists():
        return str(p2.resolve())
    return None


def search(query: str, top_k: int = 5, mode: str = "LLM（較慢但好笑）"):
    """Search and return results for Gradio display."""
    if not query.strip():
        return []

    use_llm = "LLM" in mode and retrieval_svc_llm is not None
    svc = retrieval_svc_llm if use_llm else retrieval_svc_emb

    request = SearchRequest(query=query.strip(), top_k=int(top_k))
    response = svc.search(request)

    results = []
    for r in response.results:
        seg = r.segment
        frame = _resolve_frame(r.frame_path) or _resolve_frame(seg.frame_path)
        if frame:
            caption = f"「{seg.text_current}」\n{seg.episode_id} | {seg.current_start:.0f}s"
            results.append((frame, caption))

    return results


# --- Gradio UI ---
MOBILE_CSS = """
@media (max-width: 768px) {
    .row { flex-direction: column !important; }
    .gallery .grid-wrap { grid-template-columns: repeat(2, 1fr) !important; }
}
"""

with gr.Blocks(title="甄嬛傳迷因產生器") as app:
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
        mode_input = gr.Radio(
            choices=["Embedding（快速）", "LLM（較慢但好笑）"],
            value="LLM（較慢但好笑）" if reranker else "Embedding（快速）",
            label="搜尋模式", scale=1,
        )

    search_btn = gr.Button("找迷因！", variant="primary")

    gallery = gr.Gallery(
        label="搜尋結果",
        columns=3,
        height="auto",
        object_fit="contain",
    )

    search_btn.click(fn=search, inputs=[query_input, top_k_input, mode_input], outputs=gallery)
    query_input.submit(fn=search, inputs=[query_input, top_k_input, mode_input], outputs=gallery)


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(), css=MOBILE_CSS)
