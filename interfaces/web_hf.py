"""Gradio web UI for HF Spaces — downloads data from HF Dataset at startup."""

import os
import tarfile
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

# --- Download data from HF Dataset on startup ---
DATA_REPO = "odafeng/zhenhuan-data"
CACHE_DIR = Path("data/cache/index")
FRAMES_DIR = Path("data/processed")


def setup_data():
    """Download and extract index + frames from HF Dataset if not present."""
    if not (CACHE_DIR / "segments.pkl").exists():
        print("Downloading index...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(DATA_REPO, "index.tar.gz", repo_type="dataset")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(CACHE_DIR)
        print(f"Index extracted to {CACHE_DIR}")

    if not FRAMES_DIR.exists() or len(list(FRAMES_DIR.glob("*.jpg"))) < 1000:
        print("Downloading frames...")
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(DATA_REPO, "frames.tar.gz", repo_type="dataset")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(FRAMES_DIR)
        print(f"Frames extracted to {FRAMES_DIR}")


setup_data()

# --- Init services ---
from dotenv import load_dotenv
load_dotenv()

from config import AppConfig
from core.embedder import LocalEmbedder
from core.reranker import LLMReranker
from core.searcher import Searcher
from schemas import SearchRequest
from services.retrieval_service import RetrievalService

config = AppConfig()
config.ensure_dirs()

embedder = LocalEmbedder(config.local_model_name)
searcher = Searcher(config)
searcher.load(config.cache_dir / "index")

reranker = LLMReranker()
reranker.load_segments(config.cache_dir)

retrieval_svc = RetrievalService(config, embedder, searcher, reranker=reranker)


def search(query: str, top_k: int = 5):
    if not query.strip():
        return []

    request = SearchRequest(query=query.strip(), top_k=top_k)
    response = retrieval_svc.search(request)

    results = []
    for r in response.results:
        seg = r.segment
        frame = r.frame_path or seg.frame_path
        if frame and Path(frame).exists():
            caption = f"「{seg.text_current}」\n{seg.episode_id} | {seg.current_start:.0f}s"
            results.append((frame, caption))

    return results


# --- Gradio UI ---
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

    search_btn = gr.Button("找迷因！", variant="primary")

    gallery = gr.Gallery(
        label="搜尋結果",
        columns=3,
        height="auto",
        object_fit="contain",
    )

    search_btn.click(fn=search, inputs=[query_input, top_k_input], outputs=gallery)
    query_input.submit(fn=search, inputs=[query_input, top_k_input], outputs=gallery)


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
