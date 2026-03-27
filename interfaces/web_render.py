"""Gradio web UI for Render deployment.

First run: downloads data from HF Dataset to persistent disk (/data).
Subsequent runs: reads directly from persistent disk (no re-download).
"""

import os
import subprocess
import tarfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR", "/data"))
INDEX_DIR = PERSIST_DIR / "cache" / "index"
FRAMES_DIR = PERSIST_DIR / "processed"
HF_REPO = "odafeng/zhenhuan-data"


def setup_data():
    """Download data from HF Dataset to persistent disk on first run."""
    from huggingface_hub import hf_hub_download

    if not (INDEX_DIR / "segments.pkl").exists():
        print("Downloading index from HF Dataset...", flush=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(HF_REPO, "index.tar.gz", repo_type="dataset")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(INDEX_DIR)
        print(f"Index ready at {INDEX_DIR}", flush=True)

    if not FRAMES_DIR.exists() or len(list(FRAMES_DIR.glob("*.jpg"))) < 1000:
        print("Downloading frames from HF Dataset...", flush=True)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        # Download all frame parts and reassemble
        parts = []
        for suffix in ["aa", "ab", "ac", "ad"]:
            fname = f"frames.tar.gz.part_{suffix}"
            try:
                p = hf_hub_download(HF_REPO, fname, repo_type="dataset")
                parts.append(p)
                print(f"  Downloaded {fname}", flush=True)
            except Exception:
                break

        if parts:
            # Reassemble and extract
            combined = PERSIST_DIR / "frames.tar.gz"
            with open(combined, "wb") as out:
                for p in parts:
                    with open(p, "rb") as inp:
                        while chunk := inp.read(8192 * 1024):
                            out.write(chunk)
            print("  Extracting frames...", flush=True)
            with tarfile.open(combined, "r:gz") as tar:
                tar.extractall(FRAMES_DIR)
            combined.unlink(missing_ok=True)
            print(f"Frames ready at {FRAMES_DIR}", flush=True)


setup_data()

# --- Init services ---
from config import AppConfig
from core.embedder import LocalEmbedder
from core.searcher import Searcher
from schemas import SearchRequest
from services.retrieval_service import RetrievalService

config = AppConfig()
config.cache_dir = INDEX_DIR.parent
config.output_dir = FRAMES_DIR
config.ensure_dirs()

embedder = LocalEmbedder(config.local_model_name)
searcher = Searcher(config)
searcher.load(config.cache_dir / "index")

reranker = None
try:
    from core.reranker import LLMReranker
    reranker = LLMReranker()
    reranker.load_segments(config.cache_dir)
    print("LLM mode ready", flush=True)
except Exception as e:
    print(f"LLM mode unavailable ({e}), embedding only", flush=True)

retrieval_svc_llm = RetrievalService(config, embedder, searcher, reranker=reranker) if reranker else None
retrieval_svc_emb = RetrievalService(config, embedder, searcher)


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


def search(query: str, top_k: int = 5, mode: str = "LLM（較慢但好笑）"):
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
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
