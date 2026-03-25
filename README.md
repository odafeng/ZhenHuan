# ZhenHuan 甄嬛反應圖檢索系統

Decontextualized retrieval system — match modern Chinese queries to 甄嬛傳 subtitle segments, extract the corresponding frames as reaction images.

The humor comes from the context gap between the user's modern query and the original drama context.

## Quick start

```bash
# Install dependencies
pip install -e ".[dev]"

# Build index from SRT files
python -m interfaces.cli --build --srt-dir data/subtitles

# Search
python -m interfaces.cli --query "你是在酸我嗎" --top-k 5

# Run tests
pytest
```

## Architecture

```
Interface (CLI / Gradio / FastAPI)
    ↓
Service (IndexService / RetrievalService)
    ↓
Core (SubtitleParser / SegmentBuilder / Embedder / Searcher / MediaExtractor)
    ↓
schemas (data models) + config (settings)
```

See [BLUEPRINT.md](BLUEPRINT.md) for full design decisions and [TODO.md](TODO.md) for development checklist.

## License

Private project.
