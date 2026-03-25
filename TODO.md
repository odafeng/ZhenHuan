# ZhenHuan V1 — Development TODO

## Phase 0: Project scaffold
- [x] Create GitHub repo "ZhenHuan"
- [x] Push scaffold (folder structure + empty modules + config + schemas)
- [x] Setup pyproject.toml with dependencies
- [ ] Verify `pytest` runs (0 tests, no errors)

## Phase 1: Core layer — 按 pipeline 順序開發
- [x] **1.1 schemas.py** — 定義 SubtitleEntry, Segment, SearchResult, SearchRequest, SearchResponse
- [x] **1.2 config.py** — 定義 AppConfig dataclass
- [x] **1.3 SubtitleParser** — core/subtitle_parser.py + unit test
  - [x] Parse .srt → list[SubtitleEntry]
  - [x] Handle encoding (UTF-8 / BIG5)
  - [x] Handle malformed timestamps
  - [x] test_subtitle_parser.py 通過
- [ ] **1.4 SegmentBuilder** — core/segment_builder.py + unit test
  - [ ] Build 3 window tiers (single / local_window / expanded_window)
  - [ ] Duration constraint (≤10s) as first-class
  - [ ] Edge cases: first/last subtitle, cross-episode boundary
  - [ ] test_segment_builder.py 通過
- [ ] **1.5 Embedder** — core/embedder.py + unit test
  - [ ] ABC + LocalEmbedder (text2vec-bge-large-chinese)
  - [ ] embed_texts() + embed_query()
  - [ ] test_embedder.py 通過 (mock vectors for unit test)
- [ ] **1.6 Searcher** — core/searcher.py + unit test
  - [ ] Build FAISS index
  - [ ] Search across tiers → merge → dedup → diversity
  - [ ] Save/load index to disk
  - [ ] test_searcher.py 通過
- [ ] **1.7 MediaExtractor** — core/media_extractor.py + unit test
  - [ ] extract_frame(video_path, time_sec) → Path
  - [ ] extract_clip(video_path, start, end) → Path
  - [ ] test_media_extractor.py 通過

## Phase 2: Service layer
- [ ] **2.1 IndexService** — services/index_service.py
  - [ ] build_all(): parse → segment → embed → index → save
  - [ ] load(): from cache
- [ ] **2.2 RetrievalService** — services/retrieval_service.py
  - [ ] search(SearchRequest) → SearchResponse

## Phase 3: Interface layer
- [ ] **3.1 CLI** — interfaces/cli.py
  - [ ] `python -m zhenhuan.interfaces.cli --query "你是在酸我嗎"`
  - [ ] Print top-k results + save frames

## Phase 4: Smoke tests
- [ ] **4.1 test_smoke.py** — full pipeline end-to-end
- [ ] **4.2 test_index_persistence.py** — build → save → load → same results

## Phase 5: Integration — 用真實資料跑
- [ ] 取得甄嬛傳 SRT 字幕檔 (至少 1 集)
- [ ] 取得對應影片檔 (至少 1 集)
- [ ] 跑完整 pipeline: build index → query → 看 frame
- [ ] 人工檢查 top-5 結果是否「對味」

---

## 開發順序邏輯
```
schemas + config (地基)
    ↓
SubtitleParser (有 SRT 才有東西)
    ↓
SegmentBuilder (有 entries 才能建 window)
    ↓
Embedder (有 segments 才能算 embedding)
    ↓
Searcher (有 embeddings 才能建 index)
    ↓
MediaExtractor (有 search result 才需要抽幀)
    ↓
IndexService + RetrievalService (串接所有 core)
    ↓
CLI (最後才接 interface)
    ↓
Smoke tests (整體驗收)
```
