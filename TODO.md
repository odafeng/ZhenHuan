# ZhenHuan V1 — Development TODO

## Phase 0: Project scaffold
- [x] Create GitHub repo "ZhenHuan"
- [x] Push scaffold (folder structure + empty modules + config + schemas)
- [x] Setup pyproject.toml with dependencies
- [x] Verify `pytest` runs — 21 passed

## Phase 1: Core layer
- [x] **1.1 schemas.py** — SubtitleEntry, Segment, SearchResult, SearchRequest, SearchResponse
- [x] **1.2 config.py** — AppConfig dataclass
- [x] **1.3 SubtitleParser** — 3/3 tests passed
- [x] **1.4 SegmentBuilder** — 4/4 tests passed
- [x] **1.5 Embedder** — 4/4 tests passed (mock)
- [x] **1.6 Searcher** — 5/5 tests passed
- [x] **1.7 MediaExtractor** — 3/3 tests passed

## Phase 2: Service layer
- [x] **2.1 IndexService** — build_all + load
- [x] **2.2 RetrievalService** — search(SearchRequest) → SearchResponse

## Phase 3: Interface layer
- [x] **3.1 CLI** — interfaces/cli.py

## Phase 4: Smoke tests
- [x] **4.1 test_full_pipeline** — end-to-end passed
- [x] **4.2 test_index_persistence** — save/load roundtrip passed

## Phase 5: Integration — 用真實資料跑
- [x] OCR 76 集燒錄字幕 + 參考 SRT 修正（~91% 修正率）
- [x] 76 集影片 pre-extract frames（60,872 唯一時間點）
- [x] Build index（182,616 segments，全部有 frame_path）
- [x] LLM 模式設為預設（Claude 找最好笑的回應）
- [x] CLI query 自動開圖

## Phase 6: Gradio Web UI
- [ ] Gradio app：輸入框 + 搜尋結果（截圖 + 台詞 + 集數時間）
- [ ] 一鍵複製/下載截圖
- [ ] 部署到 Hugging Face Spaces 或本地
