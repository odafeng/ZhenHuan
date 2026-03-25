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
- [ ] 取得甄嬛傳 SRT 字幕檔 (至少 1 集)
- [ ] 取得對應影片檔 (至少 1 集)
- [ ] 跑完整 pipeline: build index → query → 看 frame
- [ ] 人工檢查 top-5 結果是否「對味」
