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
- [x] Gradio app：輸入框 + 搜尋結果（截圖 + 台詞 + 集數時間）— interfaces/web_render.py
- [x] 下載截圖 — 用 gr.Gallery 內建下載鈕（未做自訂「一鍵複製」）
- [x] 部署 — Render（Dockerfile + persistent disk），非 HF Spaces；web_hf.py 為 HF 版備用

## Phase 7: 部署與成本優化
- [x] Render 部署：PORT env、背景執行緒下載資料、啟動中 loading 畫面
- [x] 資料搬上 HF Hub dataset `odafeng/zhenhuan-data`（index / embeddings / frames）
- [x] LLM-only 模式 — 拿掉 embedding model，省 1.3GB RAM 塞進 2GB 方案
- [x] 壓縮 frames（736MB）+ 自動升級畫質版本檔
- [x] Mobile RWD — 輸入框直排、gallery 手機 2 欄
- [x] Offline humor pool 預過濾（scripts/build_humor_pool.py）— runtime 成本降 10x
- [x] 修 reranker 0-picks bug（batch 編號錯位 + 全形逗號解析）

## Phase 8: 已遷移至 Cloudflare（2026-07-26）
- [x] 從 Render 遷移到 Worker + Containers + R2（見 docs/adr/0001）
- [x] humor_pool.pkl 建立（8370 candidates）並打包進 image — runtime 走 fast mode
- [x] frames 上傳 R2 bucket `zhenhuan`（60,874 張 / 6.94GB）+ Worker 代理供圖 + 邊緣快取
- [x] container requirements 精簡至 3 個套件（gradio / anthropic / python-dotenv）
- [x] 部署上線：https://zhenhuan.aidigest.workers.dev（standard-1 instance, Workers Paid）
- [x] 線上驗證：search SSE 端到端、R2 圖片絕對 URL、/frames 快取

### 部署備忘（維護時注意）
- 容器 env 變數（PUBLIC_BASE_URL / ANTHROPIC_API_KEY）設在 worker/index.ts 的 Container class，**不是** wrangler.jsonc 的 envVars（後者會被忽略）。
- ANTHROPIC_API_KEY 是 Worker secret（`wrangler secret put ANTHROPIC_API_KEY`），未進 git。
- `wrangler deploy` 會用本機 Docker 重 build image 再 push；Docker Desktop 若磁碟壓力大，push 大 layer 會斷（`docker image prune -af` 可解）。
- 記憶體不足時 instance_type 已從 basic 升 standard-1。

## Phase 9: 遺留待辦（不影響上線）
- [ ] 本地 data/cache/index 只有 segments.pkl，缺 embeddings.npy / faiss index（LLM-only 模式不需要）
- [ ] data/videos 原始影片已刪（frames 60,874 張還在，不影響檢索）
- [ ] 憑證輪替：R2 S3 key 與 cfat_ token 曾貼於對話，建議上線穩定後 rotate
