# 從 Render 遷移到 Cloudflare（R2 + Containers）

日期：2026-07-25
狀態：待實作

## 背景

原本部署在 Render 免費方案的服務已經掛掉（被回收）。專案自 2026-05-29 起休眠，最後一筆 commit 是修好 reranker 的 0-picks bug。

本次遷移的目標是讓服務重新上線，並順手解掉幾個在 Render 時期被環境限制逼出來的妥協。

### 現況盤點（已驗證）

| 項目 | 狀態 |
|---|---|
| 程式碼 | 完整，25 個測試全綠 |
| `segments.pkl` | 182,616 筆，本地與 HF Hub 各有一份（39.9 MB） |
| frames | 本地 60,874 張／7.1 GB；HF Hub 有壓縮版 `frames_small.tar.gz`（1.47 GB） |
| HF Hub dataset | `odafeng/zhenhuan-data` 仍在線、public、未停用 |
| `humor_pool.pkl` | **不存在**（本地與 HF Hub 都沒有） |
| Render 服務 | 已掛 |

### 兩個必須先處理的問題

**1. humor pool 從未真正生效**

`core/reranker.py:123` 會找 `data/cache/index/humor_pool.pkl`，找不到就 fallback 到 full scan。這個檔案從來沒被產生或上傳過，代表 `d856585`（宣稱省 10 倍成本的離線預篩）在線上從未生效。

實測 full scan 的代價：single tier 共 60,872 段、514k 字元，以 Haiku input $1/M token 計，**每次查詢約 $0.51**。

這是上線的擋門條件，不是優化項目。

**2. requirements 有大量死依賴**

LLM-only 模式的 import chain：

- `interfaces/web_render.py` → `gradio`、`python-dotenv`
- `core/reranker.py` → `anthropic`、`schemas`

完全不碰 `faiss`、`torch`、`sentence-transformers`、`numpy`、`opencc`。但 `requirements.txt` 仍裝著它們，其中 torch 一個就 800 MB+。這是 `36e27c8`（LLM-only mode）只改執行期邏輯、沒回頭清依賴留下的殘留。

在 Render 上這只浪費建置時間；在 Containers，image 大小直接影響冷啟動，清掉的收益使用者感受得到。

## 目標

1. 服務重新上線，網址穩定
2. 冷啟動從 Render 免費方案的約 50 秒降到 2–3 秒
3. 每次查詢成本從 $0.51 降到 pool 規模對應的零頭
4. 圖片走 CDN，不再佔用應用程式記憶體

## 非目標

- 不重寫 Gradio UI（Render 掛掉是部署問題，不是 app 問題）
- 不重建索引、不重跑 OCR、不重抽幀（既有資料完整可用）
- 不恢復向量檢索路徑（LLM-only 模式已是預設）

## 架構

```
使用者
  │
  ▼
Worker (TypeScript)  ── /frames/*  ──►  R2 bucket (private)
  │                      + Cache API       60,874 張 jpg
  │ 其餘路徑
  ▼
Container (Python 3.11)
  ├─ Gradio UI
  └─ reranker  ──►  Anthropic API (Haiku)
```

## 元件

### 新增

| 檔案 | 職責 |
|---|---|
| `worker/index.ts` | 路由。`/frames/*` 走 R2 binding + Cache API；其餘用 `container.fetch()` 轉發到 Container |
| `wrangler.jsonc` | Container 定義、R2 binding、instance type |
| `scripts/upload_frames_to_r2.py` | 一次性上傳 frames 到 R2 |
| `interfaces/web_cf.py` | Cloudflare 版 Gradio entry，由 `web_render.py` 改寫 |

### 修改

| 檔案 | 改動 |
|---|---|
| `Dockerfile` | entrypoint 改為 `interfaces.web_cf`；套用精簡後的 requirements |
| `requirements.txt` | 從 10 個套件砍到 3 個：`gradio`、`anthropic`、`python-dotenv` |

### 依賴清單的職責分離

兩份清單服務不同對象，**不應該互相同步**：

| 清單 | 對象 | 內容 |
|---|---|---|
| `pyproject.toml` | 本機開發與測試 | 維持完整。測試會 import `searcher`（需 faiss）與 `media_extractor`（需 Pillow） |
| `requirements.txt` | Container runtime | 只留 3 個。runtime 只走 `web_cf` → `gradio`／`dotenv`，`reranker` → `anthropic`／`schemas` |

安全性已驗證：`core/__init__.py` 與 `services/__init__.py` 皆為空，`schemas.py` 只用 stdlib，因此 `import core.reranker` 不會連帶拉進 `searcher` 或 `media_extractor`。

Pillow 不需顯式列入 runtime — runtime 沒有任何模組直接 `import PIL`，Gradio 會以傳遞依賴帶入。

### 產生

| 檔案 | 說明 |
|---|---|
| `data/cache/index/humor_pool.pkl` | 遷移前在本地跑 `scripts/build_humor_pool.py`，成本約 $1，產物打進 image |

### `web_cf.py` 與 `web_render.py` 的差異

1. 移除啟動時從 HF Hub 下載 frames 的整段邏輯（圖片已在 R2）
2. 移除 `PERSIST_DIR` / tarfile 解壓邏輯
3. `_resolve_frame()` 改為回傳相對路徑 `/frames/<檔名>`，不再回傳本機絕對路徑
4. 移除 `allowed_paths`（不再從本機檔案系統供圖）
5. 保留 loading 畫面（Container 冷啟動仍需 2–3 秒）

## 資料流

```
查詢
  → Gradio
  → reranker 讀記憶體中的 humor_pool
  → Haiku 選圖
  → 回傳 caption + /frames/ep01_5.0s.jpg
  → 瀏覽器向 Worker 請求該路徑
  → Worker 查 Cache API
      ├─ hit  → 直接回傳
      └─ miss → 讀 R2 → 回傳並寫入 cache
```

## 關鍵決策

### D1：`segments.pkl` 與 `humor_pool.pkl` 打進 image，不放 R2

Containers 的磁碟是 ephemeral，容器睡著後重置。放 R2 等於每次冷啟動要重新下載；打進 image 則由 Cloudflare 全球預先 pre-fetch，冷啟動維持 2–3 秒。40 MB 對 20 GB 的 image 上限微不足道。

### D2：frames 放 R2，不進 image

1.47 GB 打進 image 會拖垮部署與冷啟動。且 Gallery 每次只載入當次結果的 5 張，實際流量很小。R2 egress 不收費。

### D3：用壓縮版 frames（1.47 GB），不用本地原圖（7.1 GB）

壓縮版已在線上驗證可用、載入快、只佔 R2 免費額度 10 GB 的 15%（原圖會佔 71%，沒有餘裕）。

若之後覺得畫質不足，改上傳原圖即可，不影響其他設計。

### D4：圖片走 Worker + R2 binding，不用 r2.dev 也不用 custom domain

`r2.dev` 官方明言有 rate limit 且不適合 production。custom domain 需要自有網域。Worker 本來就要負責路由，順便供圖不增加元件，且 bucket 可保持 private。

### D5：遷移前先建 humor pool

不建的話上線後每次查詢 $0.51。一次性成本約 $1。

## 錯誤處理

| 情境 | 處理 |
|---|---|
| Gradio 的 SSE／WebSocket | 一律用 `container.fetch()`，**不可**用 `containerFetch()`（後者不支援 upgrade，會靜默失敗） |
| Container 冷啟動 | 用 `startAndWaitForPorts()`（20 秒上限），前端維持現有 loading 畫面 |
| Anthropic API 失敗 | 沿用現有處理（`99f2a3a` 已把 API credit 錯誤浮到 UI） |
| R2 物件不存在 | Worker 回 404 並記 log；Gallery 略過該張（`_resolve_frame` 既有的 falsy 過濾已涵蓋） |
| Container 記憶體不足 | 起手用 `basic`（1 GiB）；不足再升 `standard-1` |

## 測試策略

1. **既有 25 個測試維持綠燈** — 跑在本機 venv（`pyproject.toml`），不受 requirements 精簡影響
2. **新增 `_resolve_frame()` 單元測試** — 驗證 `ep01_5.0s.jpg` → `/frames/ep01_5.0s.jpg` 的路徑轉換
3. **精簡 requirements 的驗證方式是 `docker build` + 本地起 container** — 不是跑 pytest。要證明的是「只裝 3 個套件時 app 仍能啟動並完成一次查詢」，pytest 用的是另一份依賴清單，驗證不到這件事
4. **`wrangler dev` 本地驗證** — routing 正確、R2 讀得到圖、Gradio 搜尋不斷線
5. **上線後手動 smoke** — 查詢一次、確認圖片載入、確認結果有回來（SSE 沒斷）

## 風險

| 風險 | 影響 | 緩解 |
|---|---|---|
| Containers 仍在 beta | API 可能變更、無 SLA | 接受。Worker 與 R2 都是 GA |
| Gradio SSE 在 Worker 後失效 | 搜尋結果不回來 | 用 `fetch()`；`wrangler dev` 先驗證 |
| 精簡 requirements 踩到隱性 import | 執行期爆炸 | `docker build` 後本地起 container 完成一次查詢（pytest 驗證不到，見測試策略 3） |
| 上傳 60,874 個小檔 | 6 萬次 Class A 操作 | 在免費額度 100 萬次／月內 |
| humor pool 品質不如預期 | 選圖變差 | 建完先本地查幾筆比對再部署 |

## 成本

| 項目 | 費用 |
|---|---|
| Workers Paid | $5／月（已有） |
| R2 儲存 1.47 GB | $0（10 GB 免費額度內） |
| R2 egress | $0（不收費） |
| R2 Class A（上傳 6 萬次） | $0（100 萬次／月免費額度內） |
| Container 運算 | 低用量下在含量內（25 GiB-hours 記憶體、375 vCPU-分鐘） |
| 建 humor pool | 一次性約 $1 |
| 每次查詢 | 從 $0.51 降到 pool 規模對應的零頭 |

## 實作順序

細節留給實作計畫，此處僅列相依關係：

1. 建 humor pool（本地，需 API 費用）→ 擋住後續所有步驟
2. 精簡 requirements + `docker build` 驗證 → 與 1 可並行
3. 建 R2 bucket + 上傳 frames → 與 1、2 可並行
4. 寫 `web_cf.py` + `worker/index.ts` + `wrangler.jsonc` → 需要 2 完成
5. `wrangler dev` 本地驗證 → 需要 3、4 完成
6. 部署 → 需要 1、5 完成
7. 線上 smoke 驗證
