# 1. 從 Render 遷移到 Cloudflare（R2 + Containers）

日期：2026-07-25

## 狀態

已接受

## 背景

服務原本部署在 Render 免費方案，現已被回收下線。專案自 2026-05-29 起休眠。

Render 免費方案的 2 GB 記憶體上限，是先前多個設計妥協的來源：

- `36e27c8` 拿掉 embedding model 改走 LLM-only，為了省 1.3 GB
- `938ca64` 改用壓縮版 frames，為了塞進記憶體
- `fe8c3ac` 把資料下載搬到背景執行緒，為了避開啟動逾時

也就是說，架構有一部分是被免費方案的限制塑形的，而不是被需求塑形的。

重新選平台時，實際需求是：

1. 能跑既有的 Python + Gradio container
2. 能存放並對外供應 60,874 張圖片（壓縮版共 1.47 GB）
3. 冷啟動不要像 Render 免費方案那樣要等約 50 秒
4. 成本可控

## 決策

遷移到 Cloudflare，採用 Worker + Containers + R2：

- **Container** 跑既有的 Gradio app 與 reranker，Python 程式碼不重寫
- **R2** 存放 frames，bucket 維持 private
- **Worker** 負責路由：`/frames/*` 以 R2 binding 供圖並用 Cache API 快取，其餘轉發到 Container

同時處理兩個既有問題：

- 產生並打包 `humor_pool.pkl`。這個檔案從未被產生過，導致 reranker 一直走 full scan，每次查詢約 $0.51
- 把 container 的 `requirements.txt` 從 10 個套件砍到 3 個。LLM-only 路徑不需要 faiss、torch、sentence-transformers

不採用的替代方案：

- **純 Worker 重寫（TypeScript）**：成本最低、無 beta 風險，但要丟掉可用的 Python reranker 重寫，風險與工時不成比例
- **Worker 前端 + Container API**：能避開 Gradio 的 SSE 轉發問題，但要重寫 UI 層，而 Gradio UI 本身沒有問題
- **r2.dev 公開網址供圖**：官方明言有 rate limit 且不建議 production
- **R2 custom domain 供圖**：效能最佳，但需要自有網域掛在同一個 Cloudflare 帳號

## 後果

### 正面

- 冷啟動從約 50 秒降到 2–3 秒
- 每次查詢成本從 $0.51 降到 pool 規模對應的零頭
- 圖片走 CDN 且 R2 egress 不收費，不再佔用應用程式記憶體
- 記憶體上限從 2 GB 提高到最多 12 GiB，先前為了塞進 2 GB 的妥協不再必要
- Python 程式碼不需重寫

### 負面與取捨

- **Cloudflare Containers 仍在 beta**：API 可能變更、無 SLA。Worker 與 R2 是 GA，但 Container 這層要承擔 beta 風險
- **從免費變成付費**：Containers 沒有免費方案，需要 Workers Paid $5／月。Render 免費方案是 $0
- **磁碟語意改變**：Containers 的磁碟是 ephemeral，容器睡著即重置。因此索引檔必須打包進 image，不能像 Render 那樣依賴 persistent disk 只下載一次
- **多一層轉發**：Worker → Container 的架構比 Render 直連多一跳，且 Gradio 的 SSE 必須用 `container.fetch()` 而非 `containerFetch()`，用錯會靜默失敗
- **綁定 Cloudflare**：`wrangler.jsonc` 與 R2 binding 是平台專屬設定，日後要再搬家需要重做這一層。Dockerfile 本身仍可攜
- **R2 免費額度 10 GB 只夠放壓縮版**：若日後改用原圖（7.1 GB）會佔掉 71%，餘裕不多
