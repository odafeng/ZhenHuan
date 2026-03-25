# Zhenhuan Reaction Retrieval Project Blueprint

## Overview
Build a decontextualized retrieval system that matches user input text to drama subtitle segments via literal/surface-level similarity, then extracts the corresponding frames or clips. The humor comes from the context gap between the user's modern query and the original drama context.

## Core Pipeline
User Query → Text Embedding → Subtitle Retrieval (multi-window) → Merge & Rerank → Time Mapping → Frame/Clip Extraction → Output

---

## Design Decisions

### Embedding Model Strategy

MVP 主路線：本地中文 embedding（GanymedeNil/text2vec-large-chinese 或 shibing624/text2vec-bge-large-chinese）  
Benchmark 對照：另跑一份 OpenAI text-embedding-3-small  
評估後決定：以 top-k 命中率、reaction 感、延遲為指標，不靠印象選

選擇邏輯：
- 本地中文模型：免費、可離線、批量建庫快、反覆重建 index 無壓力，適合快速迭代
- OpenAI embedding：部署簡單、穩定，適合之後接 web/API
- 所有 embedding model 對反諷 / 弦外之音等 pragmatic meaning 本質上都弱，MVP 階段專注字面匹配與場景匹配即可，語意深度留給 V3 emotion tagging 和 V4 multimodal reranking 處理

### Text Window Design

同時建三種檢索單位，實測後決定主力：

| Tier | Window Size | 用途 | 備註 |
|------|-------------|------|------|
| single | 當句 | 名台詞、短 punchline | 精準但缺上下文 |
| local_window | 前1句 + 當句 + 後1句 (3句) | **預設主力** | 平衡精準與語境 |
| expanded_window | 前2句 + 當句 + 後2句 (5句) | 鋪陳式嘲諷、委婉反擊、情緒累積 | 避免語意稀釋 |

所有 tier stride = 1，每句字幕同時產生 3 個 representation。

**關鍵 constraint：時間上限優先於句數**
- 總時長上限 8–10 秒（first-class constraint）
- 句數為 soft guide
- 若整體超過時間上限，裁掉最遠那句
- 原因：SRT 切法不穩定，有些句子被拆成多條 entry，純靠句數會踩雷

檢索流程：三個 tier 各自搜 top-k → merge → rerank

### Serving Roadmap

| Phase | 形式 | 目的 |
|-------|------|------|
| 1 | Local script / notebook | 快速驗證 query → top-k → frame |
| 2 | Gradio web demo | 視覺化比對 top-k、**順便做人工標註** |
| 3 | FastAPI | 核心穩定後包 API |
| 4 | LINE bot | 最終 serve |

不要反過來。最大風險是檢索不對味和資料結構要重改，不是 serve 不夠漂亮。

**Caching（從 Phase 2 開始）：** 檔案級即可，不需 Redis  
快取對象：subtitle table、embedding index、extracted frame/clip

---

## Data Structure

Data models 定義在 `schemas.py`（見 Architecture 區塊），此處僅列補充欄位。

### annotations（Phase 2 開始累積，存 `data/annotations/`）
- query
- segment_id
- rank_position
- is_relevant（人工標註）
- note

---

## Folder Structure

```
zhenhuan/
├── config.py                  # 全域設定（dataclass-based, 未來直接轉 Pydantic Settings）
├── schemas.py                 # Input/Output data models（dataclass, 未來直接轉 Pydantic BaseModel）
│
├── core/                      # 純邏輯層，不依賴任何 framework
│   ├── __init__.py
│   ├── subtitle_parser.py     # SubtitleParser class
│   ├── segment_builder.py     # SegmentBuilder class
│   ├── embedder.py            # Embedder ABC + LocalEmbedder / OpenAIEmbedder
│   ├── searcher.py            # Searcher class（FAISS 操作 + merge/dedup/diversity）
│   └── media_extractor.py     # MediaExtractor class（ffmpeg wrapper）
│
├── services/                  # Orchestration 層，串接 core modules
│   ├── __init__.py
│   ├── index_service.py       # IndexService：build pipeline（parse → segment → embed → save）
│   └── retrieval_service.py   # RetrievalService：query pipeline（embed → search → extract）
│
├── interfaces/                # 薄 adapter 層，只做 I/O 格式轉換
│   ├── __init__.py
│   ├── cli.py                 # Phase 1: CLI entry point
│   ├── gradio_app.py          # Phase 2: Gradio demo（含人工標註 UI）
│   └── api.py                 # Phase 3: FastAPI routes（呼叫 RetrievalService，不碰 core）
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures（sample SRT、mock embeddings、tmp paths）
│   ├── smoke/
│   │   ├── __init__.py
│   │   └── test_smoke.py      # 端到端冒煙測試：能跑完整 pipeline 不爆炸
│   └── unit/
│       ├── __init__.py
│       ├── test_subtitle_parser.py
│       ├── test_segment_builder.py
│       ├── test_embedder.py
│       ├── test_searcher.py
│       └── test_media_extractor.py
│
├── data/
│   ├── videos/
│   ├── subtitles/
│   ├── processed/
│   ├── cache/
│   └── annotations/
│
└── pyproject.toml
```

---

## Architecture

### 設計原則

三層分離：Core → Service → Interface。  
任何 interface（CLI / Gradio / FastAPI）只透過 Service 層操作，不直接碰 Core。  
Core modules 之間也盡量不互相 import，由 Service 層負責串接。

```
Interface (cli / gradio / api)
    ↓ 呼叫
Service (IndexService / RetrievalService)
    ↓ 組合
Core (SubtitleParser / SegmentBuilder / Embedder / Searcher / MediaExtractor)
    ↓ 使用
schemas (data models) + config (settings)
```

### config.py

集中管理所有可調參數，用 dataclass 定義。  
未來轉 FastAPI 時直接改繼承 `pydantic_settings.BaseSettings`，欄位不用動。

```python
@dataclass
class AppConfig:
    # Embedding
    embedding_backend: str = "local"          # "local" | "openai"
    local_model_name: str = "shibing624/text2vec-bge-large-chinese"
    openai_model_name: str = "text-embedding-3-small"

    # Window
    window_tiers: list[str]                   # ["single", "local_window", "expanded_window"]
    window_max_duration_sec: float = 10.0     # first-class constraint
    local_window_size: int = 3
    expanded_window_size: int = 5

    # Search
    top_k_per_tier: int = 10
    final_top_k: int = 5
    max_same_episode_scene: int = 2           # diversity constraint

    # Paths
    data_dir: Path
    cache_dir: Path
    videos_dir: Path
    subtitles_dir: Path
```

### schemas.py

定義所有 module 之間傳遞的 data model，用 dataclass。  
未來轉 FastAPI 時直接改繼承 `pydantic.BaseModel`，欄位不用動。

```python
@dataclass
class SubtitleEntry:
    subtitle_id: int
    episode_id: str
    video_path: str
    start_time: float
    end_time: float
    text: str

@dataclass
class Segment:
    segment_id: str
    episode_id: str
    video_path: str
    start_time: float
    end_time: float
    duration: float
    text_current: str
    text_window: str
    window_tier: str          # "single" | "local_window" | "expanded_window"
    mid_time: float
    embedding_text: str

@dataclass
class SearchResult:
    segment: Segment
    score: float
    frame_path: str | None = None

@dataclass
class SearchRequest:
    query: str
    top_k: int = 5

@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
```

### Core Layer

每個 module 都是 class，constructor 接 config，method 接 schema，回傳 schema。  
不依賴任何 web framework、不做 I/O 格式轉換、不管 HTTP。

**SubtitleParser**
- `parse(srt_path: Path) → list[SubtitleEntry]`
- 負責 SRT → structured data，處理編碼、時間軸解析

**SegmentBuilder**
- `__init__(config: AppConfig)`
- `build(entries: list[SubtitleEntry]) → list[Segment]`
- 依據 config 建立三種 window tier，時間上限為 first-class constraint

**Embedder**（ABC + 兩個實作）
- `embed_texts(texts: list[str]) → np.ndarray`
- `embed_query(query: str) → np.ndarray`
- `LocalEmbedder` / `OpenAIEmbedder` 實作同一介面，方便切換

**Searcher**
- `__init__(config: AppConfig)`
- `build_index(segments: list[Segment], embeddings: np.ndarray)`
- `search(query_embedding: np.ndarray) → list[SearchResult]`
- 內含 merge / scene-level dedup / diversity enforcement 邏輯

**MediaExtractor**
- `extract_frame(video_path: str, time_sec: float) → Path`
- `extract_clip(video_path: str, start: float, end: float) → Path`
- ffmpeg wrapper，回傳檔案路徑

### Service Layer

把 Core 串成完整流程。Interface 只呼叫這一層。

**IndexService**
- `__init__(config, parser, builder, embedder, searcher)`
- `build_all(srt_dir: Path) → None`
  - parse all SRTs → build segments → embed → build FAISS index → save to disk
- `load() → None`
  - 從 cache 載入已建好的 index

**RetrievalService**
- `__init__(config, embedder, searcher, extractor)`
- `search(request: SearchRequest) → SearchResponse`
  - embed query → search → extract frames → 回傳 SearchResponse
- 這就是未來 FastAPI route 會呼叫的唯一 method

### Interface Layer

薄 adapter，只負責：接收外部輸入 → 轉成 schema → 呼叫 service → 轉成外部輸出。

**cli.py**
- argparse，呼叫 `RetrievalService.search()`，print 結果

**gradio_app.py**
- Gradio UI，呼叫 `RetrievalService.search()`，顯示圖片 + 標註按鈕
- 標註結果存入 `data/annotations/`

**api.py**（Phase 3）
- FastAPI，`SearchRequest` 直接改成 Pydantic model
- `POST /search` → `RetrievalService.search()` → `SearchResponse`
- 幾乎只是換個殼，核心邏輯零修改

---

## Testing

### 原則

- 每個 Core class 都有對應的 unit test
- Smoke test 跑端到端 pipeline（用 fixture 小資料集）
- CI 跑 `pytest tests/` 全過才能 merge
- Core 和 Service 測試不需要真實影片或 GPU

### conftest.py（Shared Fixtures）

```python
# 3-5 條 SRT entries 的 sample 資料，涵蓋：
#   - 正常句
#   - 極短句（測 window 邊界）
#   - 跨時間上限的連續句（測 duration constraint）
# Mock embedding function（回傳固定維度 random vector，不需要真實模型）
# tmp_path 下的假 video file（用 ffmpeg 產生 10 秒黑畫面）
```

### Unit Tests

| Test File | 測什麼 | 關鍵 Case |
|-----------|--------|-----------|
| test_subtitle_parser.py | SRT 解析 | 正常 SRT、空檔、編碼問題、時間軸格式異常 |
| test_segment_builder.py | Window 建構 | 三種 tier 各自正確、時間上限裁切、首尾句邊界、stride |
| test_embedder.py | Embedding 介面 | output shape 正確、batch 不爆、local/openai 切換 |
| test_searcher.py | 檢索邏輯 | top-k 正確、dedup 有效、diversity constraint、空 index |
| test_media_extractor.py | ffmpeg 抽幀 | frame 檔案存在且可讀、clip 長度正確、time 超出範圍 |

### Smoke Tests

```python
# test_smoke.py
def test_full_pipeline(sample_srt, tmp_path, mock_embedder):
    """
    端到端：SRT → parse → build segments → embed → index → query → results
    驗證：不爆炸、回傳正確數量、每個 result 有 segment + score
    """

def test_index_persistence(sample_srt, tmp_path, mock_embedder):
    """
    建 index → 存檔 → 重新載入 → query 結果一致
    """
```

---

## Retrieval Logic

由 `RetrievalService.search()` 執行：

1. Input query（`SearchRequest`）
2. Convert to embedding（`Embedder.embed_query()`）
3. Search FAISS index across all 3 window tiers（`Searcher.search()`）
4. Merge results, deduplicate overlapping segments（同集同場景限制上限）
5. Enforce diversity：跨集、跨場景、跨角色分散
6. Rerank（初版可用 score 排序，後續加 cross-encoder 或 emotion tag；注意不要過度 rerank 掉字面歪樓結果）
7. Return top-k segments
8. Extract frame using mid_time（`MediaExtractor.extract_frame()`）
9. 回傳 `SearchResponse`

---

## ffmpeg Example

Extract frame:
```
ffmpeg -ss 752.4 -i ep01.mp4 -frames:v 1 output.jpg
```

Extract clip:
```
ffmpeg -ss 750 -to 756 -i ep01.mp4 -c copy clip.mp4
```

---

## Version Roadmap

V1: Core layer + unit tests + smoke test + CLI interface + local embedding  
V2: Add clip/GIF + caching + OpenAI embedding benchmark + Gradio demo with annotation  
V3: Add emotion tagging（處理反諷 / 弦外之音的語意 gap）  
V4: Add multimodal reranking + FastAPI（schemas 從 dataclass 換 Pydantic）  

---

## Key Insight
This is NOT a CV classification project.

It is:
Multimodal retrieval system (text → video segment)

更精確地說，這是一個 **decontextualized retrieval** 系統。  
好笑的點來自「字面匹配 + 語境錯位」，而不是真正理解語意。  
使用者丟一句現代口語，系統回傳古裝劇台詞，笑點在於原劇語境跟使用者語境的落差。

典型範例：  
Query = 社群貼文提到「發現他很大」  
→ 系統回傳：小公主寫書法寫出「大」字、太監說「幾次差點活不下來」、皇后說「既然是好東西你就多用」、甄嬛說「今夜鐵定是不能睡了」、眾嬪妃說「快拿出來看看」  
→ 每一句在原劇都是正經台詞，但拼在一起就變成完全不同的意思

### Implications

**Embedding 不需要太「聰明」：**  
這個 use case 需要的是字面層級的匹配。太強的語意理解反而可能把這些「歪樓」結果 rank 掉。這更加確認 MVP 用 local 中文 embedding + 不過度 rerank 是正確的方向。

**「對味」的定義跟一般 IR 不同：**  
- 一般 IR：relevance = 語意相關
- 這個系統：relevance = 字面相關 + 畫面表情到位 + 語境錯位越大越好笑
- Rerank 策略不能用傳統 semantic similarity 來排，需要自訂評估指標

**Top-k 多樣性很重要：**  
最佳輸出不是一張圖，而是一組回應串（如上例 5 張）。  
Top-k 之間需要確保多樣性——不能 5 張都來自同一場戲。  
Dedup 邏輯需考慮：同集同場景的 segments 只保留最佳一個，或至少限制同場景上限。

真正的難點不在 engineering，在 curation：
- SRT 品質（時間軸準不準、有沒有漏句）
- 「對味」的定義（見上方，不是傳統 relevance）
- Rerank 策略（字面 match > 語意理解）
- 人工標註的累積（從 Gradio 階段開始收集 ground truth）
- Top-k 多樣性（跨集、跨場景、跨角色）

---

## MVP Goal
Input: "你是在酸我嗎"  
Output: Top 3–5 matching scenes + frames（across multi-window tiers, 跨集跨場景分散）
