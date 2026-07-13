# 企業級 LLM-Wiki 的 Ingest 階段:HKUDS nanobot vs LangGraph

先界定一下場景:企業 LLM-wiki 的 ingest 階段,通常要處理的是——多來源文件抓取(Confluence、Notion、PDF、Git repo...)、格式解析、chunking 策略(可能依文件類型不同)、metadata 抽取、去重/增量更新判斷、embedding、寫入向量庫、失敗重試、排程觸發、以及全流程的可觀測性與稽核紀錄。這本質上是一條**資料管線(data pipeline)**,只是中間可能夾雜 LLM 呼叫(例如用 LLM 做摘要、分類、抽取 metadata)。

## HKUDS nanobot

**優點**
- 極輕量、程式碼易讀,上手快,適合快速原型驗證某個 ingest 邏輯(例如先驗證「LLM 抽取 metadata」這個單一步驟)。
- 內建 cron scheduling,對「定期重新爬取某個來源」這種需求算是開箱即用。
- 原生 MCP 相容,如果你的資料來源已經有 MCP server(例如 Confluence MCP、Git MCP),接進來很方便。

**缺點**
- 它的設計核心是「chat agent 迴圈」(訊息進來 → LLM 決定要不要用工具 → 回應),本質上是對話導向,不是為多階段、有分支、有狀態機的 ETL 管線設計的。ingest 階段真正需要的「條件分支」「失敗重試節點」「部分完成後恢復」這些管線控制邏輯,nanobot 沒有對應的抽象,你得自己在 agent loop 外面再包一層排程/管線邏輯。
- 專案仍非常年輕(roadmap 上連「長期記憶」都還是待辦項目),缺乏企業級關注的東西:分散式執行、任務佇列整合、細粒度的可觀測性/追蹤、SLA 保證。
- 社群與生態系規模小,遇到問題時的參考資料、第三方整合(向量庫、文件解析套件的官方連接器)遠不如 LangChain 生態系豐富。
- 沒有專門的持久化/checkpoint 機制來處理「處理到一半掛掉,從哪裡繼續」這種企業 ingest 管線常見需求。

## LangGraph

**優點**
- 圖狀態機的模型剛好對應 ingest 管線的形狀:每個節點對應一個處理步驟(解析、分類、chunking、embedding、寫入),conditional edge 可以依文件類型或處理結果走不同分支(例如 PDF 走 OCR 分支、Markdown 走純文字分支)。
- 內建 checkpointer,可以把每個節點執行後的狀態存下來,天然支援「中斷後恢復」「失敗重試」「部分文件處理失敗不影響整批」這類企業管線必要的容錯需求。
- 生態系成熟:LangChain 的 document loaders、text splitters、vector store 整合都可以直接拿來用,不用重造輪子;LangSmith 提供可觀測性、trace、成本追蹤,對企業要做稽核、除錯非常關鍵。
- 支援 human-in-the-loop——如果 ingest 流程需要人工審核某些自動抽取的 metadata 或摘要,LangGraph 有現成的中斷/恢復機制。
- 在生產環境的採用案例多,招人、找解法都容易。

**缺點**
- 相對重,學習曲線較高,對於「只是想跑個簡單腳本」的場景會顯得殺雞用牛刀。
- 圖的設計如果沒規劃好,狀態物件會越長越肥,除錯時反而不如簡單腳本直觀。
- 它終究是「agent/LLM 工作流」框架,不是專門的資料工程工具——如果 ingest 的重點其實是大量的純 ETL(沒什麼 LLM 推理成分),用 Airflow / Dagster 這類專門的資料管線工具搭配 LangChain 的元件,可能比整個 ingest 都套進 LangGraph 更合適。

## 結論

**建議用 LangGraph,而不是 HKUDS nanobot。**

理由很直接:企業級 ingest 管線最在意的是「容錯、可恢復、可觀測、可分支」,這剛好是 LangGraph 的核心設計目標;而 nanobot 是為「輕量個人助理」場景設計的對話 agent,拿來扛企業級、多來源、需要重試與稽核的資料管線,等於是在框架的能力邊界外硬用,後續要補的基礎設施(排程、重試、追蹤)反而要自己造。

如果 ingest 邏輯裡 LLM 只是輔助角色(例如只用來抽 metadata、生摘要),另一個值得考慮的組合是:用 **LangChain 的元件(loaders/splitters)+ 專門的資料管線工具(Airflow/Dagster/Prefect)做整體排程**,LangGraph 則專門處理其中「需要 LLM 判斷/分支」的子流程——這樣可以避免把整個 pipeline 硬塞進一個 agent 框架裡。但如果你只是要在兩者之間二選一,LangGraph 明顯是更適合企業 ingest 場景的選擇。


好問題。跳出這兩個框架來看,ingest 階段本質上是一條資料管線,LLM 只是其中一個環節(做摘要、抽取、分類),所以我會從「這是資料工程任務,不是 agent 任務」的角度來選型。分成兩層來看:

## 專門的 LLM Ingestion 工具:LlamaIndex(IngestionPipeline)

LlamaIndex 有專門為這個場景設計的模組,叫 `IngestionPipeline`,概念上就是把「載入 → 轉換(chunking、metadata 抽取)→ embedding → 寫入向量庫」串成一條可配置的 pipeline,而且原生支援:
- **文件級別的去重與增量更新**(靠 docstore 記錄每份文件的 hash,重跑時只處理有變動的部分,這對企業 wiki 這種持續更新的來源非常關鍵)
- **並行執行**,可以用多執行緒/多進程加速大批次的處理
- 大量現成的 document loader(Confluence、Notion、Google Drive、Slack、各種檔案格式)和 vector store 整合,不用自己寫 connector

相較 LangGraph(设计給「有狀態、可能循環的 LLM 工作流」),LlamaIndex 的 IngestionPipeline 更貼近 ingest 這個具體場景的形狀——因為它就是為了「文件進、向量出」這條線設計的,不需要你自己去搭建狀態機。

## 排程與容錯:交給正統的資料管線工具

如果 ingest 要跑在企業排程系統上(定期同步、要有重試策略、要能監控每個任務的狀態、失敗要告警),這部分我會交給 **Airflow / Dagster / Prefect** 這類專門的編排工具,而不是讓 LangGraph 或任何 LLM 框架承擔這個角色。理由:
- 這些工具在「任務相依關係管理」「重試策略」「監控告警」「回溯執行(backfill)」上已經非常成熟,是資料工程的標準解法,LLM 框架在這方面都是後補的。
- 把 LLM 相關的處理(例如 LlamaIndex 的 ingestion pipeline,或需要 LLM 做判斷的步驟)包裝成 Dagster/Airflow 裡的一個 task/asset,職責分離清楚:排程工具管「何時跑、跑失敗怎麼辦」,LLM 工具管「怎麼把文件變成向量」。

## 我的具體建議組合

對企業級 LLM-wiki 的 ingest 階段,我會建議:

**LlamaIndex(IngestionPipeline + loaders)負責文件處理邏輯,搭配 Dagster 或 Airflow 做排程與編排**,兩者之間用簡單的介面串接(例如 Dagster asset 內部呼叫 LlamaIndex pipeline)。

理由總結:
- LlamaIndex 是專門為「文件轉知識庫」設計的工具,內建去重/增量更新機制,直接命中企業 wiki 持續更新的核心痛點,比 LangGraph 或 nanobot 都更省事。
- 排程、重試、監控這些非 LLM 專屬但企業級系統必要的能力,交給 Dagster/Airflow 這種本來就是為此而生的工具,不需要在 agent 框架裡重新發明。
- 這樣的分工也讓系統更好維護:資料工程師改排程邏輯不需要碰 LLM 相關程式碼,反之亦然。

如果你後續要做的是「查詢階段」(retrieval + 生成回答),那時候 LangGraph 會更適合登場——因為那才是真正需要「條件分支、多輪推理、human-in-the-loop」的場景。ingest 跟 query 這兩個階段用不同的工具組合,其實是很常見的架構分工。