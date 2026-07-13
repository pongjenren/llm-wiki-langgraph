# nanobot (HKUDS/nanobot) 簡介

**nanobot** 是一個開源的「超輕量級個人 AI Agent」專案，作者是 Xubin Ren（個人專案性質，非公司產品），精神上類似 Claude Code、Codex、OpenClaw 這類工具，但目標是把核心 agent loop 做得**極簡、可讀、可研究**。

核心設計理念：
- **極簡核心迴圈**：訊息從聊天管道進來 → LLM 決定是否要呼叫工具 → 記憶或 skill 只在需要時被拉進 context，而不是變成一個厚重的 orchestration 層。
- **多聊天管道整合**：Telegram、Discord、Slack、WeChat、Feishu、Matrix、QQ、DingTalk、WeCom、Email、WebSocket 等，幾乎是它最大量的開發精力所在。
- **MCP 支援**：可接入 MCP servers 作為工具擴充。
- **多 LLM Provider**：OpenAI、Anthropic、DeepSeek、Qwen、Kimi/Moonshot、MiniMax、VolcEngine、Ollama、vLLM、LM Studio 等。
- **記憶系統（"Dream"）**：一套簡化過的 memory 機制，用於個人助理式的長期記憶。
- **部署路徑**：Docker、Linux service、macOS LaunchAgent 等。

從 commit 紀錄可以看出這是一個**開發非常活躍、疊代速度極快**的專案（幾乎每天都有更新），社群關注度也不小（4萬+ stars）。但要注意：這是**單一開發者主導、以個人資源維護**的專案，不是企業級產品。

---

## 優點

1. **程式碼小而可讀**：核心 agent loop 刻意保持精簡，適合學習、二次開發、客製化。
2. **管道整合豐富**：如果需求是「讓同事在 Slack/Discord/Teams/Feishu 裡問問題」，這部分已經做得很成熟。
3. **MCP 生態相容**：可以接外部工具/資料源，理論上可以掛一個「文件檢索」MCP server 進去。
4. **多 Provider 彈性**：不綁死在單一 LLM 供應商，方便切換或做混合部署（本地模型 + 雲端模型）。
5. **疊代快、社群活躍**：bug 修復和功能上線速度快，Issue/PR 量大代表有人在用。

## 缺點（相對企業級需求而言）

1. **它是「個人助理 Agent」，不是「文件 ingestion / RAG pipeline」框架**。README 和架構圖裡完全沒有提到：
   - 文件解析（PDF/Office/HTML/Confluence/SharePoint 等）
   - Chunking 策略、embedding pipeline、向量資料庫整合
   - 版本控管、增量更新、資料血緣（lineage）
   這些正是「企業級 wiki ingest framework」的核心需求，nanobot 並沒有針對這塊設計，只有「Office 文件讀取」這種聊天附件層級的支援，遠不到系統化 ingestion 的程度。
2. **記憶系統偏個人化、非結構化**：Dream 記憶系統設計是給單一使用者的長期記憶（類似你自己在用的 memory 機制），不是給多租戶知識庫做語意檢索用的正規 RAG 儲存層。
3. **單一開發者維護，缺乏企業治理保證**：沒有 SLA、沒有正式的安全稽核流程（雖然有 SECURITY.md），版本疊代快也意味著 breaking change 風險較高（`nightly` branch 明確標示「breaking changes」）。
4. **權限與多租戶模型薄弱**：README 中看不到 RBAC、租戶隔離、審計日誌（audit log）等企業必備能力，聊天管道的「access control」偏向白名單層級，不是企業級身份治理。
5. **可觀測性有限**：雖然有 Langfuse 整合，但整體監控/日誌/告警體系不是為企業維運設計的。
6. **License 是 MIT**，商用沒問題，但也代表沒有官方支援保證。

## 適不適合作為企業級 LLM-Wiki 的 Ingest Framework？

**不適合直接拿來當 ingestion framework。** 原因很直接：nanobot 解決的問題是「Agent 怎麼跟人聊天、怎麼呼叫工具」，而你要的是「怎麼把大量企業文件穩定、可追蹤地轉成可檢索的知識庫」——這是兩個不同層次的問題，nanobot 完全沒有針對後者做設計。

比較合理的定位是：
- 把 nanobot（或它的 agent-loop 設計理念）當作**前端互動層**（聊天介面 + agent 決策），
- 但 ingestion pipeline（文件解析、chunking、embedding、向量索引、增量同步、權限過濾）應該用專門的框架來做，例如 LlamaIndex、Haystack、或自建 pipeline（配合 LangGraph 之類做 orchestration），
- 再透過 MCP 把這個自建的檢索服務掛給 nanobot（或任何 agent）當工具用。

如果你的目標真的是「企業級 wiki ingest framework」，我會建議把 nanobot 排除在候選之外，改去看專門做 RAG/ingestion 的框架，或直接自建 pipeline（考慮到你本身在做 RAG/memory architecture 研究，這條路對你來說可能更可控）。如果你只是想要一個聊天式的個人助理前端來「問」已經建好的知識庫，nanobot 才會是合理的選項之一。

有需要的話我可以再幫你搜一下目前比較成熟的企業級 RAG ingestion 框架做個對比。