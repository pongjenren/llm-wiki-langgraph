# LLM Wiki Ingest Eval Pipeline — 設計草案

> 狀態：討論稿（待與團隊確認方向後再開始實作）
> 目的：評估目前 ingest pipeline 的表現、定位問題所在，並支撐後續優化決策（特別是 page 更新從 full rewrite 改為 patch）

---

## 1. 背景與限制

### 1.1 資料與使用情境

- 來源：純文字英文週報，約 50–100 篇，檔名格式如 `na-report-2025-10-3.txt`
- 權限：不同單位的 report 只會更新同單位的 pages；eval 先以單一單位處理
- KB 使用者：(1) 人類直接閱讀 (2) query agent 依使用者 query 回答
- 可用模型：Qwen3.5 122B（ingest 使用）、gpt-oss-120b

### 1.2 內外分離

- 內部資料無法帶出 → **在外部開發 eval pipeline 與 golden dataset 建置流程，再搬進內部執行**
- 外部開發需要可替代的資料來源（見第 7 節 synthetic world）
- 所有 LLM 呼叫、資料讀取需透過 adapter，以 config 切換內外環境

### 1.3 Eval 的主要用途

1. 比較 pipeline 變體（prompt、模型、resolution 策略、更新策略）→ 需要穩定、可重複的指標，排序正確比絕對值重要
2. 找 failure mode → 需要能 drill down 到單一 report / entity / page 的 error analysis
3. **關鍵決策：page 更新從 full rewrite 改為 patch 是否較好**

---

## 2. 目前的 Ingest Pipeline（現況）

```
週報 ──► Extraction ──► Resolution ──► Page synthesis (full rewrite)
         (per report)    (3 stages)     
```

### 2.1 Extraction
- LLM 僅看當前 report，抽出 entity/concept 名稱、別名，以及該 report 中關於此 entity 的敘述（claims）
- **中間產物寫入 page 後即丟棄，不落地**

### 2.2 Resolution（三階段）
1. **Stage 1 — 完全比對**：entity name 是否等於既有 page 的 name 或 aliases → 是則直接視為同一 entity（不經 LLM 判斷）
2. **Stage 2 — 候選召回**：top-K，K = 10。推測為 string similarity 優先，候選不足時才用 vector similarity（**待確認，可能相反**）
3. **Stage 3 — LLM 判斷**：輸入為新 entity 的 name + claims，以及每個候選 page 的一句 description；一次看全部候選，選出一個，或判定都不是而建立新 page
4. **Merge 後**：新名稱自動加入該 page 的 aliases

### 2.3 Page synthesis
- 每次更新由 LLM **整頁重寫**（之後計畫改為 patch）
- 固定 section 只有 **timeline**，其餘 section 由 LLM 自行規劃
- page 的一句 description **每次 page 更新都會重寫**

### 2.4 日期
- 從檔名解析，格式為 `<單位>-report-YYYY-M-D.txt`（如 `na-report-2025-10-3.txt` = 2025 年 10 月 3 日，`na` 為單位名稱，eval 中忽略）
- 日期未補零，解析時需注意

---

## 3. 已識別的風險與假設

以下為討論中推論出的可能問題，需要 eval 驗證：

### 3.1 Resolution
| 風險 | 說明 | 後果 |
|---|---|---|
| Alias 汙染（Stage 1） | Merge 後新名稱**自動**加入 aliases，而 Stage 1 命中 alias 時**不經 LLM 判斷**。因此 Stage 3 只要錯一次，錯誤名稱就成為永久 alias，之後所有同名 entity 都會被直接吸進錯誤的 page；泛用詞（"Phase 2"、"v2"、"the migration"）尤其危險。錯誤會自我放大 | Over-merge |
| 候選池飢餓（Stage 2） | 若 string sim 優先且「候選足夠」即停止，則縮寫、改寫類 alias 可能永遠進不到候選清單 | Under-merge |
| 縮寫與改名 | String sim 和 embedding 對縮寫（`CPO` / `Co-Packaged Optics`）與中途改名都很弱 | Under-merge |
| 資訊不對稱（Stage 3） | 新 entity 側有完整 claims，候選側只有一句 description；description 太泛則無法分辨相似但不同的 entity | Over-merge |
| Description 漂移 | Description 每次重寫，同一 entity 的可辨識度會隨時間改變，形成「page 寫作品質 → 後續 resolution 品質」的回饋迴路 | 兩者皆可能 |
| Listwise 位置偏誤（Stage 3） | 一次看 10 個候選選一個，可能偏好清單前段；也需檢查「都不是」選項是否被充分使用（候選越多、越像，LLM 越傾向選一個） | Over-merge |

### 3.2 Page 更新
| | Full rewrite（現況） | Patch（計畫） |
|---|---|---|
| 資訊 | **侵蝕**：早期且之後不再被提到的事實在多次重寫後被丟掉 | **過時殘留**：新事實加入但舊值未刪 |
| 正確性 | 每次重寫都是新的 hallucination 機會，錯誤累積 | 新舊值並存造成矛盾 |
| 結構 | 每週版面可能重組，人類讀者難以追蹤 | 膨脹、重複段落、section 碎片化 |

- 自由 section 對 patch 不利：patch 需要穩定錨點。**建議改 patch 前先定 section 骨架（依 entity type），或在 page 建立後鎖定 section 只增不改名**

### 3.3 日期
- 檔名解析錯誤不會報錯，但會打亂 ingest 順序，連帶讓所有時效判斷錯誤
- Report 日期 ≠ 事件日期（週報會寫「上週完成」「預計下月上線」）

---

## 4. 評估指標（五層）

### L1 — Extraction（report-level）
- Entity precision / recall（可依 core / peripheral 分層或加權）
- Alias recall
- Description / claims 對原文的 faithfulness

### L2 — Resolution
- Cluster 指標（pairwise 或 B-cubed P/R）
- **Over-merge 與 under-merge 分開報告**（over-merge 會汙染內容，傷害較大）
- 逐階段歸因：
  - Stage 1 merge precision；alias collision（同一 alias 出現在多個 page）
  - **Alias 汙染追蹤**：每個錯誤加入的 alias，之後經由 Stage 1 造成多少次錯誤 merge（衡量單一錯誤的放大倍數）
  - Stage 3「建立新 page」選項的使用率 vs gold 中應建新 page 的比例
  - Stage 2 candidate recall@K，**依 alias 關係類型拆開**（拼寫變體 / 改寫 / 縮寫 / 改名）
  - Stage 3 條件準確率（gold 在候選中時，是否選對）
- **Description discriminability**：對 gold 中「相似但不同」的 entity 配對，只看兩邊 description，judge 能否分辨（= Stage 3 理論上限）
- **Teacher-forced vs free-running**：
  - Free-running：照實際流程跑，含錯誤傳播
  - Teacher-forced：每一步將 KB 狀態重設為 gold 再跑該步
  - 兩者差距 = 錯誤傳播的代價

### L3 — Page（最終快照）
- **Claim precision（faithfulness）**：page 拆成 atomic claims，對 source records / reports 驗證
  - 每個 claim 需標註為「歷史陳述」或「現況陳述」；timeline 中的舊值是合法歷史，不應判為 stale
- **Key-fact recall**：對每個 entity 的 gold key facts 清單
- **時效正確性**：現況陳述是否為最新值
- **Timeline**：事件 P/R、事件日期正確率、順序正確性（多數可規則化）
- **結構檢查**（規則）：格式、cross-link 正確性、重複或 near-duplicate page

### L4 — 縱向（每次 ingest 後的 page 快照）
- **Retention**：gold key fact 從首次出現後，在每個快照中是否仍存在（已被取代者除外）→ 畫成曲線
- **Staleness**：已被取代的事實仍以現況呈現
- **Contradiction**：新舊衝突值並存且未標註時間
- **Churn**：新週報只帶來局部變更時，無關段落被改動的程度
- **Page growth**：長度成長與重複陳述比例
- **Section stability**：相鄰版本 section 標題集合的 Jaccard similarity
- **Description stability**：相鄰版本 description 的語意變化

### L5 — Downstream QA
- Query agent 使用 KB 回答的正確率
- Baseline：query agent 直接對原始週報做 RAG
- 題目包含「as of 第 N 週」的時效題，可在縱向快照上執行
- 用途：驗證 L1–L4 指標是否與最終效用相關（page 分數上升但 QA 不變 → page 指標可能量錯東西）

### 規則檢查（附加）
- 權限：每個 page 版本的 source records 皆來自同單位 report
- 日期：解析結果在合理範圍、同單位週報大致每週一篇、無缺漏或重複

---

## 5. Eval Mode：Pipeline 需落地的資料

以 JSONL 依 run 儲存，eval 不依賴 production DB。

**run_meta**
- run_id、模型、各 prompt 版本、config hash、ingest 順序、seed

**extraction_records**
- record_id、report_id、檔名、解析出的 report_date、entity_name、aliases、description/claims

**resolution_decisions**
- record_id
- 最終決策階段（stage 1 / 2+3）與結果（new page / merge 至 page_id）
- Stage 1：命中的是 page name 或哪個 alias
- Stage 2：string sim 與 vector sim **各自的分數**、候選清單（建議記到 top-20 以上，方便離線掃 K 與 threshold）
- Stage 3：輸入的候選順序、選擇結果、理由
- Merge 後新加入的 aliases

**page_versions**
- page_id、version、觸發的 report_id、完整 content、description、section 標題列表、本次使用的 source_record_ids

`source_record_ids` 提供粗粒度 provenance，faithfulness 驗證時只需對相關 records 與 reports 查證，不需改生成 prompt。句子級 citation 可留待第二階段。

---

## 6. Golden Dataset

### 6.1 Gold 結構
- **Entity gold**：canonical name、aliases（每對標註關係類型）、entity type、core / peripheral、出現於哪些 reports
- **Key fact gold**：所屬 entity、敘述、生效週次、被哪條 fact 取代（以取代鏈推出任一時間點的「現況」）
- **Timeline gold**：entity、報告日期、事件日期、事件
- **QA gold**：問題、答案、依據的 key facts、as-of 週次

### 6.2 標註方案

| 方案 | 標註者 | 產出 | 優點 | 風險 |
|---|---|---|---|---|
| A. 自己 + 雙模型交叉 | 本人 + LLM 預標 | entity、alias cluster、key facts | 不依賴他人，可立即開始 | 缺 domain 知識，粒度判斷可能偏離使用者 |
| B. Expert 定點審核 | domain expert 每人 1–2 小時 | core 分層、爭議裁決、judge 校準 | expert 時間集中在關鍵處 | 需事先準備清單與協調 |
| C. Expert 出題 | 週報實際讀者 | QA 題目與答案 | 最貼近真實需求，負擔最輕 | 僅覆蓋 QA 視角 |
| D. Synthetic world | 程式生成（外部） | 全部 | ground truth 完美，可驗證 eval 本身 | 與真實資料有分佈差異 |

**建議：D + A + C，B 可選**
- D：外部開發與 meta-eval
- A：內部主要 entity / key fact gold
- C：請實際讀者每人寫 10–20 個真的會問的問題
- B：若爭取到，用於 core/peripheral 分層與 judge 校準

### 6.3 雙模型交叉預標
- Qwen 與 gpt-oss 各自獨立抽取同一份 report
- 不一致處優先人工審核
- **一致處也需隨機抽審**：Qwen 與 ingest 同源，兩者一致不代表正確
- Report-level 完整標註約 15–20 篇；KB-level entity 清單可接近窮舉

### 6.4 Open-world precision
- 系統抽到 gold 沒有的 entity 不一定錯（可能 gold 漏標）
- 採 pooling：未匹配的預測交給 judge 或人工判定 → 合理者回填 gold
- Gold 必須版本化，不同版本的分數不可直接比較

### 6.5 Annotation guideline
- 明確定義「什麼值得成為一個 entity / page」
- 小樣本上量 inter-annotator agreement
- 隨審核累積 edge cases 回寫 guideline

---

## 7. Synthetic World（外部開發用）

### 7.1 生成流程
1. 程式定義 world state：十餘個 project、數十個 tool / system / concept、時間軸 20–30 週的事件
2. 每週取相關事件，由 LLM 寫成週報風格文字，檔名模仿內部格式（`<單位>-report-YYYY-M-D.txt`，日期不補零）
3. 每句話來源事件已知 → entity、alias cluster、key facts、timeline、QA gold 全部自動取得

### 7.2 刻意放入的難點
- **長壽但沉寂的事實**：只在早期出現、之後不再提及 → 測 full rewrite 侵蝕
- **多次狀態變更**：同一屬性變 3–4 次 → 測 staleness / contradiction
- **一次性小事件** → 測 churn
- **錯誤後更正**
- **中途引入縮寫、中途改名** → 測 Stage 2 弱點
- **相似但不同的名稱**、**泛用詞 alias** → 測 Stage 1 汙染與 Stage 3 分辨力
- **報告日期 ≠ 事件日期**（「上週完成」「預計下月」）

### 7.3 Meta-eval（perturbation test）
- 對 page 注入已知錯誤（改數字、刪事實、合併兩個 entity、插入過時值）
- 確認每個指標有如預期反應，且幅度合理

---

## 8. 模型分配

| 角色 | 模型 | 理由 |
|---|---|---|
| Ingest | Qwen3.5 122B | 現況 |
| Judge / 主要預標 | gpt-oss-120b | 避免與 ingest 同源的 self-preference bias |
| 交叉抽取（建 gold） | 兩者皆用 | 用不一致排序審核優先度 |

- Judge 需以人工標註小樣本校準一致性
- 樣本量小 + LLM 非確定性 → 每個設定跑 2–3 次看分佈

---

## 9. 審核介面（Streamlit）

1. **Report entity 審核**：左側原文，右側兩模型結果聯集；可接受 / 拒絕 / 修改 / 補漏 / 標 core
2. **Alias cluster 審核**：拆分、合併 cluster，標註 alias 關係類型
3. **Key fact 審核**：以 entity 為單位依週次排列，標記取代關係
4. **QA 出題表單**：給 expert，只填問題、答案、大概時間點
5. **Judge 校準**：隨機抽 judge 判過的案例盲判

原則：
- 審核時不顯示結果來自哪個模型
- 每筆標註記錄 annotator、時間、gold 版本號

---

## 10. 可立即執行的實驗（eval 完成後）

1. **Stage 3 輸入 ablation**（teacher-forced，固定候選集離線重播）：
   - 一句 description（現況）
   - \+ 候選 aliases
   - \+ 最近 3 筆 timeline
   - \+ 完整 page
2. **Stage 3 候選順序 shuffle**：檢驗位置偏誤
3. **Stage 2 策略**：string sim 優先 / vector sim 優先 / 兩者聯集，比較各 alias 類型的 candidate recall@K
4. **Description 固定 vs 每次重寫**：對 resolution 的影響
5. **Full rewrite vs patch**：以 L4 縱向指標與 L5 QA 為主要比較依據

---

## 11. 階段規劃

| 階段 | 地點 | 內容 |
|---|---|---|
| 0 | 內外 | Pipeline 加 eval mode，落地第 5 節資料 |
| 1 | 外部 | Synthetic world 生成器 + 自動 gold |
| 2 | 外部 | 實作 L1–L5 指標，perturbation meta-eval |
| 3 | 內部 | Golden dataset pipeline：gpt-oss 預標、交叉抽取、Streamlit 審核、QA 收集 |
| 4 | 內部 | Full rewrite baseline → 第 10 節實驗 → patch 版本比較 |

---

## 12. 待確認事項

- [ ] Stage 2：string sim 與 vector sim 的優先順序與「候選不足」的判定條件
- [ ] Stage 3：候選順序如何決定（依相似度排序？）
- [ ] 檔名格式是否全部一致（有無例外檔名）

已確認：
- [x] Stage 2：K = 10
- [x] Stage 3：有「都不是 → 建立新 page」選項
- [x] Merge 後新名稱自動加入 aliases
- [x] 檔名最後一碼為日期；開頭為單位名稱，eval 中忽略
- [ ] 內部能投入審核的人力與時間（決定第 6.2 節方案）
- [ ] 改 patch 前是否先定 section 骨架
