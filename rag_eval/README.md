# RAG 中文问英文官方文档检索评测

适配已部署的 FastAPI `POST /search`，返回字段为 `results[].source`, `page`, `score`, `text`。

## 安装
纯 Python 标准库，不需新增依赖。使用 Windows PowerShell，确保 API `http://localhost:8000/health` 正常。

## 运行

```powershell
python evaluate_retrieval.py collect --api http://localhost:8000 --questions questions.json --name bge-m3 --out runs/bge_m3.json --top-k 10
```

查看 `runs/bge_m3.json`：每题 Top-10 命中的文件名、页码、内容摘录及排名。

独立人工检查问题与官方 PDF，填写 `qrels.json`（用 `qrels.example.json` 作为**格式示例**；示例分数未核实，不可直接作为评测真值）。0=无关；1=部分相关；2=直接相关。对每一题**全部已入库 PDF**逐一标注；没有标记的文件会被视为无关，漏标会夸大指标。建议问题涵盖每个领域，并增加明确的无法回答问题供后续端到端评估。

```powershell
python evaluate_retrieval.py score --run runs/bge_m3.json --qrels qrels.json --k 1 3 5 10
```

同一批文档+切片，切换至基线模型并确保索引同步、API 已加载正确模型后，再执行 `collect` 保存成 `runs/bge_small.json`，最后：

```powershell
python evaluate_retrieval.py compare --runs runs/bge_small.json runs/bge_m3.json --qrels qrels.json --k 1 3 5 10
```

### 解释
- **Recall@K**：前 K 个不重复来源中命中的相关文件比例；需要对全部语料的相关文件完整标注。
- **MRR@K**：首个相关来源的倒数排名（宏平均）。
- **nDCG@K**：考虑相关性等级和排序位置的指标。

**评测单位是文件，不是切片**。脚本会把同一 PDF 的多个命中去重；因此它不衡量段落是否正确、引用页码是否可信，也不判断最终回答是否忠实。对于段落级别，需要让 `/search` 返回稳定 `chunk_id`/`document_id` 后单独做 chunk-level qrels。不同模型比较时，文档集、切片和检索参数应保持一致。

### 注意
- 本工具没有连接你本地的 API，因此未对你的实际知识库生成检索评分。
- 仅用于原始官方文档；本次不考虑 NREL02.pdf。
- 包含 `qrels.example.json` 只是展示标注文件结构，并非真实标注。
