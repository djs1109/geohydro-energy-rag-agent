# GeoHydro-Energy RAG Agent

面向地质灾害、水文气象与新能源技术资料的专业 RAG 智能问答原型。

## 核心能力

- 批量上传 PDF、DOCX、Markdown、TXT；显示解析与向量化进度，SHA256 去重。
- 结构感知 Parent–Child 切片，保留章节与来源页码信息。
- BGE-M3 + Qdrant 向量召回，结合 BM25、RRF 和 BGE Reranker。
- Embedding 模型切换及 Qdrant 索引别名 `rag_active` 自动管理。
- LangGraph 多轮 Agent、专业水文计算工具、DeepSeek 生成及证据引用核查。
- FastAPI + Streamlit + SQLite + Docker Compose。

## 推荐项目结构（与已有工程保持兼容）

```text
agent-learning/
├── api_main.py                 # FastAPI 路由
├── streamlit_app.py            # Streamlit 主界面
├── rag_core.py                 # LangGraph、检索、引用处理
├── document_ingestion.py      # 上传与解析
├── structured_chunking.py     # Parent-Child 切片
├── parent_retrieval.py        # Parent 召回展开
├── document_store.py          # SQLite 文档/切片存储
├── embedding_manager.py       # 向量化与 Qdrant 别名
├── upload_tasks.py            # 批量任务与进度
├── session_store.py           # 会话存储
├── manage_index.py            # 索引维护 CLI
├── pages/                     # Streamlit 子页面
├── tests/                     # pytest 测试
├── docs/                      # 架构及开发文档
├── Dockerfile.api
├── Dockerfile.web             # 若实际文件名不同则按本地为准
├── compose.yaml
├── requirements-api.txt
├── requirements-web.txt
├── .env.example
├── .gitignore
└── README.md
```

先不把 Python 源码强行搬入 `src/`，以免破坏 Docker COPY、模块导入与 Streamlit 页面路径。

## 运行

1. 复制 `.env.example` 为 `.env`，填写个人 DeepSeek API Key，按照本地 compose 配置检查环境变量。
2. `docker compose up -d --build`
3. API 文档：`http://localhost:8000/docs`；Web：`http://localhost:8501`
4. 上传具有使用权限的官方技术 PDF，测试 `/search` 和聊天证据引用。

## 数据安全与可复现性

- 不提交 `.env`、原始 PDF、SQLite、模型权重、向量索引、个人文件及任何 API Key。
- 第三方技术报告仅记录来源链接，不默认重新分发原文。
- 结构化切片版本变更需重新切片并重建向量；单纯更换 Embedding 模型只需重新编码持久化的文本切片。
- 复杂 PDF 的表格、跨栏、公式识别和复杂问题召回仍需独立验证。
- 只有在真实评测后才公布 Recall@K、MRR、nDCG 等指标。

> 本 README 为根据当前项目设计整理的公开仓库初稿，正式发布前需要核对实际目录、接口和环境变量。
