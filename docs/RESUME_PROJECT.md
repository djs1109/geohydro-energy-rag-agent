# 简历项目经历

## 多领域专业文档 RAG Agent（个人项目）

**技术栈：** Python、FastAPI、Streamlit、LangGraph、DeepSeek、BGE-M3、Qdrant、BM25、RRF、Reranker、SQLite、Docker Compose。

- 面向地质灾害、水文气象和新能源技术资料，设计并实现支持 PDF、DOCX、Markdown、TXT 的领域问答系统，具备批量上传、进度显示、多轮对话与证据追溯能力。
- 设计结构感知 Parent–Child 切片流程，以 Child 进行精准召回、Parent 补全上下文；融合 BGE-M3 向量召回、BM25、RRF 与 BGE Reranker，并记录来源页码、章节等元数据。
- 基于 SQLite 持久化文档与切片，结合 Qdrant 构建版本化索引及别名管理，实现 SHA256 去重、增量向量化和 Embedding 模型切换，减少重复上传和解析。
- 使用 LangGraph 编排知识检索、降雨统计与简化水文计算工具，采用 SQLite Checkpointer 管理会话，并通过 FastAPI、Streamlit 和 Docker Compose 完成系统部署。

**面试口述（40 秒）：** 我独立实现了一个面向地质灾害、水文气象和新能源资料的 RAG Agent。它能解析 PDF、Word 和 Markdown，使用结构化 Parent–Child 切片，BGE-M3 与 BM25 联合召回、RRF 融合和 Reranker 重排，再通过 DeepSeek 生成带来源的回答。我还将原始文档和索引解耦，实现增量入库、去重、模型切换，并用 LangGraph 管理计算工具和多轮会话，最终用 Docker Compose 部署整个系统。

**注意：** 不填写未经真实评估验证的准确率、延迟或性能提升百分比。
