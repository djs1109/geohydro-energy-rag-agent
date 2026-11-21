
# api_main.py

import logging
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import sqlite3
from fastapi import File, UploadFile

from document_ingestion import ingest_document
from upload_tasks import UploadTaskManager, create_upload_router
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from session_store import SessionStore
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from rag_core import (
    RAGConfig,
    build_rag_agent,
    run_agent,
    extract_evidence,
    validate_citations,
    get_current_turn_messages,
    extract_agent_trace
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AgentAPI")


# =====================================================
# 1. 定义请求和响应结构
# =====================================================

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    thread_id: str = Field(
        default="demo_session",
        min_length=1,
        max_length=128
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=20)


# =====================================================
# 2. 生命周期管理
# =====================================================


# =====================================================
# FastAPI 生命周期：SQLite 持久化
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing SQLite Checkpointer...")

    # 服务期间保持连接有效
    connection = sqlite3.connect(
        str(CHECKPOINT_DB),
        check_same_thread=False
    )

    checkpointer = SqliteSaver(connection)

    # 初始化数据库表
    checkpointer.setup()

    try:
        config = RAGConfig.from_env()

        app.state.checkpointer = checkpointer
        app.state.session_store = SessionStore(
            DATA_DIR / "sessions.sqlite"
        )
        app.state.bundle = build_rag_agent(
            config,
            checkpointer=checkpointer
        )

        app.state.agent_lock = Lock()
        app.state.upload_tasks = UploadTaskManager(app)


        logger.info("RAG Agent initialized with SQLite")

        yield

    finally:
        if getattr(app.state, "upload_tasks", None):
            app.state.upload_tasks.shutdown()
        app.state.bundle = None
        connection.close()

        logger.info("SQLite connection closed")



app = FastAPI(
    title="RAG Agent API",
    description="基于 LangGraph + DeepSeek + Qdrant 的专业知识智能体",
    version="0.1.0",
    lifespan=lifespan
)


# =====================================================
# 3. 健康检查
# =====================================================

@app.get("/health")
def health(request: Request):

    bundle = request.app.state.bundle

    return {
        "status": "ok",
        "collection": bundle.config.collection_name,
        "document_count": len(bundle.documents),
        "model": bundle.config.deepseek_model,
        "embedding_model": bundle.config.embedding_model_name,
        "index_alias": bundle.config.collection_name,
        "stored_chunks": bundle.document_store.count()
    }


# =====================================================
# 4. Agent 问答接口
# =====================================================

@app.post("/chat")
def chat(data: ChatRequest, request: Request):

    start = time.perf_counter()
    store = request.app.state.session_store

    if store.get(data.thread_id) is None:
        raise HTTPException(
            status_code=404,
            detail="请先创建会话"
        )
    try:
        with request.app.state.agent_lock:
            bundle = request.app.state.bundle
            result = run_agent(
                bundle,
                query=data.query,
                thread_id=data.thread_id
            )

        answer = result["messages"][-1].content
        
        session = store.get(data.thread_id)

        # 第一次对话生成标题
        if session["title"] == "新对话":
            title = (
                data.query[:25] + "..."
                if len(data.query) > 25
                else data.query
            )
            store.update_title(data.thread_id, title)
        else:
            store.touch(data.thread_id)

        # 只提取本轮消息
        current_messages = get_current_turn_messages(
            result["messages"]
        )

        # 当前轮检索证据
        evidence = extract_evidence(
            current_messages
        )

        # 当前轮工具执行轨迹
        trace = extract_agent_trace(
            current_messages
        )

        citation_report = validate_citations(
            answer,
            evidence
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "Chat completed | latency=%.3fs",
            elapsed
        )

        return {
            "thread_id": data.thread_id,
            "answer": answer,
            "trace": trace,
            "evidence": list(evidence.values()),
            "citation_report": citation_report,
            "latency_s": round(elapsed, 4)
        }

    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Agent 执行失败，请检查服务日志"
        )


# =====================================================
# 5. 知识库检索接口
# =====================================================

@app.post("/search")
def search(data: SearchRequest, request: Request):

    try:
        with request.app.state.agent_lock:
            bundle = request.app.state.bundle
            results = bundle.hybrid_rerank_search(
                query=data.query,
                top_k=data.top_k
            )

        output = []

        for item in results:
            metadata = item.get("metadata") or {}

            output.append({
                "point_id": item.get("point_id"),
                "text": item.get("text"),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "heading": metadata.get("heading"),
                "score": item.get("rerank_score")
            })

        return {
            "query": data.query,
            "count": len(output),
            "results": output
        }

    except Exception:
        logger.exception("Search request failed")
        raise HTTPException(
            status_code=500,
            detail="知识库检索失败"
        )


# =====================================================
# 6. 工具列表接口
# =====================================================

@app.get("/tools")
def get_tools(request: Request):

    bundle = request.app.state.bundle

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in bundle.tools
        ]
    }



# ============================================
# 历史会话 API
# ============================================

from uuid import UUID


def validate_thread_id(thread_id: str):
    """
    本地演示阶段使用 UUID 作为会话 ID。
    """
    try:
        return str(UUID(thread_id))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="无效的会话 ID"
        )


@app.get("/sessions")
def list_sessions(request: Request):

    sessions = request.app.state.session_store.list()

    return {
        "count": len(sessions),
        "sessions": sessions
    }


@app.post("/sessions")
def create_session(request: Request):

    session = request.app.state.session_store.create()

    return session


@app.get("/sessions/{thread_id}")
def get_session(thread_id: str, request: Request):

    thread_id = validate_thread_id(thread_id)

    store = request.app.state.session_store

    session = store.get(thread_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    bundle = request.app.state.bundle

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with request.app.state.agent_lock:
        state = bundle.graph.get_state(config)

    all_messages = (
        state.values.get("messages", [])
        if state and state.values
        else []
    )

    # 将 LangGraph 消息转成前端展示消息
    display_messages = []
    current_turn = []

    for msg in all_messages:

        if isinstance(msg, HumanMessage):

            # 遇到下一轮用户消息，结束上一轮收集
            current_turn = [msg]

            display_messages.append({
                "role": "user",
                "content": msg.content
            })

        else:
            current_turn.append(msg)

            # 仅保存最终回答，不显示中间 Tool Call AIMessage
            if (
                isinstance(msg, AIMessage)
                and not msg.tool_calls
                and isinstance(msg.content, str)
                and msg.content.strip()
            ):

                evidence = extract_evidence(
                    current_turn
                )

                trace = extract_agent_trace(
                    current_turn
                )

                display_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "evidence": list(evidence.values()),
                    "trace": trace
                })

    return {
        **session,
        "messages": display_messages
    }


@app.delete("/sessions/{thread_id}")
def delete_session(thread_id: str, request: Request):

    thread_id = validate_thread_id(thread_id)

    store = request.app.state.session_store

    if store.get(thread_id) is None:
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    # 当前使用 SqliteSaver，删除该线程的 Checkpoint 记录
    with request.app.state.agent_lock:

        request.app.state.checkpointer.delete_thread(
            thread_id
        )

        store.delete(thread_id)

    return {
        "success": True,
        "thread_id": thread_id
    }



# ==========================================
# 文档上传与自动入库
# ==========================================

@app.post("/documents/upload")
def upload_document(
    request: Request,
    file: UploadFile = File(...)
):

    MAX_FILE_SIZE = int(__import__("os").getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

    filename = file.filename or ""

    # 限制文件大小，避免一次读取超大文件
    file_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件不能超过 {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空"
        )

    # 使用已有的全局锁，避免上传和聊天同时修改检索状态
    with request.app.state.agent_lock:

        bundle = request.app.state.bundle

        try:
            result = ingest_document(
                filename=filename,
                file_bytes=file_bytes,
                store=bundle.document_store,
                tokenizer=getattr(bundle.embedding_model, "tokenizer", None)
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            )

        except Exception:
            logger.exception("Document ingestion failed")

            raise HTTPException(
                status_code=500,
                detail="文档处理失败，请查看 API 日志"
            )

        # Sync only missing Qdrant vectors, then hot-refresh the in-memory retrieval index.
        # Both new and duplicate uploads retry a previously interrupted sync.
        try:
            stats = bundle.embedding_manager.sync_current(
                bundle.embedding_spec, bundle.embedding_model
            )
            refresh = bundle.refresh_index()
            result.update({
                "index_refreshed": True,
                "index_stats": stats,
                "document_count": refresh["document_count"],
                "message": "文档已入库，向量索引已增量同步" if not result["duplicate"]
                           else "文档已存在，已确认索引同步",
            })
        except Exception:
            logger.exception("Index synchronization failed after SQLite commit")
            # Leave text safely stored in SQLite; retry via /documents/sync.
            result["index_refreshed"] = False
            result["message"] = "文本已保存至 SQLite，但向量索引同步失败；可调用 /documents/sync 重试。"

        return result


@app.post("/documents/sync")
def sync_documents(request: Request):
    """Repair interrupted uploads without re-uploading files or reloading models."""
    with request.app.state.agent_lock:
        bundle = request.app.state.bundle
        try:
            stats = bundle.embedding_manager.sync_current(bundle.embedding_spec, bundle.embedding_model)
            bundle.refresh_index()
            return {"success": True, "stats": stats, "document_count": len(bundle.documents)}
        except Exception:
            logger.exception("Manual index synchronization failed")
            raise HTTPException(status_code=500, detail="索引同步失败，请查看日志")


@app.get("/documents")
def list_documents(request: Request):
    bundle = request.app.state.bundle
    return {"documents": bundle.document_store.list_documents(), "chunks": bundle.document_store.count()}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, request: Request):
    """Delete a document and its vectors; preserves the remaining documents."""
    with request.app.state.agent_lock:
        bundle = request.app.state.bundle
        existing = {row["document_id"] for row in bundle.document_store.list_documents()}
        if document_id not in existing:
            raise HTTPException(status_code=404, detail="文档不存在")
        bundle.document_store.remove_document(document_id)
        try:
            stats = bundle.embedding_manager.sync_current(bundle.embedding_spec, bundle.embedding_model)
            bundle.refresh_index()
            return {"success": True, "stats": stats}
        except Exception:
            logger.exception("Index sync failed after delete")
            raise HTTPException(status_code=503, detail="文档已从 SQLite 删除，但索引尚未同步；请调用 /documents/sync")

# Batch background ingestion endpoints
app.include_router(create_upload_router())
