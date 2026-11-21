
# tests/test_api_smoke.py

import os
import uuid

import pytest
import requests

from dotenv import load_dotenv

# 加载项目根目录下的 .env
load_dotenv()

BASE_URL = os.getenv(
    "AGENT_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

RUN_LLM_TEST = os.getenv("RUN_LLM_TEST") == "1"


# ========================================
# 1. HTTP 请求工具
# ========================================

def api_get(path, timeout=30):
    response = requests.get(
        BASE_URL + path,
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def api_post(path, payload, timeout=30):
    response = requests.post(
        BASE_URL + path,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


# ========================================
# 2. 会话 Fixture
# ========================================

@pytest.fixture
def test_session():
    """
    为每个测试创建独立会话。
    测试结束后清理对应会话。
    """
    session = api_post("/sessions", {})
    thread_id = session["thread_id"]

    try:
        yield thread_id
    finally:
        response = requests.delete(
            f"{BASE_URL}/sessions/{thread_id}",
            timeout=30
        )
        response.raise_for_status()


# ========================================
# 3. 健康检查
# ========================================

def test_health():
    result = api_get("/health")

    assert result["status"] == "ok"
    assert result["document_count"] > 0
    assert result["collection"] == "rag_active"


# ========================================
# 4. 工具注册检查
# ========================================

def test_tools():
    result = api_get("/tools")

    tools = {
        item["name"]
        for item in result["tools"]
    }

    expected = {
        "search_knowledge_base",
        "calculate_rainfall_statistics",
        "calculate_effective_rainfall"
    }

    assert expected.issubset(tools)


# ========================================
# 5. RAG 检索检查
# ========================================

def test_search():
    result = api_post(
        "/search",
        {
            "query": "降雨入渗为什么会影响滑坡？",
            "top_k": 4
        },
        timeout=120
    )

    assert result["count"] > 0
    assert len(result["results"]) == result["count"]

    for doc in result["results"]:
        assert doc["text"]
        assert doc["source"]
        assert doc["point_id"] is not None


# ========================================
# 6. 会话创建与读取
# ========================================

def test_session_creation(test_session):
    thread_id = test_session

    result = api_get(
        f"/sessions/{thread_id}"
    )

    assert result["thread_id"] == thread_id
    assert result["title"] == "新对话"
    assert result["messages"] == []


# ========================================
# 7. 会话列表
# ========================================

def test_session_listing(test_session):
    result = api_get("/sessions")

    ids = {
        item["thread_id"]
        for item in result["sessions"]
    }

    assert test_session in ids


# ========================================
# 8. 不存在的会话
# ========================================

def test_missing_session():
    unknown_id = str(uuid.uuid4())

    response = requests.get(
        f"{BASE_URL}/sessions/{unknown_id}",
        timeout=30
    )

    assert response.status_code == 404


# ========================================
# 9. Agent 实际问答
# 需要 DeepSeek API，默认跳过
# ========================================

@pytest.mark.skipif(
    not RUN_LLM_TEST,
    reason="设置 RUN_LLM_TEST=1 才执行真实 LLM 测试"
)
def test_agent_chat(test_session):

    result = api_post(
        "/chat",
        {
            "query": (
                "根据知识库，解释地下水位升高"
                "对边坡稳定性的影响。"
            ),
            "thread_id": test_session
        },
        timeout=180
    )

    assert result["answer"].strip()
    assert result["thread_id"] == test_session
    assert "trace" in result
    assert "citation_report" in result

    # 检查回答被持久化
    session = api_get(
        f"/sessions/{test_session}"
    )

    assert len(session["messages"]) >= 2

    assert any(
        m["role"] == "assistant"
        for m in session["messages"]
    )




@pytest.mark.skipif(
    not RUN_LLM_TEST,
    reason="需要真实 LLM 调用"
)
def test_session_isolation():

    session_a = api_post("/sessions", {})["thread_id"]
    session_b = api_post("/sessions", {})["thread_id"]

    try:
        api_post(
            "/chat",
            {
                "thread_id": session_a,
                "query": "请记住本次测试代号为 ALPHA-731。"
            },
            timeout=180
        )

        state_a = api_get(f"/sessions/{session_a}")
        state_b = api_get(f"/sessions/{session_b}")

        # 会话 A 应该包含它自己的消息
        assert len(state_a["messages"]) >= 2

        # 会话 B 尚未进行任何对话
        assert state_b["messages"] == []

        # 两个会话 ID 不相同
        assert session_a != session_b

    finally:
        for thread_id in [session_a, session_b]:
            requests.delete(
                f"{BASE_URL}/sessions/{thread_id}",
                timeout=30
            ).raise_for_status()
