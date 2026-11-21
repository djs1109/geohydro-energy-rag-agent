
# streamlit_app.py

import os
import uuid
import json
import requests
import streamlit as st

API_URL = os.getenv(
    "AGENT_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="专业领域 RAG Agent",
    page_icon="🌍",
    layout="wide"
)



# ==========================================
#  API 请求函数
# ==========================================

def api_get(path):
    response = requests.get(
        f"{API_URL}{path}",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=180
    )
    response.raise_for_status()
    return response.json()

def api_delete(path):
    response = requests.delete(
        f"{API_URL}{path}",
        timeout=15
    )
    response.raise_for_status()
    return response.json()
# =====================================================
# 初始化会话管理
# =====================================================

# ==========================================
# 从后端恢复会话
# ==========================================

if "conversations" not in st.session_state:

    st.session_state.conversations = {}

    try:
        result = api_get("/sessions")

        for session in result["sessions"]:

            thread_id = session["thread_id"]

            st.session_state.conversations[thread_id] = {
                "title": session["title"],
                "messages": []
            }

    except requests.RequestException:
        st.warning("历史会话加载失败，请检查后端服务。")


if "thread_id" not in st.session_state:

    if st.session_state.conversations:

        st.session_state.thread_id = next(
            iter(st.session_state.conversations)
        )

    else:
        try:
            session = api_post("/sessions", {})

            thread_id = session["thread_id"]

            st.session_state.thread_id = thread_id

            st.session_state.conversations[thread_id] = {
                "title": session["title"],
                "messages": []
            }

        except requests.RequestException:
            st.error("无法创建会话，请先启动 FastAPI。")
            st.stop()


def load_conversation(thread_id):

    response = api_get(
        f"/sessions/{thread_id}"
    )

    st.session_state.conversations[thread_id] = {
        "title": response["title"],
        "messages": response["messages"]
    }

    return response


# 兼容当前会话
if st.session_state.thread_id not in st.session_state.conversations:

    st.session_state.conversations[
        st.session_state.thread_id
    ] = {
        "title": "新对话",
        "messages": []
    }

# =====================================================
# 自动恢复当前会话的历史记录【新增】
# =====================================================
if "loaded_thread_id" not in st.session_state:
    st.session_state.loaded_thread_id = None

if st.session_state.loaded_thread_id != st.session_state.thread_id:

    try:
        load_conversation(st.session_state.thread_id)
        st.session_state.loaded_thread_id = st.session_state.thread_id

    except requests.RequestException as exc:
        st.error(f"恢复历史记录失败：{exc}")

# ==========================================
# 3. 侧边栏
# ==========================================

with st.sidebar:

    st.title("🌍 RAG Agent")
    st.caption("地质灾害 · 水文气象 · 新能源")

    st.divider()

    st.subheader("系统状态")

    try:
        health = api_get("/health")

        st.success("后端服务正常")
        st.write(
            "知识库文档：",
            health["document_count"]
        )
        st.write(
            "模型：",
            health["model"]
        )

    except requests.RequestException:
        st.error("无法连接 FastAPI 后端")
        st.caption(f"地址：{API_URL}")

    st.divider()

    # st.subheader("会话管理")

    # if st.button(
    #     "➕ 新建对话",
    #     use_container_width=True
    # ):
    #     st.session_state.thread_id = str(uuid.uuid4())
    #     st.session_state.messages = []
    #     st.rerun()

    # st.caption(
    #     f"会话 ID：{st.session_state.thread_id[:8]}"
    # )


    # =====================================================
    # 左侧会话管理
    # =====================================================

    st.subheader("会话管理")

    # 新建对话
    if st.button("➕ 新建对话", use_container_width=True):

        try:
            session = api_post("/sessions", {})

            thread_id = session["thread_id"]

            st.session_state.conversations[thread_id] = {
                "title": "新对话",
                "messages": []
            }

            st.session_state.thread_id = thread_id

            st.rerun()

        except requests.RequestException as exc:
            st.error(f"创建失败：{exc}")



    st.markdown("#### 历史会话")

    for thread_id, conversation in reversed(
        list(st.session_state.conversations.items())
    ):

        col1, col2 = st.columns([5, 1])

        with col1:

            if st.button(
                conversation["title"],
                key=f"switch_{thread_id}",
                use_container_width=True
            ):

                try:
                    load_conversation(thread_id)
                    st.session_state.thread_id = thread_id
                    st.session_state.loaded_thread_id = thread_id
                    st.rerun()

                except requests.RequestException as exc:
                    st.error(f"加载会话失败：{exc}")

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{thread_id}"
            ):

                try:
                    api_delete(f"/sessions/{thread_id}")

                    del st.session_state.conversations[thread_id]

                    if (
                        st.session_state.thread_id == thread_id
                    ):

                        if st.session_state.conversations:
                            st.session_state.thread_id = next(
                                iter(st.session_state.conversations)
                            )
                        else:
                            new_session = api_post("/sessions", {})
                            new_id = new_session["thread_id"]

                            st.session_state.conversations[new_id] = {
                                "title": "新对话",
                                "messages": []
                            }

                            st.session_state.thread_id = new_id

                    st.rerun()

                except requests.RequestException as exc:
                    st.error(f"删除失败：{exc}")



    st.caption(
        f"当前会话：{st.session_state.thread_id[:8]}"
    )


    st.divider()

    st.subheader("当前功能")
    st.write("✓ 专业知识库问答")
    st.write("✓ 混合检索与重排序")
    st.write("✓ 降雨统计与计算")
    st.write("✓ 多轮对话")
    st.write("✓ 文献来源展示")


# ==========================================
# 4. 页面标题与标签页
# ==========================================

st.title("专业领域智能分析 Agent")

st.caption(
    "LangGraph + DeepSeek + Qdrant + "
    "Hybrid Retrieval + Reranker"
)

chat_tab, search_tab,upload_tab, tools_tab = st.tabs([
    "智能对话",
    "知识库检索",
    "知识库管理",
    "系统工具"
])


# ==========================================
# 5. 智能对话
# ==========================================

with chat_tab:
    current_conversation = st.session_state.conversations[
        st.session_state.thread_id
    ]

    messages = current_conversation["messages"]

    if not messages:
        st.info(
            "你可以询问滑坡形成机制、"
            "降雨影响以及相关专业知识，"
            "也可以让 Agent 完成降雨统计计算。"
        )
    # current_conversation = st.session_state.conversations[
    # st.session_state.thread_id
    # ]

    messages = current_conversation["messages"]

    # 展示历史聊天
    for message in messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("trace"):

                with st.expander("查看 Agent 执行过程"):

                    for step in message["trace"]:

                        st.markdown(f"**{step['tool']}**")
                        st.json(step["arguments"])
                        st.code(str(step["result"]))

            if message.get("evidence"):

                with st.expander(
                    f"查看检索依据（{len(message['evidence'])}条）"
                ):

                    for doc in message["evidence"]:

                        st.markdown(
                            f"**[{doc.get('evidence_id', '证据')}]** "
                            f"{doc.get('source', '未知')} "
                            f"— 第 {doc.get('page', '未知')} 页"
                        )

                        st.write(doc.get("text", ""))

                        st.divider()

    # 用户输入
    prompt = st.chat_input(
        "输入你的问题，例如：地下水位升高为何影响边坡稳定性？",
        key="chat_prompt"
    )

    if prompt:

        messages.append({
            "role": "user",
            "content": prompt
        })
        
        if current_conversation["title"] == "新对话":

            current_conversation["title"] = (
                prompt[:18] + "..."
                if len(prompt) > 18
                else prompt
            )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Agent 正在分析..."):

                try:
                    response = api_post(
                        "/chat",
                        {
                            "query": prompt,
                            "thread_id": st.session_state.thread_id
                        }
                    )

                    answer = response["answer"]
                    evidence = response.get("evidence", [])
                    trace = response.get("trace", [])
                    report = response.get(
                        "citation_report", {}
                    )

                    st.markdown(answer)

                    st.caption(
                        f"本次响应耗时："
                        f"{response.get('latency_s', 0):.2f} 秒"
                    )

                    # ==========================================
                    # 展示 Agent 工具执行轨迹
                    # ==========================================

                    if trace:

                        with st.expander(
                            f"查看 Agent 执行过程（{len(trace)}次工具调用）"
                        ):

                            for idx, step in enumerate(trace, 1):

                                st.markdown(
                                    f"### Step {idx}: {step['tool']}"
                                )

                                st.caption(
                                    f"执行状态：{step['status']}"
                                )

                                st.markdown("**工具输入参数**")

                                st.json(
                                    step["arguments"]
                                )

                                st.markdown("**工具执行结果**")

                                result_text = step.get("result")

                                try:
                                    st.json(json.loads(result_text))
                                except (json.JSONDecodeError, TypeError):
                                    st.code(str(result_text))

                                st.divider()


                    if evidence:

                        with st.expander(
                            f"查看检索依据（{len(evidence)}条）"
                        ):

                            for doc in evidence:

                                st.markdown(
                                    f"**[{doc.get('evidence_id', '证据')}]** "
                                    f"{doc.get('source', '未知')} "
                                    f"— 第 {doc.get('page', '未知')} 页"
                                )

                                st.write(doc.get("text", ""))

                                st.divider()

                    if report.get("invalid_citations"):
                        st.warning(
                            "部分引用编号未能通过校验："
                            + str(report["invalid_citations"])
                        )

                    messages.append({
                        "role": "assistant",
                        "content": answer,
                        "evidence": evidence,
                        "trace": trace
                    })
                    # =====================================================
                    # 从后端同步最新会话标题【新增】
                    # =====================================================

                    try:
                        session = api_get(
                            f"/sessions/{st.session_state.thread_id}"
                        )

                        st.session_state.conversations[
                            st.session_state.thread_id
                        ]["title"] = session["title"]

                    except requests.RequestException as exc:
                        st.warning(f"会话标题同步失败：{exc}")

                except requests.RequestException as exc:
                    st.error("请求失败，请检查 FastAPI 服务。")
                    st.caption(str(exc))


# ==========================================
# 6. 知识库独立检索
# ==========================================

with search_tab:

    st.subheader("知识库混合检索")

    st.caption(
        "BM25 + Dense + RRF + BGE Reranker"
    )

    search_query = st.text_input(
        "检索问题",
        placeholder="例如：降雨入渗如何影响边坡稳定性？"
    )

    top_k = st.slider(
        "返回文档数量",
        min_value=1,
        max_value=10,
        value=4
    )

    if st.button(
        "开始检索",
        type="primary",
        disabled=not search_query.strip()
    ):

        try:
            response = api_post(
                "/search",
                {
                    "query": search_query,
                    "top_k": top_k
                }
            )

            st.success(
                f"找到 {response['count']} 条候选结果"
            )

            for idx, item in enumerate(
                response["results"], 1
            ):

                with st.container(border=True):

                    st.markdown(f"**Top {idx}**")

                    st.write(item.get("text", ""))

                    st.caption(
                        f"来源：{item.get('source', '未知')} | "
                        f"页码：{item.get('page', '未知')} | "
                        f"Point ID：{item.get('point_id')}"
                    )

                    score = item.get("score")

                    if score is not None:
                        st.caption(
                            f"Reranker Score：{score:.4f}"
                        )

        except requests.RequestException as exc:
            st.error(f"检索失败：{exc}")


# ==========================================
# 知识库文件上传
# ==========================================

with upload_tab:

    st.subheader("知识库文档上传")

    st.caption(
        "支持 PDF、DOCX、TXT、Markdown。"
        "文件上传后自动解析、切片、向量化并写入 Qdrant。"
    )

    uploaded_file = st.file_uploader(
        "选择需要加入知识库的文件",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=False,
        key="knowledge_file_upload"
    )

    if uploaded_file is not None:

        st.write("文件名：", uploaded_file.name)

        st.write(
            "文件大小：",
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        if st.button(
            "开始解析并入库",
            type="primary",
            key="upload_to_qdrant"
        ):

            with st.spinner(
                "正在解析文档、生成向量并更新检索索引..."
            ):

                try:
                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                                or "application/octet-stream"
                            )
                        },
                        timeout=600
                    )

                    response.raise_for_status()

                    result = response.json()

                    if result.get("duplicate"):

                        st.info(
                            "该文件内容已经存在于知识库中，"
                            "本次没有重复插入。"
                        )

                    else:

                        st.success("文档入库完成")

                        st.write(
                            "新增切片数量：",
                            result["chunk_count"]
                        )

                        st.write(
                            "原始文件：",
                            result["filename"]
                        )

                        if result.get("index_refreshed"):

                            st.success(
                                "检索索引已刷新，"
                                "现在可以使用新文档进行问答。"
                            )

                        else:

                            st.warning(
                                "Qdrant 已写入新文档，"
                                "但 Agent 内存索引未刷新。"
                                "请重启 API 容器。"
                            )

                except requests.RequestException as exc:

                    st.error("文档上传或处理失败")

                    st.caption(str(exc))

# ==========================================
# 7. 工具列表
# ==========================================

with tools_tab:

    st.subheader("Agent 可用工具")

    if st.button("刷新工具列表"):

        try:
            response = api_get("/tools")

            for item in response["tools"]:

                with st.container(border=True):

                    st.markdown(f"**{item['name']}**")

                    st.write(item["description"])

        except requests.RequestException as exc:
            st.error(f"获取工具列表失败：{exc}")
