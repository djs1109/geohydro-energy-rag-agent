"""Standalone Streamlit page; copy into existing project/pages/.
Works with Streamlit multipage auto-discovery, without replacing the existing app.
"""
from __future__ import annotations

import os
import time
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
STAGE_NAMES = {
    "queued": "排队中", "parsing": "解析文档", "chunking": "生成切片",
    "storing": "保存切片", "indexing": "准备索引", "embedding": "Embedding 向量化",
    "refreshing": "刷新检索索引", "done": "已完成", "error": "处理失败",
    "sync_failed": "索引同步失败",
}

st.set_page_config(page_title="批量文档入库", page_icon="📚", layout="wide")
st.title("📚 批量文档上传与解析进度")
st.caption("支持 PDF / DOCX / Markdown / TXT · 单文件默认 100 MB · 每批最多 12 份 · 任务由后端顺序处理")

files = st.file_uploader("选择需要加入知识库的文件", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
if "rag_upload_job" not in st.session_state:
    st.session_state.rag_upload_job = None

if files:
    st.write(f"已选择 **{len(files)}** 个文件，总大小 **{sum(f.size for f in files)/1048576:.2f} MB**")
    for f in files:
        st.caption(f"{f.name} · {f.size / 1048576:.2f} MB")

if st.button("开始批量解析并入库", type="primary", disabled=not bool(files)):
    if files:
        if len(files) > 12:
            st.error("每批最多 12 份文档，请分批上传。")
        elif any(f.size > 100 * 1048576 for f in files):
            st.error("存在超过 100 MB 的文件，请调整大小或后端 MAX_UPLOAD_MB 配置。")
        elif sum(f.size for f in files) > 400 * 1048576:
            st.error("本批上传总大小不能超过 400 MB。")
        else:
            try:
                # This request transfers files; the backend returns job_id after storing uploads.
                data = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
                with st.spinner("正在将文件传送到 API…"):
                    response = requests.post(f"{API_URL}/documents/upload-batch", files=data, timeout=(15, 600))
                response.raise_for_status()
                st.session_state.rag_upload_job = response.json()["job_id"]
                st.success("批量处理任务已提交。")
            except requests.RequestException as exc:
                details = getattr(getattr(exc, "response", None), "text", "")
                st.error(f"上传失败：{details or str(exc)}")

job_id = st.session_state.rag_upload_job
if job_id:
    st.write(f"任务编号：`{job_id}`")
    refresh = st.button("刷新当前进度")
    auto = st.checkbox("自动刷新进度（每 1.5 秒）", value=True)
    display = st.empty()

    def render(job):
        with display.container():
            done = job["done"]
            total = job["total"]
            st.progress(done / max(total, 1), text=f"整体进度：{done}/{total} 份处理结束")
            for f in job["files"]:
                with st.container(border=True):
                    st.markdown(f"**{f['name']}** · {f['bytes']/1048576:.2f} MB")
                    stage = STAGE_NAMES.get(f["stage"], f["stage"])
                    st.progress(max(0,min(100,int(f["progress"])))/100,
                                text=f"{stage} · {f['detail']}")
                    if f["status"] in ("failed", "stored_only"):
                        st.error(f.get("error") or f["detail"])
                    elif f["status"] == "duplicate":
                        st.info("知识库中已存在该文件")
                    elif f["status"] == "completed":
                        result = f.get("result") or {}
                        st.success(f"完成：{result.get('chunk_count', '?')} 个切片，索引已刷新")

    # UI polling happens after the POST has returned. The job itself runs on API thread.
    while True:
        try:
            result = requests.get(f"{API_URL}/documents/jobs/{job_id}", timeout=15)
            result.raise_for_status()
            job = result.json()
            render(job)
        except requests.RequestException as exc:
            st.warning(f"进度查询失败：{exc}。可稍后手动刷新。")
            break
        if job["status"] == "finished":
            st.success("本批任务处理结束。若部分文件入库失败，请查看其错误信息。")
            break
        if not auto:
            break
        time.sleep(1.5)

st.divider()
st.caption("提示：任务进度存放于 API 进程内存，重启容器后历史进度会消失，但已成功写入 SQLite/Qdrant 的数据仍保留。")
