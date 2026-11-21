"""Batch upload jobs for a single-process FastAPI server.

Task state is in memory; uploaded temporary files are on disk until processed.
Use one uvicorn worker, persistent volume /app/data, and do not expose publicly
without authentication, file inspection, and rate limits.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from document_ingestion import SUPPORTED_EXTENSIONS, ingest_document, safe_filename


class UploadTaskManager:
    def __init__(self, app):
        self.app = app
        self.lock = Lock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-upload")
        self.root = Path(__file__).resolve().parent / "data" / "upload_tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
        self.max_total = int(os.getenv("MAX_BATCH_MB", "400")) * 1024 * 1024
        self.max_files = int(os.getenv("MAX_BATCH_FILES", "12"))

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "files": [dict(x) for x in job["files"]],
                "done": sum(x["status"] in ("completed", "failed", "duplicate", "stored_only") for x in job["files"]),
                "total": len(job["files"]),
            }

    def update(self, job_id, idx, **changes):
        with self.lock:
            self.jobs[job_id]["files"][idx].update(changes)

    def new_job(self, saved):
        job_id = str(uuid.uuid4())
        files = [{"name": x["name"], "bytes": x["size"], "status": "queued",
                  "stage": "queued", "progress": 0, "detail": "等待处理", "result": None,
                  "error": None} for x in saved]
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "created_at": time.time(), "files": files}
        self.executor.submit(self.process, job_id, saved)
        return job_id

    def process(self, job_id, saved):
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        for i, item in enumerate(saved):
            def report(stage, current, total, detail):
                if stage == "parsing":
                    pct = 10 + round(30 * current / max(1,total))
                elif stage == "chunking":
                    pct = 42
                elif stage == "storing":
                    pct = 48
                elif stage == "embedding":
                    pct = 50 + round(43 * current / max(1,total))
                else:
                    pct = 5
                self.update(job_id, i, stage=stage, progress=min(98,pct), detail=detail)
            try:
                self.update(job_id, i, status="processing", stage="parsing", progress=5, detail="读取文件")
                raw = item["path"].read_bytes()
                bundle = self.app.state.bundle
                # sqlite DocumentStore opens a separate connection per operation.
                result = ingest_document(filename=item["name"], file_bytes=raw,
                                         store=bundle.document_store,
                                         tokenizer=getattr(bundle.embedding_model, "tokenizer", None),
                                         progress=report)
                del raw
                self.update(job_id, i, stage="indexing", progress=50, detail="等待增量向量化")
                try:
                    # Keep chat/search away from mutations while syncing the live index.
                    with self.app.state.agent_lock:
                        bundle = self.app.state.bundle
                        stats = bundle.embedding_manager.sync_current(
                            bundle.embedding_spec, bundle.embedding_model, progress=report
                        )
                        self.update(job_id, i, stage="refreshing", progress=96, detail="刷新 BM25 与 Dense 检索缓存")
                        bundle.refresh_index()
                    result.update(index_refreshed=True, index_stats=stats)
                    status = "duplicate" if result["duplicate"] else "completed"
                    self.update(job_id, i, status=status, stage="done", progress=100,
                                detail="已存在，跳过重复文档" if result["duplicate"] else "已解析、向量化并入库", result=result)
                except Exception as exc:
                    result["index_refreshed"] = False
                    self.update(job_id, i, status="stored_only", stage="sync_failed", progress=50,
                                detail="切片已存入 SQLite，索引同步失败；可用 /documents/sync 恢复",
                                result=result, error=f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                self.update(job_id, i, status="failed", stage="error", detail="处理失败",
                            error=f"{type(exc).__name__}: {exc}")
            finally:
                item["path"].unlink(missing_ok=True)
        with self.lock:
            self.jobs[job_id]["status"] = "finished"


def create_upload_router():
    router = APIRouter()

    @router.post("/documents/upload-batch", status_code=202)
    def upload_batch(request: Request, files: list[UploadFile] = File(...)):
        """Save files in bounded streaming chunks; return a job ID immediately after reception."""
        mgr = request.app.state.upload_tasks
        if not 1 <= len(files) <= mgr.max_files:
            raise HTTPException(400, f"每批文件数量需在 1 到 {mgr.max_files} 之间")
        saved = []
        total = 0
        try:
            for f in files:
                name = safe_filename(f.filename or "")
                if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(400, f"不支持的格式: {name}")
                path = mgr.root / (str(uuid.uuid4()) + Path(name).suffix.lower())
                size = 0
                with path.open("wb") as out:
                    while True:
                        chunk = f.file.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        total += len(chunk)
                        if size > mgr.max_file:
                            raise HTTPException(413, f"{name} 超过 {mgr.max_file // 1048576} MB 的单文件限制")
                        if total > mgr.max_total:
                            raise HTTPException(413, f"本批上传总大小超过 {mgr.max_total // 1048576} MB")
                        out.write(chunk)
                saved.append({"name": name, "path": path, "size": size})
            job_id = mgr.new_job(saved)
            return {"job_id": job_id, "status": "queued", "files": len(saved),
                    "poll_url": f"/documents/jobs/{job_id}"}
        except Exception:
            for item in saved:
                item["path"].unlink(missing_ok=True)
            if 'path' in locals():
                path.unlink(missing_ok=True)
            raise
        finally:
            for f in files:
                f.file.close()

    @router.get("/documents/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        result = request.app.state.upload_tasks.snapshot(job_id)
        if result is None:
            raise HTTPException(404, "任务不存在，或 API 已重启")
        return result

    return router
