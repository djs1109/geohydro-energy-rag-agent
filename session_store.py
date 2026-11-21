
# session_store.py

import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class SessionStore:

    def __init__(self, db_path):
        self.db_path = str(Path(db_path))

        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=10
        )
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, thread_id=None, title="新对话"):
        thread_id = thread_id or str(uuid.uuid4())
        timestamp = now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                (thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, title, timestamp, timestamp)
            )

        return self.get(thread_id)

    def get(self, thread_id):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE thread_id = ?",
                (thread_id,)
            ).fetchone()

        return dict(row) if row else None

    def list(self):
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT *
                FROM sessions
                ORDER BY updated_at DESC
            """).fetchall()

        return [dict(row) for row in rows]

    def update_title(self, thread_id, title):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (title, now(), thread_id)
            )

    def touch(self, thread_id):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE thread_id = ?
                """,
                (now(), thread_id)
            )

    def delete(self, thread_id):
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE thread_id = ?",
                (thread_id,)
            )
