"""Persistent SQLite-backed session store for multi-turn student chat memory."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional
import uuid


class SessionStore:
    """Manages chat sessions and conversation turns using local SQLite persistence."""

    def __init__(self, db_path: str = "./data/chat_sessions.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create sessions and messages tables with indexes."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sources TEXT,
                    references_data TEXT,
                    grounding TEXT,
                    verification TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at DESC);
                """
            )
            # Databases created before visitor workspaces existed lack the owner column.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "workspace" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN workspace TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions (workspace)")

    @staticmethod
    def _scope(workspace: Optional[str], alias: str = "") -> tuple[str, list[Any]]:
        """SQL condition limiting sessions to one workspace; None means no limit (single-user mode)."""
        if workspace is None:
            return "", []
        return f" AND {alias}workspace = ?", [workspace]

    def create_session(
        self, session_id: Optional[str] = None, title: str = "New Chat", workspace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new chat session."""
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, workspace) VALUES (?, ?, ?, ?, ?)",
                (sid, title, now, now, workspace),
            )
        return {
            "id": sid,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def list_sessions(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List chat sessions (of one workspace, when given) ordered by most recently updated."""
        condition, params = self._scope(workspace, "s.")
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE 1 = 1{condition}
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """,
                params,
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"],
                }
                for row in rows
            ]

    def get_session(self, session_id: str, workspace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch session metadata by ID (None when it belongs to another workspace)."""
        condition, params = self._scope(workspace, "s.")
        with self._connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.id = ?{condition}
                GROUP BY s.id
                """,
                [session_id, *params],
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": row["message_count"],
            }

    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch message history for a session in chronological order."""
        query = """
            SELECT id, session_id, sender, text, sources, references_data,
                   grounding, verification, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        params: list[Any] = [session_id]
        if limit is not None and limit > 0:
            # When limit is given, fetch the latest 'limit' messages in chronological order
            query = f"""
                SELECT * FROM (
                    SELECT id, session_id, sender, text, sources, references_data,
                           grounding, verification, timestamp
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT {int(limit)}
                ) ORDER BY timestamp ASC
            """
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            messages: List[Dict[str, Any]] = []
            for row in rows:
                sources = json.loads(row["sources"]) if row["sources"] else []
                refs = json.loads(row["references_data"]) if row["references_data"] else []
                verif = json.loads(row["verification"]) if row["verification"] else None
                messages.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "sender": row["sender"],
                        "role": "assistant" if row["sender"] in ("tutor", "assistant") else "user",
                        "text": row["text"],
                        "content": row["text"],
                        "sources": sources,
                        "references": refs,
                        "grounding": row["grounding"] or "",
                        "verification": verif,
                        "timestamp": row["timestamp"],
                        "created_at": row["timestamp"],
                    }
                )
            return messages

    def add_message(
        self,
        session_id: str,
        sender: str,
        text: str,
        sources: Optional[List[str]] = None,
        references: Optional[List[Dict[str, Any]]] = None,
        grounding: str = "",
        verification: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        auto_title: bool = True,
        workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a message to a session and touch updated_at."""
        mid = message_id or f"msg_{uuid.uuid4().hex[:12]}"
        now = time.time()
        sources_json = json.dumps(sources or []) if sources else ""
        refs_json = json.dumps(references or []) if references else ""
        verif_json = json.dumps(verification) if verification else ""

        with self._connection() as conn:
            # Ensure session exists; if not, create it
            session = self.get_session(session_id)
            if not session:
                new_title = text.strip()[:40] if sender == "user" and text else "New Chat"
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at, workspace) VALUES (?, ?, ?, ?, ?)",
                    (session_id, new_title, now, now, workspace),
                )
            else:
                # Auto-generate title on first user message if current title is default
                if auto_title and sender == "user" and session["title"] == "New Chat":
                    snippet = text.strip().replace("\n", " ")
                    new_title = snippet[:38] + "..." if len(snippet) > 38 else snippet
                    if new_title:
                        conn.execute(
                            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                            (new_title, now, session_id),
                        )
                    else:
                        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
                else:
                    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, sender, text, sources, references_data,
                    grounding, verification, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    session_id,
                    sender,
                    text,
                    sources_json,
                    refs_json,
                    grounding,
                    verif_json,
                    now,
                ),
            )

        return {
            "id": mid,
            "session_id": session_id,
            "sender": sender,
            "text": text,
            "sources": sources or [],
            "references": references or [],
            "grounding": grounding,
            "verification": verification,
            "timestamp": now,
        }

    def rename_session(self, session_id: str, title: str, workspace: Optional[str] = None) -> bool:
        """Rename an existing session."""
        now = time.time()
        condition, params = self._scope(workspace)
        with self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?{condition}",
                [title.strip(), now, session_id, *params],
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str, workspace: Optional[str] = None) -> bool:
        """Delete a session and all its messages."""
        condition, params = self._scope(workspace)
        with self._connection() as conn:
            cursor = conn.execute(f"DELETE FROM sessions WHERE id = ?{condition}", [session_id, *params])
            return cursor.rowcount > 0

    def session_owner(self, session_id: str) -> tuple[bool, Optional[str]]:
        """(exists, workspace) for a session ID, regardless of who asks."""
        with self._connection() as conn:
            row = conn.execute("SELECT workspace FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return (row is not None, row["workspace"] if row else None)

    def get_recent_history(self, session_id: str, turns: int = 4) -> List[Dict[str, str]]:
        """Return the last `turns` user-assistant message pairs formatted for LLM consumption.

        Returns:
            List of dicts: [{"role": "user"|"assistant", "content": text}, ...]
        """
        # turns * 2 gives total messages (e.g., 4 turns = 8 messages)
        limit = max(1, turns * 2)
        raw_messages = self.get_session_messages(session_id, limit=limit)
        history: List[Dict[str, str]] = []
        for msg in raw_messages:
            role = "user" if msg["sender"] == "user" else "assistant"
            history.append({"role": role, "content": msg["text"]})
        return history

    def clear_all(self) -> None:
        """Clear all sessions and messages (for testing)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM messages;")
            conn.execute("DELETE FROM sessions;")


_global_session_store: Optional[SessionStore] = None


def get_session_store(db_path: str = "./data/chat_sessions.db") -> SessionStore:
    """Return the shared SessionStore singleton."""
    global _global_session_store
    if _global_session_store is None:
        _global_session_store = SessionStore(db_path=db_path)
    return _global_session_store
