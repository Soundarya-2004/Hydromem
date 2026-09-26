"""Persistence layer for HydroMem.

Provides LocalStore supporting two persistence backends:
1. JSON (default, simple file storage)
2. SQLite (with full-text search FTS5 index and BM25 ranking)
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union


class LocalStore:
    """Manages serialization and persistence of HydroMem states to disk."""

    def __init__(self, path: Optional[str] = None, mode: Optional[str] = None) -> None:
        """Initialize LocalStore.

        Args:
            path: Target file path (e.g. 'mem.json' or 'mem.db').
            mode: Storage backend mode: 'json' or 'sqlite'. Inferred from path extension if None.
        """
        self.path = path
        if mode:
            self.mode = mode.lower()
        elif path and (path.endswith(".db") or path.endswith(".sqlite") or path.endswith(".sqlite3")):
            self.mode = "sqlite"
        else:
            self.mode = "json"

    def save(self, path: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> str:
        """Save HydroMem state dictionary to disk.

        Args:
            path: Target storage path. Defaults to self.path.
            state: Dictionary containing sensory, short, long memories, id_counter, recall_tracker.

        Returns:
            str: Path where data was persisted.
        """
        target_path = path or self.path
        if not target_path:
            raise ValueError("No persistence path specified for LocalStore.save()")
        if state is None:
            state = {}

        if self.mode == "sqlite":
            self._save_sqlite(target_path, state)
        else:
            self._save_json(target_path, state)

        return target_path

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load HydroMem state dictionary from disk.

        Args:
            path: Target storage path. Defaults to self.path.

        Returns:
            Dict[str, Any]: Reconstituted state with id_counter, recall_tracker, and tank memories.
        """
        target_path = path or self.path
        if not target_path or not os.path.exists(target_path):
            return {
                "id_counter": 0,
                "recall_tracker": {},
                "sensory": [],
                "short": [],
                "long": [],
            }

        if self.mode == "sqlite":
            return self._load_sqlite(target_path)
        return self._load_json(target_path)

    # ---------------- JSON Backend ----------------

    def _save_json(self, path: str, state: Dict[str, Any]) -> None:
        """Persist state as a formatted JSON document."""
        temp_path = f"{path}.tmp"
        payload = {
            "version": "0.0.1",
            "id_counter": state.get("id_counter", 0),
            "recall_tracker": {str(k): v for k, v in state.get("recall_tracker", {}).items()},
            "sensory": state.get("sensory", []),
            "short": state.get("short", []),
            "long": state.get("long", []),
        }
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if os.path.exists(path):
            os.replace(temp_path, path)
        else:
            os.rename(temp_path, path)

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load state from a JSON document."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "id_counter": int(payload.get("id_counter", 0)),
                "recall_tracker": {int(k): v for k, v in payload.get("recall_tracker", {}).items()},
                "sensory": payload.get("sensory", []),
                "short": payload.get("short", []),
                "long": payload.get("long", []),
            }
        except Exception:
            return {
                "id_counter": 0,
                "recall_tracker": {},
                "sensory": [],
                "short": [],
                "long": [],
            }

    # ---------------- SQLite + FTS5 Backend ----------------

    def _init_sqlite_db(self, conn: sqlite3.Connection) -> None:
        """Initialize SQLite tables and FTS5 full-text search virtual table."""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                tank TEXT,
                text TEXT,
                pressure REAL,
                timestamp REAL,
                created_at REAL,
                recall_count INTEGER DEFAULT 0,
                embedding TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                tank UNINDEXED,
                text
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.commit()

    def _save_sqlite(self, path: str, state: Dict[str, Any]) -> None:
        """Save state to SQLite and update the FTS5 search index."""
        conn = sqlite3.connect(path)
        try:
            self._init_sqlite_db(conn)
            cursor = conn.cursor()

            # Clear existing memories to mirror complete state snapshot
            cursor.execute("DELETE FROM memories")
            cursor.execute("DELETE FROM memories_fts")

            all_items: List[tuple] = []
            fts_items: List[tuple] = []

            for tank_name in ("sensory", "short", "long"):
                for m in state.get(tank_name, []):
                    m_id = int(m.get("id", 0))
                    text = str(m.get("text", ""))
                    pressure = float(m.get("pressure", 0.0))
                    timestamp = float(m.get("timestamp", 0.0))
                    created_at = float(m.get("created_at", timestamp))
                    recall_count = int(state.get("recall_tracker", {}).get(m_id, 0))
                    emb_str = json.dumps(m.get("embedding")) if m.get("embedding") else None

                    all_items.append((m_id, tank_name, text, pressure, timestamp, created_at, recall_count, emb_str))
                    # For long tank, text is encrypted ciphertext in memories; we index decrypted text if available
                    index_text = m.get("decrypted_text", text) if tank_name == "long" else text
                    fts_items.append((m_id, tank_name, index_text))

            cursor.executemany(
                """
                INSERT OR REPLACE INTO memories (id, tank, text, pressure, timestamp, created_at, recall_count, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                all_items,
            )
            cursor.executemany(
                """
                INSERT INTO memories_fts (id, tank, text)
                VALUES (?, ?, ?)
                """,
                fts_items,
            )

            # Metadata (id_counter, recall_tracker)
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('id_counter', ?)", (str(state.get("id_counter", 0)),))
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('recall_tracker', ?)", (json.dumps(state.get("recall_tracker", {})),))
            conn.commit()
        finally:
            conn.close()

    def _load_sqlite(self, path: str) -> Dict[str, Any]:
        """Load state from SQLite database."""
        conn = sqlite3.connect(path)
        try:
            self._init_sqlite_db(conn)
            cursor = conn.cursor()

            cursor.execute("SELECT key, value FROM metadata")
            meta = dict(cursor.fetchall())
            id_counter = int(meta.get("id_counter", 0))
            recall_tracker = {int(k): v for k, v in json.loads(meta.get("recall_tracker", "{}")).items()}

            cursor.execute("SELECT id, tank, text, pressure, timestamp, created_at, embedding FROM memories")
            rows = cursor.fetchall()

            sensory: List[Dict[str, Any]] = []
            short: List[Dict[str, Any]] = []
            long: List[Dict[str, Any]] = []

            for row in rows:
                m_id, tank, text, pressure, timestamp, created_at, emb_json = row
                item = {
                    "id": m_id,
                    "text": text,
                    "pressure": pressure,
                    "timestamp": timestamp,
                    "created_at": created_at,
                }
                if emb_json:
                    try:
                        item["embedding"] = json.loads(emb_json)
                    except Exception:
                        pass

                if tank == "sensory":
                    sensory.append(item)
                elif tank == "short":
                    short.append(item)
                elif tank == "long":
                    long.append(item)

            return {
                "id_counter": id_counter,
                "recall_tracker": recall_tracker,
                "sensory": sensory,
                "short": short,
                "long": long,
            }
        finally:
            conn.close()

    def search_fts5(
        self,
        query: str,
        path: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform O(log N) BM25 keyword search using the SQLite FTS5 virtual table.

        Args:
            query: Keyword search string.
            path: Target SQLite database path.
            top_k: Maximum results to retrieve.

        Returns:
            List[Dict[str, Any]]: Matching records ordered by BM25 relevance score.
        """
        target_path = path or self.path
        if not target_path or not os.path.exists(target_path):
            return []

        # Sanitize query for FTS5 (split words and quote tokens to prevent syntax errors)
        tokens = [t.strip().replace('"', "") for t in query.split() if t.strip()]
        if not tokens:
            return []
        fts_match_expr = " OR ".join(f'"{t}"' for t in tokens)

        conn = sqlite3.connect(target_path)
        try:
            cursor = conn.cursor()
            query_sql = """
                SELECT m.id, m.tank, m.text, m.pressure, m.timestamp, m.created_at, m.recall_count, bm25(memories_fts) as rank
                FROM memories_fts f
                JOIN memories m ON m.id = f.id
                WHERE memories_fts MATCH ?
                ORDER BY rank ASC
                LIMIT ?
            """
            cursor.execute(query_sql, (fts_match_expr, top_k))
            results = []
            for row in cursor.fetchall():
                m_id, tank, text, pressure, timestamp, created_at, recall_count, bm25_rank = row
                results.append(
                    {
                        "id": m_id,
                        "tank": tank,
                        "text": text,
                        "pressure": pressure,
                        "timestamp": timestamp,
                        "created_at": created_at,
                        "recall_count": recall_count,
                        "bm25_rank": bm25_rank,
                    }
                )
            return results
        except Exception:
            return []
        finally:
            conn.close()
