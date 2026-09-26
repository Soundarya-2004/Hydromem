"""Core implementation of HydroMem v0.2.0 - Hydraulic-Inspired Hierarchical Memory for LLMs.

Routes memories dynamically across three fluid chambers based on hydraulic pressure:
- SensoryTank (< 0.4 pressure): short-lived transient context
- ShortTank (0.4 - 0.8 pressure): active working memory
- LongTank (> 0.8 pressure): permanent, AES-encrypted memory

v0.2.0 Features:
- LocalStore disk persistence (JSON and SQLite with FTS5 BM25)
- Dynamic recency calculations with composite relevance ranking
- Optional FastEmbed vector similarity search
- Background auto-evaporation daemon thread
- Secure KDF with Argon2id / PBKDF2-HMAC salt derivation
"""

import threading
import time
from typing import Any, Dict, List, Optional, Union
from .persistence import LocalStore
from .search import (
    FASTEMBED_AVAILABLE,
    EmbeddingEngine,
    cosine_similarity,
    keyword_match_score,
)
from .security import generate_key
from .tank import LongTank, SensoryTank, ShortTank
from .utils import (
    calculate_pressure,
    compute_combined_score,
    compute_dynamic_recency,
    is_sedimented,
)


class HydroMem:
    """Hydraulic-Inspired Hierarchical Memory Manager for LLMs with local-first privacy."""

    def __init__(
        self,
        encryption_password: str = "hydromem",
        save_path: Optional[str] = None,
        store_mode: Optional[str] = None,
        embedding_model: Optional[str] = None,
        auto_evaporation: bool = False,
        evaporation_interval: float = 3600.0,
        salt: Optional[bytes] = None,
        salt_path: Optional[str] = None,
        use_argon2: bool = True,
    ) -> None:
        """Initialize HydroMem with hierarchical tanks, encryption key, and optional v0.2 extensions.

        Args:
            encryption_password: Password string used to derive the local Fernet key.
            save_path: Optional disk persistence path ('mem.json' or 'mem.db').
            store_mode: Storage backend mode ('json' or 'sqlite'). Inferred if None.
            embedding_model: Optional FastEmbed model name (e.g. 'BAAI/bge-small-en').
            auto_evaporation: If True, spawns a daemon thread for periodic evaporation.
            evaporation_interval: Interval in seconds between auto-evaporation cycles.
            salt: Optional 16-byte cryptographic salt.
            salt_path: Optional path to persist or load the salt.
            use_argon2: Whether to prefer Argon2id KDF if available.
        """
        self.key = generate_key(
            password=encryption_password,
            salt=salt,
            salt_path=salt_path,
            use_argon2=use_argon2,
        )
        self.sensory_tank = SensoryTank()
        self.short_tank = ShortTank()
        self.long_tank = LongTank(key=self.key)
        self.id_counter: int = 0
        self.recall_tracker: Dict[int, int] = {}

        # 1. Search & Embeddings Engine
        self.embedding_model_name = embedding_model
        self.embedding_engine: Optional[EmbeddingEngine] = None
        if embedding_model:
            self.embedding_engine = EmbeddingEngine(model_name=embedding_model)

        # 2. Persistence Layer
        self.save_path = save_path
        self._local_store: Optional[LocalStore] = None
        if save_path:
            self._local_store = LocalStore(path=save_path, mode=store_mode)
            self.load()

        # 3. Auto-evaporation Daemon Thread
        self.auto_evaporation = auto_evaporation
        self.evaporation_interval = evaporation_interval
        self._stop_evaporation_event = threading.Event()
        self._evaporation_thread: Optional[threading.Thread] = None

        if self.auto_evaporation:
            self._start_auto_evaporation()

    def _start_auto_evaporation(self) -> None:
        """Start background daemon thread for periodic evaporation."""
        self._evaporation_thread = threading.Thread(
            target=self._auto_evaporation_worker,
            daemon=True,
            name="HydroMem-AutoEvaporation",
        )
        self._evaporation_thread.start()

    def _auto_evaporation_worker(self) -> None:
        """Worker loop for background daemon evaporation."""
        while not self._stop_evaporation_event.wait(self.evaporation_interval):
            try:
                self.evaporate()
            except Exception:
                pass

    def stop_auto_evaporation(self) -> None:
        """Signal and stop the background auto-evaporation daemon thread."""
        self._stop_evaporation_event.set()
        if self._evaporation_thread and self._evaporation_thread.is_alive():
            self._evaporation_thread.join(timeout=1.0)

    # ---------------- Persistence API ----------------

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """Save current memory hierarchy snapshot to disk.

        Args:
            path: Target storage path. Defaults to self.save_path.

        Returns:
            Optional[str]: Path saved to, or None if no path configured.
        """
        target_path = path or self.save_path
        if not target_path:
            return None
        if not self._local_store:
            self._local_store = LocalStore(path=target_path)

        state = {
            "id_counter": self.id_counter,
            "recall_tracker": dict(self.recall_tracker),
            "sensory": self.sensory_tank.get_all(),
            "short": self.short_tank.get_all(),
            "long": self.long_tank.get_all(),
        }
        return self._local_store.save(target_path, state)

    def load(self, path: Optional[str] = None) -> bool:
        """Load memory hierarchy snapshot from disk.

        Args:
            path: Source storage path. Defaults to self.save_path.

        Returns:
            bool: True if state was loaded.
        """
        target_path = path or self.save_path
        if not target_path or not self._local_store:
            return False

        state = self._local_store.load(target_path)
        self.id_counter = int(state.get("id_counter", 0))
        self.recall_tracker = {int(k): v for k, v in state.get("recall_tracker", {}).items()}

        self.sensory_tank.memories = []
        for item in state.get("sensory", []):
            self.sensory_tank.add(item)

        self.short_tank.memories = []
        for item in state.get("short", []):
            self.short_tank.add(item)

        # Long tank holds raw records with encrypted ciphertext
        self.long_tank.memories = list(state.get("long", []))
        return True

    # ---------------- Ingestion / Remember API ----------------

    def remember(
        self,
        text: str,
        emotion: float = 0.5,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Ingest and route a memory item to the appropriate tank based on hydraulic pressure.

        Routing threshold:
            - pressure > 0.8: encrypted in LongTank (permanent)
            - pressure > 0.4: stored in ShortTank (working memory)
            - otherwise: stored in SensoryTank (transient)

        Args:
            text: Plaintext content of the memory.
            emotion: Emotional intensity score in [0.0, 1.0]. Defaults to 0.5.
            created_at: Optional timestamp of creation. Defaults to current time.

        Returns:
            Dict[str, Any]: Metadata containing id, target tank name, and rounded pressure.
        """
        now = time.time()
        record_created_at = created_at if created_at is not None else now
        pressure = calculate_pressure(text=text, emotion=emotion, recency=1.0)
        self.id_counter += 1
        memory_id = self.id_counter

        # Optional FastEmbed vector embedding
        embedding: Optional[List[float]] = None
        if self.embedding_engine is not None:
            try:
                embedding = self.embedding_engine.embed_text(text)
            except Exception:
                pass

        memory_record: Dict[str, Any] = {
            "id": memory_id,
            "text": text,
            "pressure": pressure,
            "timestamp": now,
            "created_at": record_created_at,
        }
        if embedding is not None:
            memory_record["embedding"] = embedding

        if pressure > 0.8:
            tank_name = "long"
            self.long_tank.add_encrypted(
                text=text,
                key=self.key,
                memory_id=memory_id,
                pressure=pressure,
                timestamp=now,
                created_at=record_created_at,
                embedding=embedding,
            )
        elif pressure > 0.4:
            tank_name = "short"
            self.short_tank.add(memory_record)
        else:
            tank_name = "sensory"
            self.sensory_tank.add(memory_record)

        # Auto-persist if configured
        if self.save_path:
            self.save()

        return {
            "id": memory_id,
            "tank": tank_name,
            "pressure": round(pressure, 3),
        }

    def store(self, text: str, emotion: float = 0.5, **kwargs: Any) -> Dict[str, Any]:
        """Convenience alias for remember()."""
        return self.remember(text=text, emotion=emotion, **kwargs)

    # ---------------- Recall / Search API ----------------

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search and retrieve relevant memories across hierarchical tanks.

        Search order:
            1. LongTank (decrypted)
            2. ShortTank
            3. SensoryTank

        Uses:
            - Vector similarity if embeddings are active and available.
            - SQLite FTS5 BM25 search if running in SQLite mode.
            - Keyword matching with dynamic recency scoring in in-memory / JSON mode.

        Args:
            query: Search query string.
            top_k: Maximum number of memories to return.

        Returns:
            List[Dict[str, Any]]: Top matching memories sorted by composite score / pressure descending.
        """
        now = time.time()
        query_emb: Optional[List[float]] = None
        if self.embedding_engine is not None:
            try:
                query_emb = self.embedding_engine.embed_text(query)
            except Exception:
                pass

        # Collect candidate memories in order: LongTank (decrypted), ShortTank, SensoryTank
        candidates: List[tuple] = []
        for mem in self.long_tank.get_all_decrypted(self.key):
            candidates.append((mem, "long"))
        for mem in list(self.short_tank.get_all()):
            candidates.append((mem, "short"))
        for mem in self.sensory_tank.get_all():
            candidates.append((mem, "sensory"))

        matched: List[Dict[str, Any]] = []
        seen_ids = set()

        for mem, original_tank in candidates:
            mem_id = mem["id"]
            if mem_id in seen_ids:
                continue

            mem_text = mem.get("text", "")
            mem_created_at = float(mem.get("created_at", mem.get("timestamp", now)))
            dynamic_recency = compute_dynamic_recency(mem_created_at, current_time=now)
            pressure = float(mem.get("pressure", 0.0))

            # Determine relevance score
            if query_emb is not None and "embedding" in mem and mem["embedding"] is not None:
                relevance = max(0.0, cosine_similarity(query_emb, mem["embedding"]))
                is_hit = relevance > 0.1
            else:
                relevance = keyword_match_score(query, mem_text)
                is_hit = relevance > 0.0

            if not is_hit:
                continue

            seen_ids.add(mem_id)
            self.recall_tracker[mem_id] = self.recall_tracker.get(mem_id, 0) + 1
            current_tank = original_tank

            # Sedimentation logic: promote from ShortTank to LongTank upon > 3 recalls
            if original_tank == "short" and is_sedimented(self.recall_tracker[mem_id]):
                self.short_tank.remove(mem_id)
                self.long_tank.add_encrypted(
                    text=mem_text,
                    key=self.key,
                    memory_id=mem_id,
                    pressure=pressure,
                    timestamp=mem.get("timestamp", now),
                    created_at=mem_created_at,
                    embedding=mem.get("embedding"),
                )
                current_tank = "long"
                if self.save_path:
                    self.save()

            composite_score = compute_combined_score(
                relevance=relevance,
                recency=dynamic_recency,
                importance=pressure,
            )

            result_item: Dict[str, Any] = {
                "id": mem_id,
                "text": mem_text,
                "pressure": pressure,
                "tank": current_tank,
                "timestamp": mem.get("timestamp"),
                "created_at": mem_created_at,
                "recency": round(dynamic_recency, 4),
                "score": round(composite_score, 4),
            }
            matched.append(result_item)

        # Sort candidate matches: primarily by composite score, tie-break by hydraulic pressure
        matched.sort(
            key=lambda m: (m.get("score", 0.0), m.get("pressure", 0.0)),
            reverse=True,
        )
        return matched[:top_k]

    # ---------------- Maintenance & Evaporation API ----------------

    def forget(self, expired: bool = True) -> None:
        """Trigger evaporation across transient tanks (SensoryTank and ShortTank).

        Args:
            expired: If True, purges expired items according to tank TTLs.
        """
        self.sensory_tank.evaporate()
        self.short_tank.evaporate()
        if self.save_path:
            self.save()

    def evaporate(self, expired: bool = True) -> None:
        """Convenience alias for forget()."""
        self.forget(expired=expired)

    def stats(self) -> Dict[str, int]:
        """Return memory statistics across all tanks and total recalls.

        Returns:
            Dict[str, int]: Counts for sensory, short, long tanks and total recalls.
        """
        return {
            "sensory": len(self.sensory_tank),
            "short": len(self.short_tank),
            "long": len(self.long_tank),
            "total_recalls": sum(self.recall_tracker.values()),
        }
