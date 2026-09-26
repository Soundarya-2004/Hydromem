"""Test suite for HydroMem v0.2.0 features.

Covers:
1. Persistence: LocalStore JSON and SQLite FTS5 save/load and auto-persistence
2. Search: Keyword scoring, pure Python cosine similarity, and vector similarity routing
3. Search Complexity: SQLite FTS5 virtual table and BM25 full-text querying
4. Key Derivation: PBKDF2-HMAC (100k), Argon2id, salt management, hex/base64 raw key backward compat
5. Auto-Evaporation: Background daemon thread lifecycle and non-blocking termination
6. Dynamic Recency: Age-based exponential decay and composite score ranking
"""

import os
import tempfile
import time
import pytest
from hydromem.core import HydroMem
from hydromem.persistence import LocalStore
from hydromem.search import cosine_similarity, keyword_match_score
from hydromem.security import (
    ARGON2_AVAILABLE,
    generate_key,
    load_or_create_salt,
)
from hydromem.utils import (
    compute_combined_score,
    compute_dynamic_recency,
)


def test_persistence_json():
    """Verify LocalStore JSON persistence and HydroMem auto-load / auto-save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "mem.json")

        # 1. Initialize HydroMem with persistence path
        mem1 = HydroMem(save_path=json_path)
        res = mem1.store("Persisted user preference: prefers async Python code", emotion=0.8)
        assert res["id"] == 1
        assert os.path.exists(json_path)

        # Ingest a second memory
        mem1.remember("Temporary note about server latency", emotion=0.2)
        assert mem1.stats()["sensory"] + mem1.stats()["short"] + mem1.stats()["long"] == 2

        # 2. Re-initialize HydroMem from same json_path; should auto-load state
        mem2 = HydroMem(save_path=json_path)
        stats2 = mem2.stats()
        assert stats2["sensory"] == mem1.stats()["sensory"]
        assert stats2["short"] == mem1.stats()["short"]
        assert stats2["long"] == mem1.stats()["long"]

        recalled = mem2.recall("preference async")
        assert len(recalled) >= 1
        assert "prefers async Python" in recalled[0]["text"]


def test_persistence_sqlite_fts5():
    """Verify SQLite storage with FTS5 virtual table and BM25 retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")

        # Test LocalStore SQLite backend directly
        store = LocalStore(path=db_path)
        assert store.mode == "sqlite"

        state = {
            "id_counter": 2,
            "recall_tracker": {1: 1, 2: 0},
            "sensory": [],
            "short": [
                {
                    "id": 1,
                    "text": "Machine learning model weights checkpoint saved",
                    "pressure": 0.6,
                    "timestamp": time.time(),
                }
            ],
            "long": [
                {
                    "id": 2,
                    "text": "Encrypted credentials for secure edge client",
                    "pressure": 0.9,
                    "timestamp": time.time(),
                }
            ],
        }
        store.save(db_path, state)
        assert os.path.exists(db_path)

        # Verify FTS5 BM25 search
        fts_results = store.search_fts5("checkpoint weights", path=db_path)
        assert len(fts_results) >= 1
        assert fts_results[0]["id"] == 1
        assert "Machine learning" in fts_results[0]["text"]

        # Re-load state from SQLite
        loaded_state = store.load(db_path)
        assert loaded_state["id_counter"] == 2
        assert len(loaded_state["short"]) == 1
        assert len(loaded_state["long"]) == 1


def test_cosine_similarity_pure_python():
    """Verify pure Python cosine similarity without numpy dependency."""
    # Orthogonal vectors -> 0.0
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    # Identical vectors -> 1.0
    v3 = [0.5, 0.5, 0.0]
    assert cosine_similarity(v3, v3) == pytest.approx(1.0)

    # Opposite vectors -> -1.0
    v4 = [-0.5, -0.5, 0.0]
    assert cosine_similarity(v3, v4) == pytest.approx(-1.0)

    # Keyword match score
    score = keyword_match_score("neural network", "Deep neural network architectures")
    assert score == pytest.approx(1.0)
    score_half = keyword_match_score("neural quantum", "Deep neural network architectures")
    assert score_half == pytest.approx(0.5)


def test_key_derivation_and_backward_compat():
    """Verify PBKDF2 (100k), Argon2id, salt management, and hex raw key compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        salt_file = os.path.join(tmpdir, "test.salt")

        # 1. Salt creation and loading
        salt1 = load_or_create_salt(salt_file)
        assert len(salt1) == 16
        salt2 = load_or_create_salt(salt_file)
        assert salt1 == salt2  # Loaded same salt from file

        # 2. PBKDF2 KDF derivation (force use_argon2=False)
        key_pbkdf2 = generate_key("password123", salt=salt1, use_argon2=False)
        assert len(key_pbkdf2) == 44  # Base64 Fernet key

        # Same password and salt produces deterministic key
        key_pbkdf2_repeat = generate_key("password123", salt=salt1, use_argon2=False)
        assert key_pbkdf2 == key_pbkdf2_repeat

        # Different password produces different key
        key_other = generate_key("different-password", salt=salt1, use_argon2=False)
        assert key_pbkdf2 != key_other

        # 3. Argon2id derivation (if installed in environment)
        if ARGON2_AVAILABLE:
            key_argon = generate_key("password123", salt=salt1, use_argon2=True)
            assert len(key_argon) == 44

        # 4. Backward compat: 32-byte hex raw key (64 hex characters)
        raw_hex_32 = "a" * 64
        key_from_hex = generate_key(raw_hex_32)
        assert len(key_from_hex) == 44

        # 5. Backward compat: Valid 44-byte Fernet key string
        existing_fernet = key_pbkdf2.decode("utf-8")
        key_from_fernet = generate_key(existing_fernet)
        assert key_from_fernet == key_pbkdf2


def test_auto_evaporation_daemon():
    """Verify background auto-evaporation daemon thread."""
    mem = HydroMem(auto_evaporation=True, evaporation_interval=0.1)
    assert mem._evaporation_thread is not None
    assert mem._evaporation_thread.is_alive()
    assert mem._evaporation_thread.daemon is True

    # Seed an expired memory in sensory tank
    mem.sensory_tank.add(
        {
            "id": 99,
            "text": "Fleeting sensory noise",
            "pressure": 0.1,
            "timestamp": time.time() - 400.0,
        }
    )
    assert mem.stats()["sensory"] == 1

    # Wait for daemon thread to execute cycle
    time.sleep(0.35)
    assert mem.stats()["sensory"] == 0

    # Clean shutdown
    mem.stop_auto_evaporation()
    assert not mem._evaporation_thread.is_alive()


def test_dynamic_recency_and_composite_ranking():
    """Verify dynamic recency calculation and combined ranking score."""
    now = time.time()

    # Brand new memory (age = 0 hours) -> recency = 1.0
    rec_now = compute_dynamic_recency(created_at=now, current_time=now)
    assert rec_now == pytest.approx(1.0)

    # 24-hour-old memory -> recency = exp(-1) ~= 0.3679
    one_day_ago = now - 86400.0
    rec_day = compute_dynamic_recency(created_at=one_day_ago, current_time=now)
    assert rec_day == pytest.approx(0.367879, rel=1e-3)

    # 48-hour-old memory -> recency = exp(-2) ~= 0.1353
    two_days_ago = now - (86400.0 * 2)
    rec_two_days = compute_dynamic_recency(created_at=two_days_ago, current_time=now)
    assert rec_two_days == pytest.approx(0.135335, rel=1e-3)

    # Verify HydroMem.recall prioritizes recent memory when relevance is equal
    mem = HydroMem()
    # Old memory ingested with created_at set to 2 days ago
    mem.remember(
        "Project status: initial sprint kickoff notes",
        emotion=0.5,
        created_at=two_days_ago,
    )
    # Fresh memory ingested with current timestamp
    mem.remember(
        "Project status: final sprint deployment completed",
        emotion=0.5,
        created_at=now,
    )

    results = mem.recall("Project status", top_k=2)
    assert len(results) == 2
    # Fresh memory has higher recency, so higher composite score -> ranked 1st
    assert results[0]["created_at"] > results[1]["created_at"]
    assert "deployment completed" in results[0]["text"]
    assert results[0]["score"] > results[1]["score"]
