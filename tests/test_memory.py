"""Unit test suite for HydroMem package.

Covers:
- test_remember_routing: Validates pressure-based chamber distribution
- test_recall_keyword: Tests keyword search, multi-tank discovery, and pressure sorting
- test_evaporation: Verifies TTL evaporation for sensory (300s) and short-term (86400s) tanks
- test_encryption: Tests key derivation, Fernet encryption/decryption, and LongTank storage
- test_sedimentation: Verifies automated memory promotion from ShortTank to LongTank (> 3 recalls)
"""

import time
import pytest
from hydromem.core import HydroMem
from hydromem.security import decrypt_data, encrypt_data, generate_key
from hydromem.tank import LongTank, SensoryTank, ShortTank
from hydromem.utils import calculate_pressure, evaporation_strength, is_sedimented


def test_remember_routing():
    """Verify that memories are routed to the proper tank according to hydraulic pressure."""
    mem = HydroMem(encryption_password="routing-test-password")

    # High pressure (> 0.8) -> LongTank
    # Len 100 -> relevance 1.0; emotion 0.9; recency 1.0 -> 0.6 + 0.27 + 0.1 = 0.97 > 0.8
    long_text = "A" * 120
    res_long = mem.remember(long_text, emotion=0.9)
    assert res_long["tank"] == "long"
    assert res_long["pressure"] > 0.8

    # Medium pressure (0.4 < pressure <= 0.8) -> ShortTank
    # Len 40 -> relevance 0.4; emotion 0.5; recency 1.0 -> 0.24 + 0.15 + 0.1 = 0.49
    res_short = mem.remember("Working memory buffer for daily task management", emotion=0.5)
    assert res_short["tank"] == "short"
    assert 0.4 < res_short["pressure"] <= 0.8

    # Low pressure (<= 0.4) -> SensoryTank
    # Len 5 -> relevance 0.05; emotion 0.1; recency 1.0 -> 0.03 + 0.03 + 0.1 = 0.16 <= 0.4
    res_sensory = mem.remember("Hello", emotion=0.1)
    assert res_sensory["tank"] == "sensory"
    assert res_sensory["pressure"] <= 0.4

    # Verify overall tank counts via stats()
    stats = mem.stats()
    assert stats["long"] == 1
    assert stats["short"] == 1
    assert stats["sensory"] == 1
    assert stats["total_recalls"] == 0


def test_recall_keyword():
    """Verify multi-tank keyword discovery, recall tracking, and pressure-based descending sort."""
    mem = HydroMem(encryption_password="recall-test-password")

    # Store items with distinct content and known pressures
    # Long text (len > 100, emotion=0.9 -> pressure > 0.8 -> LongTank)
    mem.remember(
        "Federated learning preserves privacy across distributed edge devices and local client nodes in enterprise AI",
        emotion=0.9,
    )
    # Medium text (pressure in (0.4, 0.8] -> ShortTank)
    mem.remember("Optimization algorithms for convex loss minimization", emotion=0.5)
    # Short text (pressure <= 0.4 -> SensoryTank)
    mem.remember("Quick note on today's weather", emotion=0.1)

    # Search query targeting LongTank item
    res_fed = mem.recall("federated edges")
    assert len(res_fed) >= 1
    assert "Federated learning" in res_fed[0]["text"]
    assert res_fed[0]["tank"] == "long"

    # Search query targeting multiple items: should return sorted by pressure descending
    res_multi = mem.recall("learning optimization weather", top_k=3)
    assert len(res_multi) >= 2
    for i in range(len(res_multi) - 1):
        assert res_multi[i]["pressure"] >= res_multi[i + 1]["pressure"]

    # Verify recall_tracker was updated
    stats = mem.stats()
    assert stats["total_recalls"] > 0


def test_evaporation():
    """Verify time-based memory evaporation in SensoryTank, ShortTank, and forget() method."""
    now = time.time()

    # 1. Sensory Tank: TTL = 300s
    sensory = SensoryTank()
    old_sensory = {"id": 1, "text": "Old sensory memory", "pressure": 0.2, "timestamp": now - 350.0}
    fresh_sensory = {"id": 2, "text": "Fresh sensory memory", "pressure": 0.2, "timestamp": now}
    sensory.add(old_sensory)
    sensory.add(fresh_sensory)
    assert len(sensory) == 2

    evaporated_sensory = sensory.evaporate(max_age=300.0, current_time=now)
    assert len(evaporated_sensory) == 1
    assert evaporated_sensory[0]["id"] == 1
    assert len(sensory) == 1
    assert sensory.get_all()[0]["id"] == 2

    # 2. Short Tank: TTL = 86400s
    short = ShortTank()
    old_short = {"id": 3, "text": "Yesterday's context", "pressure": 0.5, "timestamp": now - 90000.0}
    fresh_short = {"id": 4, "text": "Current context", "pressure": 0.5, "timestamp": now}
    short.add(old_short)
    short.add(fresh_short)
    assert len(short) == 2

    evaporated_short = short.evaporate(max_age=86400.0, current_time=now)
    assert len(evaporated_short) == 1
    assert evaporated_short[0]["id"] == 3
    assert len(short) == 1
    assert short.get_all()[0]["id"] == 4

    # 3. HydroMem.forget() integration
    mem = HydroMem()
    mem.sensory_tank.add(old_sensory)
    mem.short_tank.add(old_short)
    assert mem.stats()["sensory"] == 1
    assert mem.stats()["short"] == 1
    mem.forget()
    assert mem.stats()["sensory"] == 0
    assert mem.stats()["short"] == 0

    # 4. Mathematical evaporation decay formula verification
    decayed = evaporation_strength(initial=1.0, lambda_val=0.1, time_elapsed=10.0)
    assert decayed == pytest.approx(0.367879, rel=1e-3)


def test_encryption():
    """Verify cryptographic key derivation, Fernet ciphertext safety, and LongTank storage."""
    password = "secret-hydro-encryption-key"
    key = generate_key(password)
    assert isinstance(key, bytes)
    assert len(key) == 44  # Base64-encoded 32-byte hash

    plaintext = "Sensitive medical diagnostics and private client records"
    ciphertext = encrypt_data(plaintext, key)
    assert ciphertext != plaintext
    assert plaintext not in ciphertext

    decrypted = decrypt_data(ciphertext, key)
    assert decrypted == plaintext

    # Verify wrong password / key fails decryption
    wrong_key = generate_key("wrong-password")
    with pytest.raises(Exception):
        decrypt_data(ciphertext, wrong_key)

    # Test LongTank storage and decryption
    long_tank = LongTank(key=key)
    stored_record = long_tank.add_encrypted(plaintext, key=key, memory_id=42, pressure=0.95)
    assert long_tank.count() == 1
    assert stored_record["text"] != plaintext

    decrypted_memories = long_tank.get_all_decrypted(key)
    assert len(decrypted_memories) == 1
    assert decrypted_memories[0]["text"] == plaintext
    assert decrypted_memories[0]["id"] == 42
    assert decrypted_memories[0]["pressure"] == 0.95


def test_sedimentation():
    """Verify that frequently recalled memories sediment from ShortTank into LongTank (> 3 recalls)."""
    mem = HydroMem(encryption_password="sedimentation-password")

    # Ingest a memory that routes to ShortTank (pressure ~0.5)
    res = mem.remember("Frequent task pattern: data cleaning pipeline before model training", emotion=0.5)
    mem_id = res["id"]
    assert res["tank"] == "short"

    # Initial state: 1 short, 0 long
    assert mem.stats()["short"] == 1
    assert mem.stats()["long"] == 0

    # Recall 1st time (recall_count = 1) -> still in ShortTank
    mem.recall("pipeline")
    assert mem.stats()["short"] == 1
    assert mem.stats()["long"] == 0
    assert not is_sedimented(1)

    # Recall 2nd time (recall_count = 2) -> still in ShortTank
    mem.recall("pipeline")
    assert mem.stats()["short"] == 1
    assert mem.stats()["long"] == 0
    assert not is_sedimented(2)

    # Recall 3rd time (recall_count = 3) -> still in ShortTank (threshold is > 3)
    mem.recall("pipeline")
    assert mem.stats()["short"] == 1
    assert mem.stats()["long"] == 0
    assert not is_sedimented(3)

    # Recall 4th time (recall_count = 4) -> Sedimentation triggers (> 3)!
    mem.recall("pipeline")
    assert is_sedimented(4)
    assert mem.stats()["short"] == 0
    assert mem.stats()["long"] == 1

    # Verify the sedimented memory is stored encrypted and still recallable from LongTank
    recalled_after = mem.recall("pipeline")
    assert len(recalled_after) == 1
    assert recalled_after[0]["id"] == mem_id
    assert recalled_after[0]["tank"] == "long"
    assert "cleaning pipeline" in recalled_after[0]["text"]
