<div align="center">

# 💧 HydroMem

**Hydraulic-Inspired Hierarchical Memory for LLMs with Local-First Privacy**

[![CI & Tests](https://github.com/Soundarya-2004/Hydromem/actions/workflows/ci.yml/badge.svg)](https://github.com/Soundarya-2004/Hydromem/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.0.1-blue.svg)](https://pypi.org/project/hydromem/)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://github.com/Soundarya-2004/Hydromem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Privacy: Local-First](https://img.shields.io/badge/privacy-100%25%20local--first-emerald.svg)](https://github.com/Soundarya-2004/Hydromem)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-mathematical-formulation">Formulation</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-comparison">Comparison</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 🌊 Overview

Modern Large Language Model (LLM) agent frameworks face a fundamental memory dilemma:
1. **Context Window Bloat**: Stuffing uncompressed chat history into prompt context rapidly exhausts token quotas and inflates inference latency.
2. **Privacy Vulnerabilities**: Offloading private memory records to third-party vector databases leaks confidential personal and enterprise data over remote network connections.
3. **Static Retention**: Simple FIFO sliding windows arbitrarily prune vital facts while retaining irrelevant conversational filler.

**HydroMem resolves this by modeling memory as a dynamic hydraulic fluid network.**

Instead of treating context as an unmanaged buffer, HydroMem regulates memory retention through **hydraulic pressure**, **dynamic recency decay**, and **continuous sedimentation**. Transient conversational filler naturally evaporates, active working memory is preserved during current tasks, and vital, high-pressure information is automatically consolidated and encrypted with AES-256 Fernet at rest.

---

## ✨ Key Features

- **🌊 Hydraulic Pressure Routing**: Dynamically computes fluid pressure based on context density, emotional valence, and recency, automatically dispatching memories to the appropriate chamber.
- **⏳ Continuous Recency Decay**: Evaluates temporal relevance on the fly using exponential age decay ($e^{-\Delta t / 24\text{h}}$) rather than static timestamps.
- **💎 Automated Sedimentation**: Frequently recalled working memories ($> 3$ recalls) automatically consolidate ("sediment") into permanent storage.
- **🔒 Local-First Cryptography**: Permanent memories are stored as authenticated AES-256 Fernet ciphertext. Master keys are derived via **Argon2id** or **PBKDF2-HMAC-SHA256** (100,000 rounds) using local 16-byte cryptographic salts.
- **⚡ Zero Heavy Dependencies by Default**: Core package requires only Python's standard library and `cryptography`. NumPy and heavy vector frameworks are strictly optional.
- **💾 Dual Persistence Backends**:
  - **JSON**: Human-readable, atomic state snapshots with zero configuration.
  - **SQLite FTS5**: Native full-text index with BM25 ranking for $O(\log N)$ search complexity.
- **🔍 Pluggable Semantic Search**: Seamlessly upgrade from normalized token-overlap keyword matching to local ONNX vector embeddings via `fastembed`.
- **🕒 Background Evaporation Daemon**: Optional non-blocking daemon thread automatically purges expired transient memories without blocking your agent's execution loop.

---

## 🏛️ Architecture

```text
                        [ Ingest Memory / store() ]
                                     │
                                     ▼
                           [ Calculate Pressure ]
                            (Length, Emotion)
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
            P <= 0.4           0.4 < P <= 0.8         P > 0.8
                  │                  │                  │
                  ▼                  ▼                  ▼
            ┌───────────┐      ┌───────────┐      ┌───────────┐
            │  Sensory  │      │   Short   │      │   Long    │
            │   Tank    │      │   Tank    │      │   Tank    │
            │  (5 min)  │      │  (1 day)  │      │(Permanent)│
            └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
                  │                  │                  ▲
              Evaporates         Evaporates             │
               (300s)            (86400s)               │
                                     │                  │
                                     └──────────────────┘
                                        Sedimentation
                                        (Recalls > 3)
```

### Hydraulic Memory Chambers

| Chamber | Retention Window (TTL) | Encryption | Typical Contents |
| :--- | :--- | :--- | :--- |
| **Sensory Tank** | 300 seconds (5 minutes) | Plaintext (in-memory) | Fleeting utterances, greetings, filler dialogue. |
| **Short Tank** | 86,400 seconds (24 hours) | Plaintext (working memory)| Active project objectives, immediate session context. |
| **Long Tank** | $\infty$ (Permanent bedrock) | **AES-256 Fernet Ciphertext** | Verified user preferences, system constraints, core facts. |

---

## 📦 Installation

### Core (Lightweight, Zero-Heavy Dependencies)
```bash
pip install hydromem
```

### Optional Feature Bundles
```bash
# Enable local vector embeddings via FastEmbed
pip install "hydromem[embed]"

# Enable Argon2id hardware-hardened key derivation
pip install "hydromem[secure]"

# Install all optional extensions
pip install "hydromem[all]"
```

---

## 🚀 Quickstart

### 1. Basic In-Memory Memory Management

```python
from hydromem import HydroMem

# Initialize HydroMem instance
mem = HydroMem(encryption_password="master-agent-key")

# Ingest memories with emotional / significance intensity
mem.store("User prefers concise Python code with strict type annotations.", emotion=0.9)
mem.store("Current local temperature is 24°C.", emotion=0.1)
mem.store("Working on sprint roadmap: deliver authentication module by Friday.", emotion=0.6)

# Query memories across all chambers
results = mem.recall("python annotations")
for item in results:
    print(f"[{item['tank'].upper()}] (Score: {item['score']}) -> {item['text']}")

# Inspect memory distribution
print("Stats:", mem.stats())
```

### 2. Auto-Persistent Memory (JSON or SQLite FTS5)

```python
# JSON backend: Automatically synchronizes state on ingestion
mem_json = HydroMem(save_path="workspace_memories.json")

# SQLite backend: Uses native FTS5 virtual table with BM25 ranking
mem_sql = HydroMem(save_path="workspace_memories.db", store_mode="sqlite")

mem_sql.store("Client requires SOC2-compliant data storage.", emotion=0.8)
```

### 3. Background Evaporation Daemon

```python
# Spawns a non-blocking background daemon thread to evaporate stale context
mem = HydroMem(
    auto_evaporation=True,
    evaporation_interval=3600.0,  # Run cleanup cycle every hour
)

# Shutdown cleanly when your application exits
mem.stop_auto_evaporation()
```

---

## 📐 Mathematical Formulation

### 1. Hydraulic Pressure ($P$)
Hydraulic pressure regulates which storage chamber a memory enters upon ingestion:

$$P = \text{clamp}\left( 0.6 \cdot \min\left(\frac{\text{len}(\text{text})}{100}, 1.0\right) + 0.3 \cdot \text{emotion} + 0.1 \cdot \text{recency},\; 0.0,\; 1.0 \right)$$

- $P > 0.8 \implies$ **LongTank** (Directly encrypted with AES-256)
- $0.4 < P \le 0.8 \implies$ **ShortTank** (Working memory buffer)
- $P \le 0.4 \implies$ **SensoryTank** (Transient buffer)

### 2. Dynamic Recency Decay ($R$)
Rather than freezing recency at ingestion, HydroMem evaluates time elapsed continuously during recall:

$$R(\Delta t) = \exp\left( -\frac{\Delta t_{\text{hours}}}{24.0} \right)$$

A memory created 24 hours ago retains $\approx 36.8\%$ recency score, decaying gradually to prioritize fresher dialogue.

### 3. Composite Retrieval Score
Candidate memories across all chambers are scored and ranked via a weighted multi-factor objective:

$$\text{Score} = 0.5 \cdot \text{Relevance} + 0.3 \cdot R(\Delta t) + 0.2 \cdot P$$

Where $\text{Relevance}$ is computed via normalized token overlap or cosine similarity.

---

## 🔒 Security Architecture

HydroMem was built from the ground up for zero-trust, enterprise, and local-first AI pipelines:

1. **Key Derivation (KDF)**:
   - **Argon2id**: Memory-hardened key derivation (default if `argon2-cffi` is installed).
   - **PBKDF2-HMAC-SHA256**: 100,000 rounds fallback utilizing an isolated 16-byte cryptographic salt.
   - **Raw Key Support**: Backward-compatible with direct 32-byte hex strings and pre-generated Fernet tokens.
2. **AES-128/256 Symmetric Encryption**: Authenticated symmetric encryption via Fernet guarantees both confidentiality and ciphertext tamper protection at rest.
3. **Zero Network Egress**: Absolutely no outbound network calls, telemetry, or external API dependencies.

---

## 📖 API Reference

### `HydroMem`

```python
HydroMem(
    encryption_password: str = "hydromem",
    save_path: Optional[str] = None,
    store_mode: Optional[str] = None,         # "json" or "sqlite"
    embedding_model: Optional[str] = None,    # e.g. "BAAI/bge-small-en"
    auto_evaporation: bool = False,           # Enable background cleanup thread
    evaporation_interval: float = 3600.0,     # Interval in seconds
    salt: Optional[bytes] = None,             # Custom 16-byte salt
    salt_path: Optional[str] = None,          # Path to load/store salt file
    use_argon2: bool = True,                  # Prefer Argon2id KDF
)
```

#### Core Methods

| Method | Signature | Description |
| :--- | :--- | :--- |
| `store` / `remember` | `(text: str, emotion: float = 0.5, created_at: Optional[float] = None) -> dict` | Computes hydraulic pressure and dispatches memory to target tank. |
| `recall` | `(query: str, top_k: int = 3) -> list[dict]` | Searches all tanks, evaluates dynamic recency, sediments frequent items, and returns ranked results. |
| `forget` / `evaporate`| `(expired: bool = True) -> None` | Purges expired memories according to tank TTLs (300s Sensory, 86400s Short). |
| `save` | `(path: Optional[str] = None) -> Optional[str]` | Persists current state snapshot to disk (JSON or SQLite). |
| `load` | `(path: Optional[str] = None) -> bool` | Reconstitutes memory state from a persisted snapshot. |
| `stats` | `() -> dict` | Returns item counts across sensory, short, long tanks and lifetime recalls. |
| `stop_auto_evaporation`| `() -> None` | Signals and cleanly shuts down the background daemon thread. |

---

## 📊 Comparison

| Feature | **HydroMem 💧** | **MemGPT / Letta** | **LangChain Memory** | **Vector DBs** |
| :--- | :---: | :---: | :---: | :---: |
| **Core Architecture** | Fluid Hydraulic Hierarchy | OS Virtual Memory Hierarchy | Sliding Window / Buffer | Flat Vector Index |
| **Decay Mechanism** | Exponential Fluid Decay | LLM-driven eviction | Manual / None | Manual TTL |
| **Dynamic Recency** | Continuous $e^{-\Delta t / 24\text{h}}$ | Static Timestamps | None | Timestamp Filter |
| **Consolidation** | Automated Sedimentation ($>3$) | LLM Function-Calling | None | None |
| **Local Privacy** | **100% Local-First** | Requires Server / LLM Backend | Dependent on Provider | Cloud or Self-Hosted |
| **Encryption at Rest** | **AES Fernet (Argon2id/PBKDF2)**| Plaintext Database | None (In-Memory) | Database Specific |
| **Persistence** | Zero-Config JSON & SQLite FTS5 | PostgreSQL / Redis | Vector DB or In-Memory | Dedicated Server |
| **Core Dependencies** | Ultra-light (`cryptography` only)| Heavy (OpenAI, Server, DB) | Heavy Framework Stack | Heavy Client SDKs |

---

## 🧪 Testing

HydroMem maintains a comprehensive unit test suite tested across Linux, macOS, and Windows on Python 3.8 through 3.12:

```bash
# Run full test suite with verbose output
pytest -v -rA
```

```text
============================= test session starts =============================
tests/test_memory.py::test_remember_routing PASSED                       [  9%]
tests/test_memory.py::test_recall_keyword PASSED                         [ 18%]
tests/test_memory.py::test_evaporation PASSED                            [ 27%]
tests/test_memory.py::test_encryption PASSED                             [ 36%]
tests/test_memory.py::test_sedimentation PASSED                          [ 45%]
tests/test_v02.py::test_persistence_json PASSED                          [ 54%]
tests/test_v02.py::test_persistence_sqlite_fts5 PASSED                   [ 63%]
tests/test_v02.py::test_cosine_similarity_pure_python PASSED             [ 72%]
tests/test_v02.py::test_key_derivation_and_backward_compat PASSED        [ 81%]
tests/test_v02.py::test_auto_evaporation_daemon PASSED                   [ 90%]
tests/test_v02.py::test_dynamic_recency_and_composite_ranking PASSED     [100%]
============================= 11 passed in 3.15s ==============================
```

---

## 📑 Citation

If you use HydroMem in your research or AI applications, please cite:

```bibtex
@software{hydromem2026,
  author = {Soundarya R},
  title = {HydroMem: Hydraulic-Inspired Hierarchical Memory for LLMs with Local-First Privacy},
  year = {2026},
  url = {https://github.com/Soundarya-2004/Hydromem},
  note = {PyPI package version 0.0.1}
}
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](file:///c:/Users/Soundarya/OneDrive/Desktop/Hydromem/LICENSE) for more information.

Copyright (c) 2026 Soundarya R.
