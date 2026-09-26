# HydroMem 💧

**Hydraulic-Inspired Hierarchical Memory for LLMs with Local-First Privacy**

[![PyPI version](https://img.shields.io/badge/pypi-v0.0.1-blue.svg)](https://pypi.org/project/hydromem/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://pypi.org/project/hydromem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Privacy: Local-First](https://img.shields.io/badge/privacy-local--first-emerald.svg)](https://github.com/Soundarya-2004/Hydromem)

```bash
pip install hydromem
```

HydroMem models memory as a **continuous hydraulic fluid system**. Instead of stuffing prompt windows until they overflow or relying on opaque external vector databases that leak private user data over HTTP APIs, HydroMem dynamically regulates memory retention through **hydraulic pressure**, **dynamic recency**, **evaporative decay**, and **continuous sedimentation**—all locally encrypted with AES-256 Fernet.

---

## 🌊 Why Hydraulics?

Traditional LLM memory models treat context as a static bucket: memories are either stuffed verbatim into the prompt window until it overflows, or dumped into massive external vector databases that leak private user data over HTTP APIs.

**HydroMem reimagines memory retention through fluid dynamics:**
- **Hydraulic Pressure**: High-pressure memories (dense context, heightened emotional valence, recency) break through high-resistance valves into deeper storage chambers.
- **Natural Evaporation**: Low-pressure sensory impressions naturally evaporate without requiring manual cleanup or arbitrary token truncation.
- **Dynamic Recency**: Ingestion times decay continuously according to $e^{-\text{age\_hours}/24}$, balancing relevance and freshness in retrieval.
- **Sedimentation**: Frequently recalled working memories gradually consolidate ("sediment") over time into permanent, encrypted bedrock.
- **Zero Cloud Leakage**: All keys and ciphertexts remain strictly on the host device. Zero external API calls, zero telemetry, zero cloud dependencies.

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

### Architectural Pipeline
```text
[Sensory Tank 5min] -> Valve(Pressure) -> [Short Tank 1 day] -> Sedimentation -> [Long Tank Encrypted Permanent]
```

1. **Sensory Tank (Transient)**:
   - Buffer TTL: 300 seconds (5 minutes).
   - Fleeting conversational noise, filler utterances, and short-term dialogue nuances.
2. **Short Tank (Working Memory)**:
   - Retention window: 86,400 seconds (24 hours).
   - Active project context, current task objectives, and intermediate dialogue state.
3. **Long Tank (Encrypted Bedrock)**:
   - Permanent storage.
   - Critical facts, persistent preferences, and high-impact information.
   - Stored strictly as AES-encrypted ciphertext derived from user-controlled passwords via Argon2id / PBKDF2.

---

## 📦 Installation

### Core (Zero heavy dependencies, ultra-lightweight)
```bash
pip install hydromem
```

### Optional Extras
```bash
# With fast local vector embeddings (FastEmbed)
pip install "hydromem[embed]"

# With Argon2id hardware-hardened key derivation
pip install "hydromem[secure]"

# Complete installation with all optional features
pip install "hydromem[all]"
```

---

## 🚀 Quickstart

```python
from hydromem import HydroMem

# 1. Initialize HydroMem with optional disk persistence and encryption password
mem = HydroMem(
    encryption_password="my-secure-master-password",
    save_path="memories.json",       # Auto-saves and loads state
    auto_evaporation=True,           # Starts background cleanup daemon
    evaporation_interval=3600,       # Periodic cleanup every hour
)

# 2. Ingest memories with emotional intensity ratings
mem.store("User prefers federated learning and local-first AI architectures", emotion=0.9)
mem.store("Current weather outside is 24 degrees Celsius", emotion=0.2)
mem.store("Working on sprint deliverables: complete API integration by Thursday", emotion=0.6)

# 3. Recall relevant memories using keyword matching or vector search
results = mem.recall("federated architectures", top_k=3)
for item in results:
    print(f"[{item['tank'].upper()}] (Score: {item['score']}) {item['text']}")

# 4. Inspect system statistics
print("Memory Stats:", mem.stats())
```

---

## 🔒 Security & Local-First Cryptography

HydroMem is engineered from the ground up for strict zero-trust, local-first environments:

- **Key Derivation (KDF)**:
  - **Argon2id**: Memory-hardened key derivation preferred when `argon2-cffi` is installed.
  - **PBKDF2-HMAC-SHA256**: 100,000 rounds standard fallback with cryptographically random 16-byte salts.
  - **Raw Key Compatibility**: Seamlessly accepts 32-byte hex strings or pre-generated Fernet keys.
- **AES-128/256 Fernet Encryption**: Ciphertext integrity guaranteed via authenticated HMAC-SHA256.
- **Zero Network Egress**: Absolutely no outbound network calls, analytics, or third-party telemetry.

---

## 💾 Persistence Backends

HydroMem includes built-in `LocalStore` persistence:

1. **JSON Backend (Default)**:
   ```python
   mem = HydroMem(save_path="workspace_memory.json")
   ```
   Simple, human-readable snapshot that automatically synchronizes on ingestion.

2. **SQLite Backend with FTS5 BM25 Search**:
   ```python
   mem = HydroMem(save_path="workspace_memory.db", store_mode="sqlite")
   ```
   Uses SQLite's native virtual table `fts5` with BM25 ranking to perform $O(\log N)$ full-text search.

---

## 📖 API Reference

### `HydroMem(...)`
Main orchestrator managing sensory, short-term, and long-term tanks.

```python
HydroMem(
    encryption_password: str = "hydromem",
    save_path: Optional[str] = None,
    store_mode: Optional[str] = None,         # "json" or "sqlite"
    embedding_model: Optional[str] = None,    # e.g. "BAAI/bge-small-en"
    auto_evaporation: bool = False,           # Background daemon thread
    evaporation_interval: float = 3600.0,     # Evaporation cycle interval
    salt: Optional[bytes] = None,             # 16-byte cryptographic salt
    use_argon2: bool = True,                  # Prefer Argon2id KDF
)
```

#### `remember(text: str, emotion: float = 0.5, created_at: Optional[float] = None) -> dict`
#### `store(...)` *(alias for remember)*
Evaluates hydraulic pressure and routes memory:
- **Parameters**:
  - `text` (*str*): The memory text to store.
  - `emotion` (*float*, optional): Emotional significance score from `0.0` to `1.0`. Defaults to `0.5`.
  - `created_at` (*float*, optional): Creation timestamp. Defaults to current epoch time.
- **Returns**: `{"id": int, "tank": str, "pressure": float}`

#### `recall(query: str, top_k: int = 3) -> list[dict]`
Searches across all chambers in sequence (`LongTank` decrypted $\rightarrow$ `ShortTank` $\rightarrow$ `SensoryTank`), scores candidates via:
$$\text{Score} = 0.5 \times \text{Relevance} + 0.3 \times \text{Recency} + 0.2 \times \text{Importance}$$
Promotes working memories to permanent encrypted storage upon reaching sedimentation threshold ($> 3$ recalls).

#### `forget(expired: bool = True) -> None`
#### `evaporate(...)` *(alias for forget)*
Purges expired memories across transient chambers:
- Removes memories older than 300 seconds from `SensoryTank`.
- Removes memories older than 86,400 seconds from `ShortTank`.

#### `stop_auto_evaporation() -> None`
Gracefully signals and terminates the background auto-evaporation daemon thread.

#### `stats() -> dict`
Returns real-time status across tanks:
```python
{
    "sensory": int,       # Active memories in sensory tank
    "short": int,         # Active memories in short tank
    "long": int,          # Encrypted memories in long tank
    "total_recalls": int  # Lifetime recall event counter
}
```

---

## 📊 Comparison Table

| Feature | **HydroMem 💧** | **MemGPT / Letta** | **LangChain Memory** |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Hydraulic fluid hierarchy | OS-style hierarchical virtual memory | Linear chat message buffers |
| **Decay Mechanism** | Exponential fluid evaporation | LLM-driven eviction function | Sliding window / manual clear |
| **Dynamic Recency** | $e^{-\text{age\_hours}/24}$ composite | Static timestamps | None |
| **Consolidation** | Automated sedimentation (>3 recalls) | Agent function-calling to archive | None / manual summary chains |
| **Local Privacy** | 100% Local-First (Zero HTTP/Cloud) | Requires LLM/Server backend | Dependent on provider |
| **Permanent Encryption** | AES Fernet at rest (Argon2id/PBKDF2) | Database-dependent (often plaintext) | None (in-memory plaintext) |
| **Persistence** | JSON & SQLite with FTS5 BM25 | PostgreSQL / Vector DB | Vector DB or In-Memory |
| **Token Overhead** | Minimal (top-k pressure ranked) | Heavy (self-editing prompts) | Heavy (accumulating history) |
| **Dependencies** | Ultra-light (`cryptography` only) | Heavy (Server, OpenAI, pgvector) | Heavy framework dependencies |

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest -v
```

Test coverage includes:
- `test_remember_routing`: Validates pressure calculations and dynamic routing thresholds.
- `test_recall_keyword`: Tests multi-tank keyword discovery and pressure-based ranking.
- `test_evaporation`: Verifies time-based evaporation across sensory and short tanks.
- `test_encryption`: Confirms key derivation, ciphertext safety, and decryption.
- `test_sedimentation`: Validates promotion from ShortTank to LongTank upon frequent recall.
- `test_persistence_json`: Verifies JSON auto-saving and auto-loading on initialization.
- `test_persistence_sqlite_fts5`: Verifies SQLite FTS5 table creation and BM25 search.
- `test_cosine_similarity_pure_python`: Tests pure-Python vector similarity calculation.
- `test_key_derivation_and_backward_compat`: Tests PBKDF2, Argon2id, and raw key compatibility.
- `test_auto_evaporation_daemon`: Tests background daemon thread execution and shutdown.
- `test_dynamic_recency_and_composite_ranking`: Verifies time-decay scoring and ranking.

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

Distributed under the **MIT License**. See `LICENSE` for more information.  
Copyright (c) 2026 Soundarya R.
