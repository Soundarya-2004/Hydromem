"""HydroMem v0.2.0 - Hydraulic-Inspired Hierarchical Memory for LLMs with Local-First Privacy.

Exposes HydroMem memory manager along with individual tanks, security routines,
persistence engines, vector search, and dynamic hydraulic decay utilities.
"""

from .core import HydroMem
from .persistence import LocalStore
from .search import EmbeddingEngine, cosine_similarity, keyword_match_score
from .security import (
    decrypt_data,
    encrypt_data,
    generate_key,
    load_or_create_salt,
)
from .tank import DecryptedMemory, LongTank, SensoryTank, ShortTank
from .utils import (
    calculate_pressure,
    compute_combined_score,
    compute_dynamic_recency,
    evaporation_strength,
    is_sedimented,
)

__version__ = "0.0.1.post1"
__author__ = "Soundarya R"
__email__ = "soundaryaramachandra2003@gmail.com"

__all__ = [
    "HydroMem",
    "LocalStore",
    "EmbeddingEngine",
    "cosine_similarity",
    "keyword_match_score",
    "SensoryTank",
    "ShortTank",
    "LongTank",
    "DecryptedMemory",
    "generate_key",
    "encrypt_data",
    "decrypt_data",
    "load_or_create_salt",
    "calculate_pressure",
    "compute_dynamic_recency",
    "compute_combined_score",
    "evaporation_strength",
    "is_sedimented",
    "__version__",
    "__author__",
]
