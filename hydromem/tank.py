"""Storage tanks for HydroMem hydraulic memory hierarchy.

Provides three specialized, user-configurable storage tiers:
- SensoryTank: Transient short-lived buffer (configurable TTL, default 300s / 5min)
- ShortTank: Intermediate working memory (configurable TTL, default 86400s / 24h)
- LongTank: Encrypted permanent storage (AES Fernet encrypted, never expires)
"""

import time
from typing import Any, Dict, List, Optional, Union
from .security import decrypt_data, encrypt_data


class DecryptedMemory(dict):
    """A dictionary representing a decrypted memory item.

    Inherits from dict and supports direct equality comparison with strings
    against its 'text' field for seamless testing and search convenience.
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("text") == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return str(self.get("text", ""))


class SensoryTank:
    """Sensory buffer tank with a user-configurable evaporation window (default 300s).

    Stores dictionaries with: id, text, pressure, timestamp, created_at.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        """Initialize an empty SensoryTank.

        Args:
            ttl: Evaporation TTL window in seconds (default 300.0).
        """
        self.ttl = float(ttl)
        self.memories: List[Dict[str, Any]] = []

    def add(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Add a memory dictionary to the sensory tank.

        Args:
            memory: Dictionary containing id, text, pressure, timestamp, and optional created_at.

        Returns:
            Dict[str, Any]: The stored memory dictionary.
        """
        now = time.time()
        ts = float(memory.get("timestamp", now))
        created_at = float(memory.get("created_at", ts))
        record = {
            "id": memory.get("id"),
            "text": str(memory.get("text", "")),
            "pressure": float(memory.get("pressure", 0.0)),
            "timestamp": ts,
            "created_at": created_at,
        }
        if "embedding" in memory and memory["embedding"] is not None:
            record["embedding"] = memory["embedding"]

        self.memories.append(record)
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all active memories in the sensory tank.

        Returns:
            List[Dict[str, Any]]: List of stored memory dictionaries.
        """
        return list(self.memories)

    def evaporate(self, max_age: Optional[float] = None, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Remove memories older than the configured TTL (or custom max_age).

        Args:
            max_age: Maximum age in seconds before evaporation occurs (defaults to self.ttl).
            current_time: Reference timestamp; defaults to time.time().

        Returns:
            List[Dict[str, Any]]: The list of evaporated (removed) memories.
        """
        limit = max_age if max_age is not None else self.ttl
        now = current_time if current_time is not None else time.time()
        retained: List[Dict[str, Any]] = []
        evaporated: List[Dict[str, Any]] = []

        for mem in self.memories:
            age = now - mem.get("timestamp", now)
            if age >= limit:
                evaporated.append(mem)
            else:
                retained.append(mem)

        self.memories = retained
        return evaporated

    def remove(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Remove and return a specific memory by its ID.

        Args:
            memory_id: Unique identifier of the memory to remove.

        Returns:
            Optional[Dict[str, Any]]: The removed memory dict, or None if not found.
        """
        for i, mem in enumerate(self.memories):
            if mem.get("id") == memory_id:
                return self.memories.pop(i)
        return None

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self.memories)

    def __len__(self) -> int:
        return len(self.memories)


class ShortTank:
    """Short-term working memory tank with a user-configurable evaporation window (default 86400s).

    Stores dictionaries with: id, text, pressure, timestamp, created_at.
    """

    def __init__(self, ttl: float = 86400.0) -> None:
        """Initialize an empty ShortTank.

        Args:
            ttl: Evaporation TTL window in seconds (default 86400.0).
        """
        self.ttl = float(ttl)
        self.memories: List[Dict[str, Any]] = []

    def add(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Add a memory dictionary to the short tank.

        Args:
            memory: Dictionary containing id, text, pressure, timestamp, and optional created_at.

        Returns:
            Dict[str, Any]: The stored memory dictionary.
        """
        now = time.time()
        ts = float(memory.get("timestamp", now))
        created_at = float(memory.get("created_at", ts))
        record = {
            "id": memory.get("id"),
            "text": str(memory.get("text", "")),
            "pressure": float(memory.get("pressure", 0.0)),
            "timestamp": ts,
            "created_at": created_at,
        }
        if "embedding" in memory and memory["embedding"] is not None:
            record["embedding"] = memory["embedding"]

        self.memories.append(record)
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all active memories in the short tank.

        Returns:
            List[Dict[str, Any]]: List of stored memory dictionaries.
        """
        return list(self.memories)

    def evaporate(self, max_age: Optional[float] = None, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Remove memories older than the configured TTL (or custom max_age).

        Args:
            max_age: Maximum age in seconds before evaporation occurs (defaults to self.ttl).
            current_time: Reference timestamp; defaults to time.time().

        Returns:
            List[Dict[str, Any]]: The list of evaporated (removed) memories.
        """
        limit = max_age if max_age is not None else self.ttl
        now = current_time if current_time is not None else time.time()
        retained: List[Dict[str, Any]] = []
        evaporated: List[Dict[str, Any]] = []

        for mem in self.memories:
            age = now - mem.get("timestamp", now)
            if age >= limit:
                evaporated.append(mem)
            else:
                retained.append(mem)

        self.memories = retained
        return evaporated

    def remove(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Remove and return a specific memory by its ID.

        Args:
            memory_id: Unique identifier of the memory to remove.

        Returns:
            Optional[Dict[str, Any]]: The removed memory dict, or None if not found.
        """
        for i, mem in enumerate(self.memories):
            if mem.get("id") == memory_id:
                return self.memories.pop(i)
        return None

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self.memories)

    def __len__(self) -> int:
        return len(self.memories)


class LongTank:
    """Permanent encrypted storage tank.

    Stores ciphertext encrypted with AES Fernet. Plaintext is never stored in memory
    structures long-term.
    """

    def __init__(self, key: Optional[Union[str, bytes]] = None) -> None:
        """Initialize LongTank with an optional default encryption key.

        Args:
            key: Default Fernet key used for encryption and decryption.
        """
        self.key = key
        self.memories: List[Dict[str, Any]] = []

    def _resolve_key(self, key: Optional[Union[str, bytes]]) -> Union[str, bytes]:
        effective_key = key if key is not None else self.key
        if effective_key is None:
            raise ValueError("No encryption key provided or configured for LongTank.")
        return effective_key

    def add_encrypted(
        self,
        text: str,
        key: Optional[Union[str, bytes]] = None,
        memory_id: Optional[int] = None,
        pressure: float = 1.0,
        timestamp: Optional[float] = None,
        created_at: Optional[float] = None,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Encrypt and store a memory string.

        Args:
            text: Plaintext memory content to encrypt and store.
            key: Fernet key. If omitted, uses self.key.
            memory_id: Optional unique identifier.
            pressure: Memory hydraulic pressure (defaults to 1.0).
            timestamp: Optional creation timestamp.
            created_at: Optional original creation timestamp.
            embedding: Optional numerical embedding vector.
            **kwargs: Extra parameters like 'id'.

        Returns:
            Dict[str, Any]: The stored record containing encrypted ciphertext.
        """
        effective_key = self._resolve_key(key)
        encrypted_text = encrypt_data(text, effective_key)
        now = time.time()
        final_id = memory_id if memory_id is not None else kwargs.get("id", len(self.memories) + 1)
        final_time = timestamp if timestamp is not None else kwargs.get("timestamp", now)
        final_created = created_at if created_at is not None else kwargs.get("created_at", final_time)

        record: Dict[str, Any] = {
            "id": final_id,
            "text": encrypted_text,
            "pressure": float(pressure),
            "timestamp": float(final_time),
            "created_at": float(final_created),
        }
        if embedding is not None:
            record["embedding"] = embedding
        elif "embedding" in kwargs:
            record["embedding"] = kwargs["embedding"]

        self.memories.append(record)
        return record

    def get_all_decrypted(self, key: Optional[Union[str, bytes]] = None) -> List[DecryptedMemory]:
        """Retrieve and decrypt all memories stored in the tank.

        Args:
            key: Fernet key for decryption. If omitted, uses self.key.

        Returns:
            List[DecryptedMemory]: List of decrypted memory items with id, text, pressure, timestamp.
        """
        effective_key = self._resolve_key(key)
        decrypted_list: List[DecryptedMemory] = []

        for record in self.memories:
            plain_text = decrypt_data(record["text"], effective_key)
            item_data = {
                "id": record["id"],
                "text": plain_text,
                "pressure": record.get("pressure", 1.0),
                "timestamp": record.get("timestamp"),
                "created_at": record.get("created_at", record.get("timestamp")),
            }
            if "embedding" in record:
                item_data["embedding"] = record["embedding"]
            decrypted_item = DecryptedMemory(item_data)
            decrypted_list.append(decrypted_item)

        return decrypted_list

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all raw stored records with encrypted ciphertext.

        Returns:
            List[Dict[str, Any]]: List of raw encrypted records.
        """
        return list(self.memories)

    def count(self) -> int:
        """Return the number of stored memories in permanent storage.

        Returns:
            int: Number of memories.
        """
        return len(self.memories)

    def __len__(self) -> int:
        return len(self.memories)
