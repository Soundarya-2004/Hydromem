"""Security and cryptographic key derivation module for HydroMem.

Provides secure key derivation using Argon2id (if installed) or PBKDF2-HMAC (100k iterations),
salt generation and file persistence, raw key backward-compatibility, and Fernet encryption.
"""

import base64
import hashlib
import os
import secrets
from typing import Optional, Union
from cryptography.fernet import Fernet

# Optional Argon2id support
try:
    import argon2.low_level as argon2_ll
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False


DEFAULT_SALT_PATH = ".hydromem_salt"


def load_or_create_salt(salt_path: Optional[str] = None) -> bytes:
    """Load an existing 16-byte cryptographic salt from disk or generate and persist a new one.

    Args:
        salt_path: Path to the salt file. Defaults to '.hydromem_salt'.

    Returns:
        bytes: 16-byte random salt.
    """
    path = salt_path or DEFAULT_SALT_PATH
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                salt = f.read().strip()
                if len(salt) >= 16:
                    return salt[:16]
        except Exception:
            pass

    # Generate fresh 16-byte random salt
    salt = secrets.token_bytes(16)
    try:
        with open(path, "wb") as f:
            f.write(salt)
    except Exception:
        # If filesystem path is non-writable, keep salt in memory
        pass
    return salt


def generate_key(
    password: str = "hydromem",
    salt: Optional[bytes] = None,
    salt_path: Optional[str] = None,
    use_argon2: bool = True,
) -> bytes:
    """Derive a 32-byte URL-safe base64-encoded Fernet key from a password.

    Backward compatibility:
    - If password is 32+ bytes hex (>=64 hex chars), decodes and treats as raw key.
    - If password is already a valid 44-byte URL-safe base64 Fernet key, uses as-is.

    KDF hierarchy:
    1. Argon2id (if argon2-cffi installed and use_argon2=True)
    2. PBKDF2-HMAC-SHA256 (100,000 iterations standard fallback)

    Args:
        password: Password string or raw hex key.
        salt: Optional 16-byte salt. If None, loaded/created via salt_path.
        salt_path: Optional path to store/load salt.
        use_argon2: Prefer Argon2id if library is available.

    Returns:
        bytes: URL-safe base64-encoded 32-byte Fernet key.
    """
    # 1. Backward compat: Check if password is raw 32-byte hex (64 hex characters)
    if isinstance(password, str) and len(password) >= 64:
        try:
            raw_bytes = bytes.fromhex(password[:64])
            return base64.urlsafe_b64encode(raw_bytes)
        except ValueError:
            pass

    # Check if already a valid 44-character base64 Fernet key
    if isinstance(password, str) and len(password) == 44:
        try:
            decoded = base64.urlsafe_b64decode(password.encode("utf-8"))
            if len(decoded) == 32:
                return password.encode("utf-8")
        except Exception:
            pass

    # 2. Resolve 16-byte salt
    effective_salt = salt if salt is not None else load_or_create_salt(salt_path)
    if len(effective_salt) < 16:
        effective_salt = effective_salt.ljust(16, b"\x00")

    password_bytes = password.encode("utf-8")

    # 3. Derive 32-byte key via Argon2id if available
    if use_argon2 and ARGON2_AVAILABLE:
        raw_key = argon2_ll.hash_secret_raw(
            secret=password_bytes,
            salt=effective_salt[:16],
            time_cost=2,
            memory_cost=65536,
            parallelism=1,
            hash_len=32,
            type=argon2_ll.Type.ID,
        )
        return base64.urlsafe_b64encode(raw_key)

    # 4. Standard PBKDF2-HMAC-SHA256 with 100,000 iterations
    raw_key = hashlib.pbkdf2_hmac(
        "sha256",
        password_bytes,
        effective_salt,
        iterations=100000,
        dklen=32,
    )
    return base64.urlsafe_b64encode(raw_key)


def _ensure_bytes(key: Union[str, bytes]) -> bytes:
    """Ensure key is in bytes format suitable for Fernet initialization."""
    if isinstance(key, str):
        return key.encode("utf-8")
    return key


def encrypt_data(data: str, key: Union[str, bytes]) -> str:
    """Encrypt a plaintext string using a Fernet key.

    Args:
        data: Plaintext string to encrypt.
        key: Fernet key (bytes or string).

    Returns:
        str: Encrypted ciphertext as a UTF-8 string.
    """
    fernet = Fernet(_ensure_bytes(key))
    encrypted_bytes = fernet.encrypt(data.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")


def decrypt_data(encrypted: str, key: Union[str, bytes]) -> str:
    """Decrypt an encrypted ciphertext string back to plaintext using a Fernet key.

    Args:
        encrypted: Encrypted ciphertext string.
        key: Fernet key (bytes or string).

    Returns:
        str: Original decrypted plaintext string.
    """
    fernet = Fernet(_ensure_bytes(key))
    decrypted_bytes = fernet.decrypt(encrypted.encode("utf-8"))
    return decrypted_bytes.decode("utf-8")
