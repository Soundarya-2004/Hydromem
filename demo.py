"""Demonstration script showcasing all core capabilities of HydroMem.

Highlights:
1. Hydraulic pressure calculation and automatic hierarchical chamber routing.
2. Multi-chamber keyword recall ranked by hydraulic pressure.
3. Automated sedimentation of frequently accessed memories into encrypted LongTank.
4. Evaporative decay and memory forget cycles.
5. Local-first AES encryption at rest in permanent storage.
"""

import sys
import time
from hydromem import HydroMem

# Ensure utf-8 output encoding for consoles that support it
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_banner(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  [*] {title}")
    print(f"{'=' * 65}")


def main() -> None:
    print_banner("HydroMem: Hydraulic-Inspired Hierarchical Memory for LLMs")

    # 1. Initialize HydroMem with custom password
    print("\n[1] Initializing HydroMem instance with local encryption...")
    mem = HydroMem(encryption_password="master-hydromem-pass")
    print("    Local AES-256 Fernet key generated securely from password.")
    print("    Initial state:", mem.stats())

    # 2. Ingesting memories with varying pressure
    print_banner("Step 1: Ingesting Memories with Hydraulic Pressure Routing")

    memories_to_add = [
        ("User prefers concise, bulleted responses in technical discussions and Python code.", 0.9),
        ("Working on project deadline: implement feature X for client demo on Friday afternoon.", 0.5),
        ("Current ambient temperature is 24 degrees Celsius with light rain outside.", 0.1),
    ]

    for text, emotion in memories_to_add:
        res = mem.remember(text, emotion=emotion)
        print(f"  • Added ID {res['id']}: [{res['tank'].upper()} TANK] Pressure={res['pressure']:.3f}")
        print(f"    Text: \"{text[:60]}...\"")

    print("\n  Current Stats after ingestion:")
    print(" ", mem.stats())

    # 3. Recall demonstration
    print_banner("Step 2: Multi-Tank Keyword Recall (Ranked by Pressure)")

    query = "project technical python"
    print(f"  Searching for query: '{query}'")
    results = mem.recall(query, top_k=3)
    for rank, item in enumerate(results, 1):
        print(f"  {rank}. [ID {item['id']}] [{item['tank'].upper()}] (Pressure: {item['pressure']:.3f})")
        print(f"     \"{item['text']}\"")

    # 4. Sedimentation demonstration
    print_banner("Step 3: Sedimentation Mechanism (Promotion to Permanent Storage)")
    print("  When a working memory in ShortTank is recalled > 3 times,")
    print("  it 'sediments' into encrypted LongTank permanent storage.")

    short_mem_query = "deadline"
    print(f"\n  Initial stats before repeated recalls: {mem.stats()}")

    for i in range(1, 5):
        recalled = mem.recall(short_mem_query)
        tank = recalled[0]["tank"] if recalled else "none"
        print(f"  Recall #{i} for '{short_mem_query}' -> Current Tank: {tank.upper()}")

    print(f"\n  Stats after 4 recalls (Sedimentation complete!):")
    print(" ", mem.stats())

    # 5. Local-first Encryption inspection
    print_banner("Step 4: Local-First AES Encryption at Rest")
    print("  Inspecting raw storage of LongTank:")
    for raw_item in mem.long_tank.get_all():
        print(f"  - Memory ID {raw_item['id']} Ciphertext: {raw_item['text'][:45]}... (Length: {len(raw_item['text'])} chars)")

    print("\n  Inspecting decrypted view (using authorized local key):")
    for dec_item in mem.long_tank.get_all_decrypted(mem.key):
        print(f"  - Memory ID {dec_item['id']} Plaintext:  \"{dec_item['text'][:65]}...\"")

    # 6. Evaporation demonstration
    print_banner("Step 5: Evaporation & Forget Cycle")
    print("  Simulating memory decay: artificially aging sensory memories beyond 300s...")

    # Age sensory tank item
    for sensory_item in mem.sensory_tank.memories:
        sensory_item["timestamp"] = time.time() - 350.0

    print("  Before forget():", mem.stats())
    mem.forget()
    print("  After forget(): ", mem.stats())
    print("  Sensory buffer cleared of expired transient noise!")

    print_banner("Demo Complete! HydroMem is ready for production use.")


if __name__ == "__main__":
    main()
