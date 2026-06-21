#!/usr/bin/env python3
"""
build_index_v4.py — Fast hybrid index builder
"""

import os
import sys
import time
import pickle
from dotenv import load_dotenv

# ── Add parent directory to path ─────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load environment ──────────────────────────────────────────────────────
load_dotenv()

# ── Set HF token ──────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("⚠️ HF_TOKEN not found in environment")

# ── Import rag_engine ────────────────────────────────────────────────────
from src.rag_engine_v4 import build_faiss_index

# ── Main ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("🧬 MediNova — Index Builder v4")
print("=" * 60)

# ── Load database ────────────────────────────────────────────────────────
DB_PATH = "../data/hpo_disease_db_v3.pkl"

# Try different paths
possible_paths = [
    "../data/hpo_disease_db_v3.pkl",
    "data/hpo_disease_db_v3.pkl",
    "hpo_disease_db_v3.pkl",
]

for path in possible_paths:
    if os.path.exists(path):
        DB_PATH = path
        break

if not os.path.exists(DB_PATH):
    print(f"❌ Database not found: {DB_PATH}")
    print("   Please run: python src/data_loader_v3.py first")
    sys.exit(1)

print("📂 Loading database...")
t0 = time.time()
with open(DB_PATH, "rb") as f:
    db = pickle.load(f)
print(f"✅ Loaded {len(db)} diseases in {time.time()-t0:.1f}s")

# ── Build index ──────────────────────────────────────────────────────────
t1 = time.time()
build_faiss_index(db, use_ivf=(len(db) > 2000))
elapsed = time.time() - t1

print(f"\n✅ Done in {elapsed/60:.1f} min ({elapsed:.0f}s)")
print("=" * 60)