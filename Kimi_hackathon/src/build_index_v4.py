#!/usr/bin/env python3
"""
build_index_v4.py — Fast hybrid index builder (dense + BM25) — ACCURACY OPTIMIZED.

CHANGES:
  - Auto GPU/MPS detection (CUDA > MPS > CPU)
  - Parallel BM25 tokenization (~4-8x faster on multi-core)
  - IVF FAISS index for large corpora (>2k diseases) — faster queries
  - tqdm progress bars
  - Estimated time-to-completion
  - ~60-80% faster than v3 on CPU, even faster with GPU
  - v4 accuracy: Enhanced scoring with name/alias/HPO/marker boosts
"""

import os
import sys
import time
import pickle
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()

# Set HF token from environment (NOT hardcoded!)
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("✅ HF_TOKEN loaded from .env")
else:
    print("⚠️ HF_TOKEN not found in .env file!")
    print("   Please add HF_TOKEN to your .env file")
    print("   Get your token from: https://huggingface.co/settings/tokens")
    print("   Continuing without HF_TOKEN may cause download issues...")

# ── Add parent directory to path for imports ─────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import rag_engine ──────────────────────────────────────────────────────
from src.rag_engine_v4 import build_faiss_index

# ── Main ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("🧬 MediNova — Index Builder v4 (Accuracy Optimized)")
print("=" * 60)

# Check for GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Apple MPS (Metal) detected")
    else:
        print("ℹ️  Running on CPU (consider GPU for 5x+ speedup)")
except ImportError:
    print("ℹ️  PyTorch not found, will use CPU")

# ── Load database from data folder ──────────────────────────────────────
DB_PATH = "../data/hpo_disease_db_v3.pkl"  # ← Path relative to src/

if not os.path.exists(DB_PATH):
    print(f"\n❌ Database not found: {DB_PATH}")
    print("   Please run: python src/data_loader_v3.py first")
    sys.exit(1)

print("📂 Loading database...")
t0 = time.time()
try:
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    print(f"✅ Loaded {len(db)} diseases in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"❌ Error loading database: {e}")
    sys.exit(1)

# Estimate time
n = len(db)
est_cpu_min = n * 0.025 / 60
est_v4_min = n * 0.004 / 60

print(f"\n⏱️  Estimated time: ~{est_v4_min:.1f} min (v4 optimized) vs ~{est_cpu_min:.1f} min (v3)")
print("🔨 Building hybrid indexes (MedCPT dense + BM25)...")
print("   • GPU/MPS acceleration (if available)")
print("   • Batch size: 64 (vs 16 in v3)")
print("   • Parallel BM25 tokenization")
print("   • IVF index for faster queries on large corpora")
print("   • v4 accuracy: Name/alias/HPO/marker boosts")
print()

t1 = time.time()
try:
    build_faiss_index(db, use_ivf=(n > 2000))
except Exception as e:
    print(f"\n❌ Error building index: {e}")
    print("   Make sure you have HF_TOKEN in .env for MedCPT download")
    sys.exit(1)

elapsed = time.time() - t1

print(f"\n✅ Done in {elapsed/60:.1f} min ({elapsed:.0f}s)")
print("   Dense:  data/faiss_index_medcpt_v4.bin")
print("   Sparse: data/faiss_meta_medcpt_v4.pkl (BM25 embedded)")
print("\n🚀 Run: python evaluation_runner.py")
print("=" * 60)