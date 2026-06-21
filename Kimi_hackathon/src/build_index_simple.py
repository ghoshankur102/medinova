#!/usr/bin/env python3
"""
build_index_simple.py - Simplified index builder for Streamlit Cloud
"""

import os
import sys
import pickle
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

print("=" * 60)
print("🧬 MediNova — Simple Index Builder")
print("=" * 60)

# Find database file
db_paths = [
    os.path.join(parent_dir, "data", "hpo_disease_db_v3.pkl"),
    "data/hpo_disease_db_v3.pkl",
    "../data/hpo_disease_db_v3.pkl",
]

db_path = None
for path in db_paths:
    if os.path.exists(path):
        db_path = path
        break

if db_path is None:
    print("❌ Database not found at:")
    for path in db_paths:
        print(f"   - {path}")
    print("")
    print("📌 Please run data_loader_v3.py locally and commit the database.")
    print("   Or upload it to Streamlit Cloud using Git LFS.")
    sys.exit(1)

print(f"📂 Loading database from: {db_path}")
try:
    with open(db_path, "rb") as f:
        db = pickle.load(f)
    print(f"✅ Loaded {len(db)} diseases")
except Exception as e:
    print(f"❌ Error loading database: {e}")
    sys.exit(1)

# Import and build
try:
    from src.rag_engine_v4 import build_faiss_index
    
    print("🔨 Building FAISS index...")
    build_faiss_index(db, use_ivf=(len(db) > 2000))
    print("✅ Index built successfully!")
    print(f"   Index saved to: {parent_dir}/data/faiss_index_medcpt_v4.bin")
    print(f"   Metadata saved to: {parent_dir}/data/faiss_meta_medcpt_v4.pkl")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure rag_engine_v4.py is in the src folder")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error building index: {e}")
    sys.exit(1)

print("=" * 60)