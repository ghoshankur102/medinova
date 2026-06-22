#!/usr/bin/env python3
"""
app.py — MediNova Streamlit App v4
"""

import os
import sys
import streamlit as st

# ── Add src folder to Python path ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ── DEBUG: Check files ──────────────────────────────────────────────────
st.write("### 📂 File Check")
st.write("Current directory:", os.getcwd())

# Check if data folder exists
if os.path.exists("data"):
    st.write("data/ folder found!")
    st.write("Files in data/:", os.listdir("data/"))
else:
    st.write("❌ data/ folder NOT found!")

# Check if index files exist
index_path = "data/faiss_index_medcpt_v4.bin"
meta_path = "data/faiss_meta_medcpt_v4.pkl"

if os.path.exists(index_path):
    st.success(f"✅ {index_path} found!")
else:
    st.error(f"❌ {index_path} NOT found!")

if os.path.exists(meta_path):
    st.success(f"✅ {meta_path} found!")
else:
    st.error(f"❌ {meta_path} NOT found!")

# ── Try loading index ──────────────────────────────────────────────────
try:
    from rag_engine_v4 import load_faiss_index
    st.success("✅ rag_engine_v4 imported successfully!")
    
    idx, db, bm25 = load_faiss_index()
    st.success(f"✅ Index loaded! {len(db)} diseases found!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.text(traceback.format_exc())