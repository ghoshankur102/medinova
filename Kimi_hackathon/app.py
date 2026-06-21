#!/usr/bin/env python3
"""
app.py — MediNova Streamlit App v4
"""

import os
import sys
import io
import json
import time
import pickle

# ── streamlit must be imported BEFORE using @st.cache_resource ────────────
import streamlit as st
from PIL import Image

# ── Add src folder to Python path ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediNova — Rare Disease Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Now you can use @st.cache_resource ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_index():
    from rag_engine_v4 import load_faiss_index
    return load_faiss_index()


def ensure_index_exists():
    """Check if index exists."""
    index_path = "data/faiss_index_medcpt_v4.bin"
    meta_path = "data/faiss_meta_medcpt_v4.pkl"
    
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        return True
    
    st.error("❌ Index files not found!")
    st.markdown("""
    **📌 Please upload the following files to the `data/` folder:**
    
    - `faiss_index_medcpt_v4.bin` (FAISS vector index)
    - `faiss_meta_medcpt_v4.pkl` (FAISS metadata)
    """)
    return False


def check_index_exists() -> bool:
    return (os.path.exists("data/faiss_index_medcpt_v4.bin") and
            os.path.exists("data/faiss_meta_medcpt_v4.pkl"))

# ── Rest of your app code ────────────────────────────────────────────────
# ... CSS, sidebar, tabs, etc.