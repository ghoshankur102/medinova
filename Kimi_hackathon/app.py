# ── Index loading (cached) ─────────────────────────────────────────────────
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
    
    # Show helpful error message
    st.error("❌ Index files not found!")
    st.markdown("""
    **📌 Please upload the following files to the `data/` folder:**
    
    - `faiss_index_medcpt_v4.bin` (FAISS vector index)
    - `faiss_meta_medcpt_v4.pkl` (FAISS metadata)
    
    **To generate these files locally:**
    1. Make sure you have the database file: `data/hpo_disease_db_v3.pkl`
    2. Run: `python src/build_index_v4.py`
    3. Commit the generated files and redeploy
    """)
    
    return False


def check_index_exists() -> bool:
    """Quick check for index files."""
    return (os.path.exists("data/faiss_index_medcpt_v4.bin") and
            os.path.exists("data/faiss_meta_medcpt_v4.pkl"))