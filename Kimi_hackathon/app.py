#!/usr/bin/env python3
"""
app.py — MediNova Streamlit App v4
Rare disease diagnosis support with hybrid RAG + medical image analysis.
"""

import os
import sys
import io
import json
import time
import pickle
import streamlit as st
from PIL import Image


# ── DEBUG: Find index files ──────────────────────────────────────────────────
import subprocess

st.write("### 🔍 Finding Index Files")

# Search for the index file
result = subprocess.run(["find", "/mount", "-name", "faiss_index_medcpt_v4.bin", "-type", "f"], 
                        capture_output=True, text=True)
st.write("📍 Index file locations:")
st.code(result.stdout)

# List the entire project structure
result = subprocess.run(["ls", "-la", "/mount/src/medinova/Kimi_hackathon/"], 
                        capture_output=True, text=True)
st.write("📂 Kimi_hackathon/ contents:")
st.code(result.stdout)

# ── DEBUG: Check paths ──────────────────────────────────────────────────
import os
st.write("### 🔍 Path Debug")

# Check current directory
st.write(f"Current directory: {os.getcwd()}")

# Check if data folder exists
st.write(f"data/ exists: {os.path.exists('data')}")

# List files in data/
if os.path.exists("data"):
    st.write(f"Files in data/: {os.listdir('data')}")

# Check rag_engine_v4 paths
try:
    from rag_engine_v4 import FAISS_INDEX_PATH, FAISS_META_PATH
    st.write(f"FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    st.write(f"FAISS_INDEX_PATH exists: {os.path.exists(FAISS_INDEX_PATH)}")
    st.write(f"FAISS_META_PATH: {FAISS_META_PATH}")
    st.write(f"FAISS_META_PATH exists: {os.path.exists(FAISS_META_PATH)}")
except Exception as e:
    st.write(f"Error importing from rag_engine_v4: {e}")

# ── Add src folder to Python path ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediNova — Rare Disease Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { color: #a8b2d8; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Result cards */
    .disease-card {
        background: #1e2235;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .disease-card:hover { border-color: #e94560; }
    .disease-card h3 { color: #e2e8f0; margin: 0 0 0.5rem; font-size: 1.1rem; }
    .disease-card .score { color: #e94560; font-size: 0.8rem; font-weight: 600; }
    .disease-card .orpha { color: #64748b; font-size: 0.78rem; }
    .disease-card .symptoms { color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem; }
    .disease-card .badge {
        display: inline-block; background: #0f3460; color: #7dd3fc;
        border-radius: 6px; padding: 2px 8px; font-size: 0.73rem;
        margin: 2px; border: 1px solid #1e4080;
    }
    .disease-card .badge-red {
        background: #3b0a0a; color: #fca5a5; border-color: #7f1d1d;
    }

    /* Image analysis */
    .img-finding {
        background: #0f2235;
        border-left: 3px solid #3b82f6;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .disclaimer-box {
        background: #2d1b00;
        border: 1px solid #b45309;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #fbbf24;
        font-size: 0.82rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #1e2235;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2d3561;
    }

    /* Metric badges */
    .stMetric { background: #1e2235 !important; border-radius: 10px !important; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Index loading (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)

@st.cache_resource(show_spinner=False)
def load_index():
    """Load index with fallback paths."""
    try:
        from rag_engine_v4 import load_faiss_index
        return load_faiss_index()
    except Exception as e:
        st.error(f"Error loading index: {e}")
        # Try direct import
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from rag_engine_v4 import load_faiss_index
            return load_faiss_index()
        except Exception as e2:
            st.error(f"Fallback error: {e2}")
            raise

def check_index_exists() -> bool:
    """Check if index exists."""
    # The actual location on Streamlit Cloud
    possible_paths = [
        "data/faiss_index_medcpt_v4.bin",
        "../data/faiss_index_medcpt_v4.bin",
        "Kimi_hackathon/data/faiss_index_medcpt_v4.bin",
        "/mount/src/medinova/Kimi_hackathon/data/faiss_index_medcpt_v4.bin",
        "/mount/src/medinova/data/faiss_index_medcpt_v4.bin",
        "/mount/src/medinova/faiss_index_medcpt_v4.bin",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.write(f"✅ Found index at: {path}")
            return True
    
    st.write("❌ Index not found in any location")
    return False

# ── Rendering helpers ─────────────────────────────────────────────────────
def render_disease_card(disease: dict, rank: int):
    score = disease.get("_boosted_score", disease.get("_fused_score", 0))
    name = disease.get("name", "Unknown")
    orpha_id = disease.get("orpha_id", "")
    definition = (disease.get("definition") or "")[:200]
    symptoms = disease.get("display_symptoms", [])[:8]
    distinguishing = disease.get("distinguishing_features", "")
    gender_ctx = disease.get("gender_context", "")
    inheritance = disease.get("inheritance_context", "")

    sym_html = "".join(f'<span class="badge">{s}</span>' for s in symptoms)

    extra_badges = ""
    if gender_ctx:
        extra_badges += f'<span class="badge badge-red">🔬 {gender_ctx}</span>'
    if inheritance:
        extra_badges += f'<span class="badge badge-red">🧬 {inheritance}</span>'

    st.markdown(f"""
    <div class="disease-card">
      <div style="display:flex; justify-content:space-between; align-items:start;">
        <h3>#{rank} — {name}</h3>
        <div>
          <span class="score">Score: {score:.3f}</span><br>
          <span class="orpha">{f"ORPHA:{orpha_id}" if orpha_id else ""}</span>
        </div>
      </div>
      {f'<p class="symptoms" style="margin-bottom:0.3rem;color:#64748b;font-style:italic;">{definition}...</p>' if definition else ''}
      <div style="margin-top:0.5rem;">{sym_html}</div>
      {f'<div style="margin-top:0.5rem;">{extra_badges}</div>' if extra_badges else ''}
      {f'<p class="symptoms" style="margin-top:0.5rem;color:#7dd3fc;">🔍 {distinguishing}</p>' if distinguishing else ''}
    </div>
    """, unsafe_allow_html=True)


def render_image_analysis(analysis: dict, suggested_symptoms: list):
    modality = analysis.get("modality", "Unknown")
    region = analysis.get("body_region", "Unknown")
    findings = analysis.get("key_findings", [])
    diff_pointers = analysis.get("differential_pointers", "")
    quality = analysis.get("image_quality", "unknown")

    col1, col2, col3 = st.columns(3)
    col1.metric("Modality", modality)
    col2.metric("Body Region", region.title())
    col3.metric("Image Quality", quality.title())

    st.markdown("#### 🔍 Key Findings")
    for f in findings:
        st.markdown(f'<div class="img-finding">• {f}</div>', unsafe_allow_html=True)

    if diff_pointers:
        st.markdown("#### 🧬 Differential Pointers")
        st.info(diff_pointers)

    if suggested_symptoms:
        st.markdown("#### 🏷️ Suggested Symptoms for Search")
        st.code(", ".join(suggested_symptoms), language=None)

    st.markdown(f"""
    <div class="disclaimer-box">
      ⚠️ <strong>Disclaimer:</strong> {analysis.get("disclaimer", "For educational support only. Requires radiologist review.")}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 MediNova")
    st.markdown("Rare disease diagnosis support")
    st.divider()

    st.markdown("#### ⚙️ Search Settings")
    top_k = st.slider("Results to show", 3, 15, 5)
    retrieval_k = st.slider("Candidate pool size", 20, 100, 50, step=10)
    fusion_method = st.selectbox("Score fusion", ["weighted", "rrf"])
    alpha = st.slider("Dense weight (α)", 0.3, 0.9, 0.6, step=0.05,
                      help="Higher = more weight on semantic (dense) retrieval")

    st.divider()
    st.markdown("#### 📊 System Status")
    if check_index_exists():
        st.success("✅ Index ready")
        try:
            from rag_engine_v4 import FAISS_INDEX_PATH
            size_mb = os.path.getsize(FAISS_INDEX_PATH) / 1e6
            st.caption(f"Index size: {size_mb:.1f} MB")
        except Exception:
            pass
    else:
        st.warning("⚠️ Index not found")
        st.caption("Run: `python build_index_v4.py`")

    st.divider()
    st.markdown("""
    <small style="color:#64748b;">
    Data: Orphanet XML<br>
    Model: ncbi/MedCPT<br>
    Retrieval: Dense + BM25<br>
    v4 — Optimized
    </small>
    """, unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧬 MediNova — Rare Disease Support</h1>
  <p>Hybrid semantic + keyword search over 10,000+ rare diseases · Medical image analysis via AI</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Symptom Search", "🩻 Medical Image Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SYMPTOM SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Describe the clinical presentation")

    col_input, col_hints = st.columns([2, 1])

    with col_input:
        query = st.text_area(
            "Enter symptoms, HPO IDs, or clinical findings",
            placeholder="e.g. progressive muscle weakness, elevated creatine kinase, pseudohypertrophy, X-linked\n\nor: drooping eyelids, double vision, worsens throughout day",
            height=130,
        )

    with col_hints:
        st.markdown("**💡 Query tips**")
        st.markdown("""
        - Use colloquial terms: *drooping eyelids*, *salty sweat*
        - Include HPO IDs: `HP:0001638`
        - Mention gender: *girls*, *boys*, *male*, *female*
        - Add inheritance: *X-linked*, *autosomal recessive*
        - Specify onset: *infantile*, *neonatal*, *adult*
        """)

    # Example queries
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(4)
    examples = [
        ("Myasthenia Gravis", "drooping eyelids double vision difficulty swallowing worsens throughout day"),
        ("Duchenne MD", "progressive muscle weakness calf pseudohypertrophy elevated creatine kinase boys"),
        ("Wilson Disease", "copper accumulation liver disease personality changes Kayser-Fleischer rings"),
        ("Angelman Syndrome", "happy demeanor absent speech seizures girls chromosome 15"),
    ]
    for i, (label, ex_query) in enumerate(examples):
        if ex_cols[i].button(label, use_container_width=True):
            query = ex_query
            st.rerun()

    search_col, _ = st.columns([1, 3])
    search_clicked = search_col.button("🔍 Search Diseases", type="primary", use_container_width=True)

    if search_clicked or (query and st.session_state.get("last_query") != query):
        if not query.strip():
            st.warning("Please enter at least one symptom or finding.")
        elif not check_index_exists():
            st.error("❌ Index not found. Please run `python build_index_v4.py` first.")
        else:
            st.session_state["last_query"] = query
            with st.spinner("🔎 Searching across 10,000+ rare diseases..."):
                try:
                    from rag_engine_v4 import retrieve_diseases
                    t0 = time.time()
                    index, db, bm25 = load_index()
                    results = retrieve_diseases(
                        query, index, db, bm25,
                        top_k=top_k,
                        retrieval_k=retrieval_k,
                        fusion_method=fusion_method,
                        alpha=alpha,
                    )
                    elapsed = time.time() - t0
                    st.session_state["results"] = results
                    st.session_state["query_time"] = elapsed
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.exception(e)

    # Display results
    if "results" in st.session_state:
        results = st.session_state["results"]
        elapsed = st.session_state.get("query_time", 0)

        st.markdown(f"---\n**{len(results)} results** · Query time: `{elapsed:.2f}s`")

        if not results:
            st.info("No results found. Try broadening your query.")
        else:
            for i, disease in enumerate(results, 1):
                render_disease_card(disease, i)

                with st.expander(f"📋 Full details — {disease.get('name', '')}", expanded=False):
                    detail_cols = st.columns(2)
                    with detail_cols[0]:
                        st.markdown("**All Symptoms**")
                        all_syms = disease.get("display_symptoms", [])
                        for s in all_syms:
                            st.markdown(f"• {s}")
                    with detail_cols[1]:
                        st.markdown("**HPO IDs**")
                        hpo_ids = disease.get("hpo_ids", [])[:15]
                        for h in hpo_ids:
                            st.markdown(f"`{h}`")

                    if disease.get("definition"):
                        st.markdown("**Definition**")
                        st.markdown(disease["definition"])

                    # Scores debug
                    with st.expander("🔢 Score breakdown", expanded=False):
                        score_data = {
                            "Boosted score": disease.get("_boosted_score", "N/A"),
                            "Fused score": disease.get("_fused_score", "N/A"),
                            "Dense rank": disease.get("_dense_rank", "N/A"),
                            "Sparse rank": disease.get("_sparse_rank", "N/A"),
                        }
                        st.json(score_data)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MEDICAL IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload a Medical Image for AI Analysis")

    st.markdown("""
    <div class="disclaimer-box" style="margin-bottom:1rem;">
      ⚠️ <strong>Educational Use Only</strong> — This tool is for rare disease research support only.
      It does NOT replace a qualified radiologist or physician. Never use this for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

    img_col, ctrl_col = st.columns([2, 1])

    with ctrl_col:
        st.markdown("#### Supported formats")
        st.markdown("""
        - **X-ray** (chest, bone, abdomen)
        - **CT scan** (axial slices)
        - **MRI** (brain, spine, body)
        - **Ultrasound**
        - Other medical images
        
        **File types:** JPG, PNG, WEBP  
        **Max size:** 10 MB
        """)
        auto_search = st.checkbox("Auto-search diseases after analysis", value=True)

    with img_col:
        uploaded_file = st.file_uploader(
            "Upload medical image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    if uploaded_file is not None:
        analyze_clicked = st.button("🩻 Analyze Image", type="primary", use_container_width=False)

        if analyze_clicked:
            # Determine media type
            ext = uploaded_file.name.lower().split(".")[-1]
            media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                              "png": "image/png", "webp": "image/webp"}
            media_type = media_type_map.get(ext, "image/jpeg")

            with st.spinner("🔬 Analyzing medical image with AI..."):
                try:
                    from rag_engine_v4 import analyze_medical_image
                    analysis = analyze_medical_image(image_bytes, media_type)
                    st.session_state["img_analysis"] = analysis
                    suggested = analysis.get("suggested_symptoms", [])

                    st.success("✅ Analysis complete!")
                    render_image_analysis(analysis, suggested)

                    # Auto-search if enabled
                    if auto_search and suggested and check_index_exists():
                        auto_query = ", ".join(suggested[:6])
                        st.markdown("---")
                        st.markdown(f"#### 🔍 Auto-searching diseases for: `{auto_query}`")

                        with st.spinner("Searching diseases..."):
                            from rag_engine_v4 import retrieve_diseases
                            index, db, bm25 = load_index()
                            img_results = retrieve_diseases(
                                auto_query, index, db, bm25,
                                top_k=5, retrieval_k=50
                            )

                        if img_results:
                            st.markdown(f"**Top {len(img_results)} diseases based on imaging findings:**")
                            for i, disease in enumerate(img_results, 1):
                                render_disease_card(disease, i)
                        else:
                            st.info("No disease matches found from image findings.")

                except json.JSONDecodeError as e:
                    st.error(f"JSON parse error from Claude: {e}")
                    st.caption("Claude returned non-JSON. Check API key and try again.")
                except Exception as e:
                    st.error(f"Image analysis error: {e}")
                    if "ANTHROPIC_API_KEY" in str(e) or "api_key" in str(e).lower():
                        st.warning("Set ANTHROPIC_API_KEY environment variable to enable image analysis.")
                    else:
                        st.exception(e)

    else:
        # Demo mode placeholder
        st.markdown("""
        <div style="
            background: #1e2235;
            border: 2px dashed #2d3561;
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            color: #64748b;
            margin-top: 1rem;
        ">
          <div style="font-size:3rem;">🩻</div>
          <h3 style="color:#94a3b8; margin:0.5rem 0;">Upload an X-ray, CT, or MRI</h3>
          <p>AI will identify key findings and auto-suggest diseases to search</p>
        </div>
        """, unsafe_allow_html=True)