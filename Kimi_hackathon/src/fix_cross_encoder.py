#!/usr/bin/env python3
"""Quick fix: Disable cross-encoder reranking which overrides direct matches"""

with open("rag_engine_v4.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Change default use_cross_encoder from True to False
content = content.replace(
    'use_cross_encoder: bool = True,',
    'use_cross_encoder: bool = False,'
)

# Fix 2: Also update the call in the test section
content = content.replace(
    'results = retrieve_diseases(q, idx, db, bm25, top_k=3)',
    'results = retrieve_diseases(q, idx, db, bm25, top_k=3, use_cross_encoder=False)'
)

with open("rag_engine_v4.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Cross-encoder disabled. Direct match scores will now determine ranking.")
print("Run: python evaluate_rag_v4_realistic.py")
