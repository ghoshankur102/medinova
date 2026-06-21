"""
evaluation_config.py - Configuration for MediNova Evaluation Suite
Uses MedCPT for retrieval (existing) and Gemini Pro for image analysis
"""

import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data_loader_v3 import NAME_ALIASES, DISTINGUISHING_MARKERS

DB_PATH = "hpo_disease_db_v3.pkl"

@dataclass
class ModelConfig:
    """Configuration for model APIs"""
    # Gemini Pro for computer vision / image analysis
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.2
    
    # MedCPT (existing) for embeddings - no API key needed
    medcpt_model: str = "ncbi/MedCPT-Query-Encoder"


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    test_size: int = 150  # 100-150 test queries
    top_k: int = 5
    retrieval_k: int = 50
    fusion_method: str = "weighted"
    alpha: float = 0.6
    benchmark_queries: int = 100
    
    # Output paths
    output_dir: str = "evaluation_output"
    report_path: str = "evaluation_output/eval_report.json"
    html_report: str = "evaluation_output/eval_report.html"
    csv_results: str = "evaluation_output/results.csv"
    
    # Categories to test (from your slide)
    categories: List[str] = field(default_factory=lambda: [
        "classic_symptoms",
        "biochemical_markers",
        "gender_specific",
        "vague_single_symptom",
        "colloquial_terms",
        "name_aliases",
        "distinguishing_features",
        "gender_filtering",
        "inheritance_terms",
        "onset_terms"
    ])


# Known disease names for verification (from your database)
KNOWN_DISEASES: List[str] = [
    "Myasthenia gravis",
    "Duchenne muscular dystrophy",
    "Wilson disease",
    "Angelman syndrome",
    "Phenylketonuria",
    "Marfan syndrome",
    "Huntington disease",
    "Cystic fibrosis",
    "Neurofibromatosis type 1",
    "Tuberous sclerosis",
    "Ehlers-Danlos syndrome",
    "Pompe disease",
    "Fabry disease",
    "Gaucher disease",
    "Prader-Willi syndrome",
    "Rett syndrome",
    "Turner syndrome",
    "Klinefelter syndrome",
    "Williams syndrome",
    "Fragile X syndrome"
]


# ── Hand-authored clinical context (not derivable from Orphanet HPO data) ──
# Inheritance pattern per disease. Myasthenia gravis is intentionally
# omitted — it's autoimmune/acquired, not a classic Mendelian disorder.
INHERITANCE_PATTERNS = {
    "Duchenne muscular dystrophy": "X-linked recessive",
    "Wilson disease": "autosomal recessive",
    "Angelman syndrome": "chromosome 15q11-q13 deletion, often de novo",
    "Phenylketonuria": "autosomal recessive",
    "Marfan syndrome": "autosomal dominant",
    "Huntington disease": "autosomal dominant with anticipation",
    "Cystic fibrosis": "autosomal recessive",
    "Neurofibromatosis type 1": "autosomal dominant",
    "Tuberous sclerosis": "autosomal dominant",
    "Ehlers-Danlos syndrome": "autosomal dominant in most subtypes",
    "Pompe disease": "autosomal recessive",
    "Fabry disease": "X-linked",
    "Gaucher disease": "autosomal recessive",
    "Prader-Willi syndrome": "paternal 15q11-q13 deletion, usually de novo",
    "Rett syndrome": "X-linked dominant, MECP2, usually de novo",
    "Turner syndrome": "sporadic monosomy X, not inherited",
    "Klinefelter syndrome": "sporadic 47XXY, not inherited",
    "Williams syndrome": "7q11.23 microdeletion, usually de novo",
    "Fragile X syndrome": "X-linked dominant with anticipation",
}

# Typical age of onset per disease.
ONSET_PATTERNS = {
    "Myasthenia gravis": "adult onset, can also occur in childhood",
    "Duchenne muscular dystrophy": "early childhood onset, before age 5",
    "Wilson disease": "adolescent or young adult onset",
    "Angelman syndrome": "infantile onset, apparent by 6-12 months",
    "Phenylketonuria": "neonatal onset, found on newborn screening",
    "Marfan syndrome": "present from birth, often diagnosed in childhood or adolescence",
    "Huntington disease": "adult onset, typically in the 30s-40s",
    "Cystic fibrosis": "neonatal or infantile onset",
    "Neurofibromatosis type 1": "childhood onset",
    "Tuberous sclerosis": "infantile onset, seizures often in the first year",
    "Ehlers-Danlos syndrome": "early childhood onset",
    "Pompe disease": "infantile-onset or late-onset forms",
    "Fabry disease": "childhood onset of pain, organ involvement in adulthood",
    "Gaucher disease": "variable onset, from infancy to adulthood",
    "Prader-Willi syndrome": "neonatal hypotonia with childhood hyperphagia onset",
    "Rett syndrome": "regression after normal early development, around 6-18 months",
    "Turner syndrome": "present from birth, often diagnosed in childhood or puberty",
    "Klinefelter syndrome": "often diagnosed at puberty or in adult infertility workup",
    "Williams syndrome": "infantile onset, distinctive facial features from infancy",
    "Fragile X syndrome": "developmental delay apparent in early childhood",
}

# Lay-language descriptions, to test retrieval on colloquial phrasing.
COLLOQUIAL_PHRASES = {
    "Myasthenia gravis": "drooping eyelids and double vision that gets worse through the day",
    "Duchenne muscular dystrophy": "boy uses his hands to climb up his own legs to stand, big calves",
    "Wilson disease": "personality changes, tremor, trouble speaking, golden-brown ring around the eyes",
    "Angelman syndrome": "always smiling and laughing, doesn't talk, jerky movements",
    "Phenylketonuria": "baby smells musty, fair skin and hair, not developing normally",
    "Marfan syndrome": "very tall and thin with long fingers and an eye lens problem",
    "Huntington disease": "jerky uncontrollable movements and mood swings, runs in the family",
    "Cystic fibrosis": "salty-tasting skin, chronic cough, greasy stools, recurrent lung infections",
    "Neurofibromatosis type 1": "many coffee-colored skin spots and soft bumps under the skin",
    "Tuberous sclerosis": "facial red bumps, white skin patches, seizures since infancy",
    "Ehlers-Danlos syndrome": "joints bend too far, stretchy skin, bruises very easily",
    "Pompe disease": "floppy baby with an enlarged heart and trouble breathing",
    "Fabry disease": "burning pain in hands and feet, small dark red skin spots, kidney problems",
    "Gaucher disease": "easy bruising, big belly from an enlarged spleen, bone pain",
    "Prader-Willi syndrome": "floppy baby who later becomes insatiably hungry and gains weight",
    "Rett syndrome": "girl who loses language and hand use, repetitive hand wringing",
    "Turner syndrome": "short girl with a webbed neck and periods that never start",
    "Klinefelter syndrome": "tall man with low testosterone, enlarged breast tissue, infertility",
    "Williams syndrome": "elfin facial features, very friendly personality, heart murmur",
    "Fragile X syndrome": "boy with a learning disability, long face, large ears, anxious behavior",
}


def _load_disease_db():
    if not os.path.exists(DB_PATH):
        return None
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def _find_known_disease_records(db):
    """Match KNOWN_DISEASES names to entries in the built database."""
    by_name = {d["name"]: d for d in db}
    return {name: by_name[name] for name in KNOWN_DISEASES if name in by_name}


def generate_test_dataset(n: int = 150, seed: int = 42):
    """
    Build a labeled test set for evaluation_runner_v2.py.

    Combines real disease data (symptoms, orpha codes, distinguishing
    features, gender context) from hpo_disease_db_v3.pkl with curated
    alias/marker tables and the inheritance/onset/colloquial tables above
    to produce queries across all EvalConfig.categories.

    Returns:
        (test_queries, metadata)
        test_queries: list of {"query", "expected_disease", "orpha_code", "category"}
        metadata: dict describing how the set was built
    """
    random.seed(seed)
    db = _load_disease_db()
    if db is None:
        raise FileNotFoundError(
            f"{DB_PATH} not found in the current directory. "
            "Run `python data_loader_v3.py` (and `python build_index_v4.py`) first."
        )

    records = _find_known_disease_records(db)
    missing = [d for d in KNOWN_DISEASES if d not in records]
    if missing:
        print(f"⚠️  {len(missing)} known disease(s) not found in {DB_PATH}, skipping: {missing}")

    queries = []

    for name, rec in records.items():
        orpha = rec.get("orpha_code", "")
        symptoms = rec.get("display_symptoms") or []
        distinguishing = (rec.get("distinguishing_features") or "").replace(
            "Distinguishing features: ", "").rstrip(".")
        gender_ctx = rec.get("gender_context") or ""
        alias_text = NAME_ALIASES.get(name, "")
        markers = DISTINGUISHING_MARKERS.get(name, [])

        def add(query_text, category):
            q = " ".join(query_text.split())
            if q:
                queries.append({
                    "query": q,
                    "expected_disease": name,
                    "orpha_code": orpha,
                    "category": category,
                })

        if len(symptoms) >= 3:
            add(", ".join(symptoms[:3]), "classic_symptoms")

        if symptoms:
            add(random.choice(symptoms[:min(5, len(symptoms))]), "vague_single_symptom")

        if markers:
            add(", ".join(markers[:3]), "biochemical_markers")

        if distinguishing:
            add(distinguishing, "distinguishing_features")

        if gender_ctx:
            add(gender_ctx, "gender_specific")
            if symptoms:
                pronoun_group = "girls" if "female" in gender_ctx.lower() else "boys"
                add(f"{symptoms[0]} in {pronoun_group}", "gender_filtering")

        if alias_text:
            add(alias_text, "name_aliases")

        if name in COLLOQUIAL_PHRASES:
            add(COLLOQUIAL_PHRASES[name], "colloquial_terms")

        if name in INHERITANCE_PATTERNS:
            anchor = symptoms[0] if symptoms else name.lower()
            add(f"{INHERITANCE_PATTERNS[name]} condition with {anchor}", "inheritance_terms")

        if name in ONSET_PATTERNS:
            anchor = symptoms[0] if symptoms else name.lower()
            add(f"{ONSET_PATTERNS[name]}, {anchor}", "onset_terms")

    random.shuffle(queries)

    if len(queries) > n:
        queries = queries[:n]
    elif len(queries) < n and queries:
        pool = queries.copy()
        i = 0
        while len(queries) < n:
            queries.append(pool[i % len(pool)])
            i += 1

    metadata = {
        "total_queries": len(queries),
        "diseases_covered": len(records),
        "diseases_missing_from_db": missing,
        "categories": sorted(set(q["category"] for q in queries)),
        "seed": seed,
    }

    return queries, metadata