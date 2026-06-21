"""
data_loader_v2.py — Enhanced Orphanet HPO data loader with distinguishing features.
Run: python data_loader_v3.py

CHANGES vs v2:
  - Distinguishing feature extraction for diseases with overlapping generic symptoms
  - Query-aware synonym expansion (clinical ↔ colloquial mappings)
  - Name alias injection with biomedical context anchors
  - Improved frequency-based symptom weighting
  - Reduced name repetition noise in searchable_text
  - Added biochemical marker and gender-specific signals
"""

import os
import json
import pickle
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import Counter

PRODUCT4_URL = "https://www.orphadata.com/data/xml/en_product4.xml"
PRODUCT1_URL = "https://www.orphadata.com/data/xml/en_product1.xml"
CACHE_FILE = "hpo_disease_db_v3.pkl"
CACHE_METADATA = "hpo_cache_metadata_v3.json"

MAX_DEFINITION_CHARS = 150
MAX_SYMPTOMS_IN_TEXT = 30

# ── Symptom synonym expansion ──────────────────────────────────────────────
SYMPTOM_SYNONYMS = {
    "muscle weakness": "muscle weakness hypotonia",
    "intellectual disability": "intellectual disability cognitive impairment mental retardation",
    "seizures": "seizures epilepsy convulsions",
    "short stature": "short stature growth failure dwarfism",
    "obesity": "obesity overweight hyperphagia",
    "tall stature": "tall stature tall height",
    "cardiomegaly": "cardiomegaly enlarged heart cardiomyopathy",
    "hepatomegaly": "hepatomegaly enlarged liver",
    "splenomegaly": "splenomegaly enlarged spleen",
    "lymphadenopathy": "lymphadenopathy enlarged lymph nodes",
    "developmental delay": "developmental delay delayed milestones",
    "failure to thrive": "failure to thrive poor growth growth failure",
    "respiratory failure": "respiratory failure breathing difficulty",
    "hearing loss": "hearing loss deafness hypoacusis",
    "vision loss": "vision loss blindness visual impairment",
    "ataxia": "ataxia incoordination gait disturbance",
    "tremor": "tremor shaking",
    "chorea": "chorea involuntary movements",
    "dysarthria": "dysarthria speech difficulty",
    "dysphagia": "dysphagia swallowing difficulty",
}

# ── Name aliases with biomedical context anchors ───────────────────────────
NAME_ALIASES = {
    "Myasthenia gravis": "myasthenia gravis autoimmune neuromuscular junction acetylcholine receptor",
    "Marfan syndrome": "marfan syndrome connective tissue fibrillin aortic root dilation",
    "Wilson disease": "wilson disease hepatolenticular copper accumulation ceruloplasmin",
    "Huntington disease": "huntington disease huntington chorea CAG repeat striatal degeneration",
    "Phenylketonuria": "phenylketonuria PKU phenylalanine hydroxylase deficiency hyperphenylalaninemia",
    "Cystic fibrosis": "cystic fibrosis CFTR mucoviscidosis pancreatic insufficiency chloride channel",
    "Duchenne muscular dystrophy": "duchenne muscular dystrophy DMD dystrophin pseudohypertrophy Gowers",
    "Neurofibromatosis type 1": "neurofibromatosis type 1 NF1 von recklinghausen RAS pathway cafe au lait",
    "Tuberous sclerosis": "tuberous sclerosis complex TSC hamartomas mTOR cortical tubers",
    "Ehlers-Danlos syndrome": "ehlers-danlos syndrome EDS collagen hypermobility skin fragility",
    "Pompe disease": "pompe disease glycogen storage type 2 acid alpha-glucosidase GAA cardiomegaly",
    "Fabry disease": "fabry disease alpha-galactosidase A GL3 globotriaosylceramide angiokeratoma",
    "Gaucher disease": "gaucher disease glucocerebrosidase GBA glucosylceramide lipidosis splenomegaly",
    "Prader-Willi syndrome": "prader-willi syndrome PWS imprinting chromosome 15q11-q13 hyperphagia",
    "Angelman syndrome": "angelman syndrome happy puppet UBE3A chromosome 15q11-q13 ataxia",
    "Turner syndrome": "turner syndrome 45X monosomy X gonadal dysgenesis webbed neck",
    "Klinefelter syndrome": "klinefelter syndrome 47XXY hypogonadism gynecomastia tall stature",
    "Williams syndrome": "williams syndrome williams-beuren 7q11.23 elfin facies supravalvular aortic stenosis",
    "Rett syndrome": "rett syndrome MECP2 CDKL5 hand wringing stereotypies regression girls",
    "Fragile X syndrome": "fragile X syndrome FMR1 CGG repeat macroorchidism autism features",
}

# ── Distinguishing biochemical/clinical markers ────────────────────────────
DISTINGUISHING_MARKERS = {
    "Phenylketonuria": ["phenylalanine", "musty odor", "mousy odor", "fair skin", "eczema", "light hair"],
    "Wilson disease": ["copper", "Kayser-Fleischer", "ceruloplasmin", "hepatolenticular"],
    "Pompe disease": ["glycogen storage", "acid alpha-glucosidase", "GAA", "exercise intolerance"],
    "Fabry disease": ["alpha-galactosidase", "GL3", "globotriaosylceramide", "acroparesthesia", "angiokeratoma"],
    "Gaucher disease": ["glucocerebrosidase", "GBA", "glucosylceramide", "bone crisis", "splenomegaly hepatomegaly"],
    "Prader-Willi syndrome": ["hyperphagia", "infantile hypotonia", "failure to thrive then obesity", "hypogonadism"],
    "Angelman syndrome": ["happy puppet", "UBE3A", "absent speech", "paroxysmal laughter", "hand flapping"],
    "Rett syndrome": ["MECP2", "hand wringing", "regression", "stereotypies", "breathing abnormalities", "girls"],
    "Fragile X syndrome": ["FMR1", "CGG repeat", "macroorchidism", "prominent jaw", "large ears"],
    "Williams syndrome": ["elfin facies", "7q11.23", "williams-beuren", "friendly personality", "hypercalcemia"],
    "Myasthenia gravis": ["acetylcholine receptor", "autoimmune", "diurnal fluctuation", "fatigable weakness"],
    "Marfan syndrome": ["fibrillin", "aortic root dilation", "ectopia lentis", "arachnodactyly"],
    "Huntington disease": ["CAG repeat", "striatal", "chorea", "behavioral changes", "anticipation"],
    "Cystic fibrosis": ["CFTR", "thick mucus", "salty sweat", "pancreatic insufficiency", "chronic lung infection"],
    "Duchenne muscular dystrophy": ["dystrophin", "pseudohypertrophy", "Gowers sign", "creatine kinase CK"],
    "Neurofibromatosis type 1": ["NF1", "cafe au lait spots", "neurofibromas", "Lisch nodules", "optic glioma"],
    "Tuberous sclerosis": ["TSC", "hamartomas", "mTOR", "facial angiofibroma", "ash leaf spot", "renal angiomyolipoma"],
    "Ehlers-Danlos syndrome": ["collagen", "joint hypermobility", "skin hyperextensibility", "easy bruising", "fragile skin"],
    "Turner syndrome": ["45X", "monosomy X", "webbed neck", "lymphedema", "primary amenorrhea", "bicuspid aortic valve"],
    "Klinefelter syndrome": ["47XXY", "gynecomastia", "small testes", "infertility", "learning difficulties"],
}


def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"   ✅ {dest} already exists, skipping download.")
        return
    print(f"   ⬇️  Downloading {dest} (~46MB)...")

    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        print(f"\r   Progress: {min(pct, 100)}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r   ✅ Downloaded {dest}            ")


def parse_product1(filepath: str) -> dict:
    print(f"📂 Parsing {filepath}...")
    tree = ET.parse(filepath)
    root = tree.getroot()
    diseases = {}
    for disorder in root.iter("Disorder"):
        code = disorder.findtext("OrphaCode")
        name = disorder.findtext("Name")
        summary = disorder.findtext("SummaryInformation/TextSection/Contents")
        if code and name:
            diseases[code] = {
                "orpha_code": code,
                "name": name,
                "definition": summary or "",
                "symptoms": [],
                "hpo_terms": [],
                "hpo_ids": [],
                "symptom_freqs": [],
            }
    print(f"   ✅ Found {len(diseases)} diseases")
    return diseases


def parse_product4(filepath: str, diseases: dict) -> dict:
    print(f"📂 Parsing {filepath}...")
    tree = ET.parse(filepath)
    root = tree.getroot()
    enriched = 0

    freq_rank = {
        "Obligate (100%)": -1,
        "Very frequent (99-80%)": 0,
        "Frequent (79-30%)": 1,
        "Occasional (29-5%)": 2,
        "Very rare (<4-1%)": 3,
        "Excluded (0%)": 5,
    }

    for disorder in root.iter("Disorder"):
        code = disorder.findtext("OrphaCode")
        if not code:
            continue
        if code not in diseases:
            diseases[code] = {
                "orpha_code": code,
                "name": disorder.findtext("Name") or "Unknown",
                "definition": "",
                "symptoms": [],
                "hpo_terms": [],
                "hpo_ids": [],
                "symptom_freqs": [],
            }

        assoc_rows = []
        for assoc in disorder.iter("HPODisorderAssociation"):
            hpo_id = assoc.findtext("HPO/HPOId")
            hpo_term = assoc.findtext("HPO/HPOTerm")
            freq = assoc.findtext("HPOFrequency/Name") or "Unknown"
            if hpo_id and hpo_term:
                assoc_rows.append((hpo_id, hpo_term, freq))

        if assoc_rows:
            assoc_rows.sort(key=lambda r: freq_rank.get(r[2], 4))
            diseases[code]["hpo_ids"] = [r[0] for r in assoc_rows]
            diseases[code]["hpo_terms"] = [r[1] for r in assoc_rows]
            diseases[code]["symptoms"] = [f"{r[1]} ({r[2]})" for r in assoc_rows]
            diseases[code]["symptom_freqs"] = [r[2] for r in assoc_rows]
            enriched += 1

    print(f"   ✅ Enriched {enriched} diseases with HPO symptoms")
    return diseases


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _expand_symptoms(symptom_text: str) -> str:
    expanded = symptom_text
    for term, synonyms in SYMPTOM_SYNONYMS.items():
        if term.lower() in expanded.lower():
            expanded += f" {synonyms}"
    return expanded


def _extract_distinguishing_features(info: dict) -> str:
    """Extract distinguishing biochemical/clinical markers for this disease."""
    name = info.get("name", "")
    definition = (info.get("definition") or "").lower()
    hpo_terms_lower = [t.lower() for t in info.get("hpo_terms", [])]

    markers = DISTINGUISHING_MARKERS.get(name, [])
    found_markers = []

    for marker in markers:
        marker_lower = marker.lower()
        if marker_lower in definition or any(marker_lower in ht for ht in hpo_terms_lower):
            found_markers.append(marker)

    if found_markers:
        return f"Distinguishing features: {', '.join(found_markers)}."
    return ""


def _get_gender_context(info: dict) -> str:
    """Add gender-specific context if present."""
    definition = (info.get("definition") or "").lower()
    name = info.get("name", "").lower()

    if "girls" in definition or "female" in definition or "rett" in name:
        return "Gender: female predominance."
    if "xxy" in definition or "klinefelter" in name:
        return "Gender: male. Karyotype 47XXY."
    if "45,x" in definition or "turner" in name:
        return "Gender: female. Karyotype 45X monosomy X."
    if "prader-willi" in name or "angelman" in name:
        return "Genomic imprinting disorder chromosome 15q11-q13."
    return ""


def build_database(diseases: dict) -> list:
    db = []
    for info in diseases.values():
        if not info.get("symptoms"):
            continue

        clean_symptoms = [s.split(" (")[0] for s in info["symptoms"]]
        clean_symptoms = _dedupe_preserve_order(clean_symptoms)
        top_symptoms = clean_symptoms[:MAX_SYMPTOMS_IN_TEXT]
        expanded_symptoms = [_expand_symptoms(s) for s in top_symptoms]
        top_hpo_ids = info.get("hpo_ids", [])[:MAX_SYMPTOMS_IN_TEXT]
        short_definition = (info["definition"] or "")[:MAX_DEFINITION_CHARS]

        # Build weighted searchable text with v3 enhancements:
        # 1. Disease name ONCE (reduced from 2x to reduce noise)
        # 2. Name aliases with biomedical context anchors
        # 3. HPO IDs (exact semantic anchors)
        # 4. Distinguishing features (biochemical markers, unique signs)
        # 5. Gender/genomic context
        # 6. Symptoms with frequency and synonyms
        # 7. Truncated definition

        name_part = f"{info['name']}."
        alias_part = NAME_ALIASES.get(info['name'], "")
        if alias_part:
            alias_part = f"Aliases: {alias_part}."

        hpo_part = f"HPO terms: {', '.join(top_hpo_ids)}." if top_hpo_ids else ""

        distinguishing_part = _extract_distinguishing_features(info)
        gender_part = _get_gender_context(info)

        symptoms_part = f"Clinical features: {', '.join(expanded_symptoms)}."
        definition_part = short_definition if short_definition else ""

        parts = [p for p in [name_part, alias_part, hpo_part, distinguishing_part, 
                             gender_part, symptoms_part, definition_part] if p]
        info["searchable_text"] = " ".join(parts)

        info["display_symptoms"] = top_symptoms
        info["distinguishing_features"] = distinguishing_part
        info["gender_context"] = gender_part

        db.append(info)
    return db


def main():
    print("🧬 MediNova — Disease Database Builder (v3.0)")

    print("📡 Step 1: Downloading Orphanet XML files...")
    download_file(PRODUCT1_URL, "en_product1.xml")
    download_file(PRODUCT4_URL, "en_product4.xml")

    print("\n🔬 Step 2: Parsing XML files...")
    diseases = parse_product1("en_product1.xml")
    diseases = parse_product4("en_product4.xml", diseases)

    print("\n💾 Step 3: Building database...")
    db = build_database(diseases)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(db, f)

    with open(CACHE_METADATA, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_diseases": len(db),
            "source": "Orphanet en_product1.xml + en_product4.xml",
            "version": "3.0",
            "features": [
                "hpo_ids_in_text",
                "symptom_synonyms",
                "frequency_ranking",
                "distinguishing_markers",
                "name_alias_injection",
                "gender_context",
                "reduced_name_noise"
            ]
        }, f, indent=2)

    print(f"\n✅ Done! {len(db)} diseases saved to {CACHE_FILE}")
    print(f"\n🔍 Sample: {db[0]['name']} — {', '.join(db[0]['symptoms'][:3])}")
    print(f"\n📝 Sample searchable_text (first 300 chars):")
    print(f"   {db[0]['searchable_text'][:300]}...")


if __name__ == "__main__":
    main()
