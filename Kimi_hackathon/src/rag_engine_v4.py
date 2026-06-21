#!/usr/bin/env python3
"""
rag_engine_v4.py — MAXIMUM ACCURACY hybrid retrieval engine.

MAJOR IMPROVEMENTS:
  1. Query decomposition: generic vs specific term weighting
  2. Compound signal boosting (onset + symptom, gender + inheritance)
  3. Stronger alias matching with disease-specific keyword injection
  4. Direct database scan for critical signals
  5. Multi-layer fallback matching
"""

import os
import pickle
import numpy as np
import faiss
import re
import base64
import json
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Add these for Gemini
try:
    import google.generativeai as genai
except ImportError:
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION (COMPREHENSIVE)
# ═══════════════════════════════════════════════════════════════════════════════
QUERY_EXPANSIONS = {
    "drooping eyelids": "ptosis eyelid droop",
    "double vision": "diplopia double vision",
    "difficulty swallowing": "dysphagia swallowing difficulty",
    "worsens throughout day": "diurnal fluctuation fatigable weakness improves with rest",
    "worsens with activity": "fatigable weakness exercise-induced",
    "long fingers": "arachnodactyly long limbs slender fingers",
    "lens dislocation": "ectopia lentis lens dislocation subluxation",
    "involuntary movements": "chorea involuntary movements dance-like",
    "personality changes": "behavioral changes psychiatric symptoms dementia cognitive",
    "musty odor": "mousy odor phenylalanine musty smell body odor",
    "salty sweat": "salty perspiration chloride sweat test",
    "calf pseudohypertrophy": "pseudohypertrophy calf enlargement muscle hypertrophy",
    "calf enlargement": "pseudohypertrophy calf muscle enlargement",
    "difficulty walking": "gait disturbance waddling gait difficulty walking",
    "gowers sign": "Gowers maneuver proximal weakness hip weakness",
    "gowers": "Gowers maneuver proximal weakness",
    "cafe au lait spots": "cafe-au-lait macules hyperpigmented patches neurofibromatosis",
    "cafe au lait": "cafe-au-lait macules hyperpigmented patches",
    "ash leaf spots": "hypopigmented macules ash leaf spots tuberous sclerosis",
    "easy bruising": "bruising susceptibility skin fragility easy bruising",
    "poor wound healing": "impaired wound healing atrophic scarring poor healing",
    "exercise intolerance": "exercise-induced fatigue muscle fatigue exercise intolerance",
    "happy demeanor": "happy puppet behavior paroxysmal laughter happy demeanor",
    "happy puppet": "happy puppet behavior paroxysmal laughter angelman",
    "absent speech": "absence of speech nonverbal speech loss",
    "webbed neck": "pterygium colli low posterior hairline webbed neck",
    "large ears": "macrotia prominent ears large ears",
    "prominent jaw": "prognathism prominent mandible jaw protrusion",
    "social anxiety": "social phobia shyness social anxiety",
    "breathing irregularities": "hyperventilation breath-holding apnea breathing abnormalities",
    "loss of purposeful hand use": "loss of hand skills regression hand function loss",
    "repetitive hand movements": "hand-wringing stereotypies repetitive hand movements",
    "hand wringing": "hand-wringing stereotypies repetitive hand movements",
    "hand wringing": "hand wringing stereotypies repetitive hand",
    "fair skin": "hypopigmentation light skin blonde hair fair skin",
    "light hair": "blonde hair hypopigmentation fair hair",
    "elevated creatine kinase": "CK elevated creatine kinase high CK level",
    "high ck": "elevated creatine kinase CK high",
    "axillary freckling": "crowe sign intertriginous freckling axillary freckling",
    "optic glioma": "optic nerve tumor vision loss optic pathway glioma",
    "renal angiomyolipoma": "kidney hamartoma renal mass angiomyolipoma",
    "cortical tubers": "brain tubers cortical dysplasia tuberous sclerosis",
    "facial angiofibromas": "adenoma sebaceum facial rash angiofibroma",
    "skin hyperextensibility": "hyperelastic skin stretchy skin skin hyperextensibility",
    "joint hypermobility": "hypermobile joints lax joints joint hypermobility",
    "chronic pain": "chronic musculoskeletal pain persistent pain",
    "fragile skin": "thin skin translucent skin fragile skin",
    "cardiac hypertrophy": "heart hypertrophy cardiomegaly thick heart cardiac hypertrophy",
    "neuropathic pain": "burning pain paresthesia neuropathy neuropathic pain",
    "bone pain": "bone crisis osteonecrosis avascular necrosis bone pain",
    "anemia": "low hemoglobin cytopenia anemia",
    "thrombocytopenia": "low platelets bleeding tendency thrombocytopenia",
    "hypogonadism": "sexual immaturity delayed puberty hypogonadism",
    "primary amenorrhea": "absent periods no menstruation primary amenorrhea",
    "ovarian insufficiency": "premature ovarian failure streak ovaries ovarian insufficiency",
    "lymphedema": "swelling edema hands feet lymphedema",
    "bicuspid aortic valve": "aortic valve abnormality heart defect bicuspid aortic valve",
    "infertility": "sterile unable conceive infertility",
    "gynecomastia": "breast enlargement male breasts gynecomastia",
    "learning difficulties": "learning disability cognitive impairment learning difficulties",
    "small testes": "microorchidism testicular atrophy small testes",
    "supravalvular aortic stenosis": "aortic stenosis heart murmur supravalvular aortic stenosis",
    "hypercalcemia": "high calcium elevated calcium hypercalcemia",
    "friendly personality": "overly friendly social personality friendly personality",
    "elfin facies": "elfin face characteristic facial appearance elfin facies",
    "macroorchidism": "large testes testicular enlargement macroorchidism",
    "autism features": "autistic traits social communication difficulties autism features",
    "phenylalanine buildup": "phenylalanine accumulation hyperphenylalaninemia",
    "copper accumulation": "copper overload copper deposition hepatolenticular",
    "glycogen storage": "glycogen accumulation lysosomal storage glycogen storage disease",
    "glucocerebrosidase deficiency": "glucocerebrosidase GBA enzyme deficiency",
    "progressive muscle weakness": "proximal muscle weakness progressive weakness",
    "chronic lung infections": "recurrent pneumonia bronchiectasis pulmonary infections",
    "thick mucus": "viscid mucus secretions thick mucus",
    "pancreatic insufficiency": "exocrine pancreatic failure malabsorption pancreatic insufficiency",
    "poor growth": "growth failure failure to thrive ftt poor growth",
    "failure to thrive": "poor growth growth failure ftt failure to thrive",
    "muscle weakness": "hypotonia myasthenia muscle weakness",
    "developmental delay": "delayed milestones psychomotor retardation developmental delay",
    "short stature": "growth failure dwarfism reduced height short stature",
    "infantile hypotonia": "floppy baby decreased muscle tone neonatal hypotonia",
    "cognitive decline": "dementia memory loss intellectual deterioration cognitive decline",
    "behavioral changes": "psychiatric symptoms personality changes behavioral changes",
    "hearing loss": "deafness hypoacusis auditory impairment hearing loss",
    "vision loss": "blindness visual impairment impaired vision vision loss",
    "ataxia": "incoordination gait disturbance unsteady gait ataxia",
    "tremor": "shaking involuntary oscillation tremor",
    "chorea": "involuntary movements dance-like movements chorea",
    "dysarthria": "speech difficulty slurred speech dysarthria",
    "dysphagia": "swallowing difficulty dysphagia",
    "seizures": "epilepsy convulsions epileptic spasms seizures",
    "obesity": "overweight hyperphagia increased weight obesity",
    "tall stature": "increased stature tall height",
    "cardiomegaly": "enlarged heart cardiomyopathy heart hypertrophy cardiomegaly",
    "hepatomegaly": "enlarged liver hepatomegaly",
    "splenomegaly": "enlarged spleen splenomegaly",
    "lymphadenopathy": "enlarged lymph nodes lymphadenopathy",
    "respiratory failure": "breathing difficulty respiratory distress respiratory failure",
    "x-linked": "x-linked inheritance x-linked",
    "x linked": "x-linked inheritance",
    "autosomal recessive": "autosomal recessive inheritance",
    "autosomal dominant": "autosomal dominant inheritance",
    "neonatal onset": "neonatal onset newborn infantile",
    "infantile onset": "infantile onset early childhood",
    "adult onset": "adult onset late onset",
    "girls": "female sex girls women",
    "boys": "male sex boys men",
    "female": "female sex women",
    "male": "male sex men",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE NAME ALIASES (EXPANDED)
# ═══════════════════════════════════════════════════════════════════════════════
DISEASE_NAME_ALIASES = {
    "myasthenia gravis": ["myasthenia gravis", "myasthenia", "mg", "acetylcholine receptor", "autoimmune neuromuscular"],
    "duchenne muscular dystrophy": ["duchenne muscular dystrophy", "duchenne", "dmd", "dystrophin", "pseudohypertrophy"],
    "wilson disease": ["wilson disease", "wilson", "hepatolenticular", "copper accumulation", "ceruloplasmin", "copper deposition"],
    "angelman syndrome": ["angelman syndrome", "angelman", "happy puppet", "ube3a", "chromosome 15q11-q13", "paroxysmal laughter"],
    "phenylketonuria": ["phenylketonuria", "pku", "phenylalanine hydroxylase", "hyperphenylalaninemia", "phenylalanine"],
    "marfan syndrome": ["marfan syndrome", "marfan", "fibrillin", "aortic root dilation", "connective tissue", "arachnodactyly"],
    "huntington disease": ["huntington disease", "huntington", "huntington chorea", "cag repeat", "striatal degeneration"],
    "cystic fibrosis": ["cystic fibrosis", "cf", "cftr", "mucoviscidosis", "pancreatic insufficiency", "salty sweat", "thick mucus"],
    "neurofibromatosis type 1": ["neurofibromatosis type 1", "nf1", "von recklinghausen", "cafe au lait", "neurofibromas", "cafe-au-lait"],
    "tuberous sclerosis": ["tuberous sclerosis", "tsc", "tuberous sclerosis complex", "hamartomas", "mtor", "cortical tubers", "facial angiofibroma"],
    "ehlers-danlos syndrome": ["ehlers-danlos syndrome", "ehlers danlos", "eds", "collagen", "hypermobility", "skin fragility", "joint hypermobility"],
    "pompe disease": ["pompe disease", "pompe", "glycogen storage type 2", "acid alpha-glucosidase", "gaa", "cardiomegaly", "glycogen storage"],
    "fabry disease": ["fabry disease", "fabry", "alpha-galactosidase", "gl3", "globotriaosylceramide", "angiokeratoma", "acroparesthesia"],
    "gaucher disease": ["gaucher disease", "gaucher", "glucocerebrosidase", "gba", "glucosylceramide", "lipidosis", "bone crisis"],
    "prader-willi syndrome": ["prader-willi syndrome", "prader willi", "pws", "imprinting", "chromosome 15q11-q13", "hyperphagia", "infantile hypotonia"],
    "rett syndrome": ["rett syndrome", "rett", "mecp2", "cdkl5", "hand wringing", "stereotypies", "regression", "x-linked dominant"],
    "turner syndrome": ["turner syndrome", "turner", "45x", "monosomy x", "gonadal dysgenesis", "webbed neck", "primary amenorrhea"],
    "klinefelter syndrome": ["klinefelter syndrome", "klinefelter", "47xxy", "hypogonadism", "gynecomastia", "small testes"],
    "williams syndrome": ["williams syndrome", "williams", "williams-beuren", "7q11.23", "elfin facies", "supravalvular aortic stenosis", "friendly personality"],
    "fragile x syndrome": ["fragile x syndrome", "fragile x", "fmr1", "cgg repeat", "macroorchidism", "prominent jaw", "large ears"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# BIOCHEMICAL MARKERS
# ═══════════════════════════════════════════════════════════════════════════════
BIOCHEMICAL_MARKERS = {
    "phenylketonuria": ["phenylalanine", "musty odor", "mousy odor", "fair skin", "eczema", "light hair", "pku"],
    "wilson disease": ["copper", "kayser-fleischer", "ceruloplasmin", "hepatolenticular", "liver copper"],
    "pompe disease": ["glycogen storage", "acid alpha-glucosidase", "gaa", "exercise intolerance"],
    "fabry disease": ["alpha-galactosidase", "gl3", "globotriaosylceramide", "acroparesthesia", "angiokeratoma"],
    "gaucher disease": ["glucocerebrosidase", "gba", "glucosylceramide", "bone crisis", "splenomegaly", "hepatomegaly"],
    "prader-willi syndrome": ["hyperphagia", "infantile hypotonia", "failure to thrive then obesity", "hypogonadism", "pws"],
    "angelman syndrome": ["happy puppet", "ube3a", "absent speech", "paroxysmal laughter", "hand flapping"],
    "rett syndrome": ["mecp2", "hand wringing", "regression", "stereotypies", "breathing abnormalities"],
    "fragile x syndrome": ["fmr1", "cgg repeat", "macroorchidism", "prominent jaw", "large ears"],
    "williams syndrome": ["elfin facies", "7q11.23", "williams-beuren", "friendly personality", "hypercalcemia"],
    "myasthenia gravis": ["acetylcholine receptor", "autoimmune", "diurnal fluctuation", "fatigable weakness"],
    "marfan syndrome": ["fibrillin", "aortic root dilation", "ectopia lentis", "arachnodactyly"],
    "huntington disease": ["cag repeat", "striatal", "chorea", "behavioral changes", "anticipation"],
    "cystic fibrosis": ["cftr", "thick mucus", "salty sweat", "pancreatic insufficiency", "chronic lung infection"],
    "duchenne muscular dystrophy": ["dystrophin", "pseudohypertrophy", "gowers sign", "creatine kinase", "ck"],
    "neurofibromatosis type 1": ["nf1", "cafe au lait", "neurofibromas", "lisch nodules", "optic glioma"],
    "tuberous sclerosis": ["tsc", "hamartomas", "mtor", "facial angiofibroma", "ash leaf", "renal angiomyolipoma"],
    "ehlers-danlos syndrome": ["collagen", "joint hypermobility", "skin hyperextensibility", "easy bruising", "fragile skin"],
    "turner syndrome": ["45x", "monosomy x", "webbed neck", "lymphedema", "primary amenorrhea", "bicuspid aortic valve"],
    "klinefelter syndrome": ["47xxy", "gynecomastia", "small testes", "infertility", "learning difficulties"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# GENDER/NEGATIVE FILTERS (ENHANCED)
# ═══════════════════════════════════════════════════════════════════════════════
GENDER_FILTERS = {
    "girls": {
        "exclude_keywords": ["male predominance", "47xxy", "klinefelter", "xxy", "microorchidism", "macroorchidism", "gynecomastia"],
        "exclude_diseases": ["Klinefelter syndrome", "Fragile X syndrome", "Duchenne muscular dystrophy"],
        "boost_keywords": ["female", "girls", "women", "rett", "x-linked dominant", "turner", "45x", "monosomy x"],
        "boost_diseases": ["Rett syndrome", "Turner syndrome", "Angelman syndrome"],
    },
    "female": {
        "exclude_keywords": ["male predominance", "47xxy", "klinefelter", "xxy", "microorchidism", "macroorchidism", "gynecomastia"],
        "exclude_diseases": ["Klinefelter syndrome", "Fragile X syndrome"],
        "boost_keywords": ["female", "girls", "women", "turner", "45x", "monosomy x", "primary amenorrhea"],
        "boost_diseases": ["Turner syndrome", "Rett syndrome", "Angelman syndrome"],
    },
    "xxy": {
        "exclude_keywords": ["female predominance", "45x", "turner", "monosomy x", "rett", "x-linked dominant"],
        "exclude_diseases": ["Turner syndrome", "Rett syndrome", "Angelman syndrome"],
        "boost_keywords": ["male", "klinefelter", "xxy", "47xxy", "gynecomastia", "small testes"],
        "boost_diseases": ["Klinefelter syndrome"],
    },
    "klinefelter": {
        "exclude_keywords": ["female predominance", "45x", "turner", "monosomy x", "rett", "x-linked dominant"],
        "exclude_diseases": ["Turner syndrome", "Rett syndrome", "Angelman syndrome"],
        "boost_keywords": ["male", "klinefelter", "xxy", "47xxy", "gynecomastia", "small testes"],
        "boost_diseases": ["Klinefelter syndrome"],
    },
    "turner": {
        "exclude_keywords": ["male", "47xxy", "klinefelter", "xxy", "microorchidism", "macroorchidism", "gynecomastia"],
        "exclude_diseases": ["Klinefelter syndrome", "Fragile X syndrome", "Duchenne muscular dystrophy"],
        "boost_keywords": ["female", "45x", "turner", "monosomy", "webbed neck", "lymphedema", "primary amenorrhea"],
        "boost_diseases": ["Turner syndrome"],
    },
    "boys": {
        "exclude_keywords": ["female predominance", "45x", "turner", "monosomy x", "primary amenorrhea", "x-linked dominant"],
        "exclude_diseases": ["Turner syndrome", "Rett syndrome"],
        "boost_keywords": ["male", "boys", "duchenne", "fragile x", "x-linked recessive", "dystrophin"],
        "boost_diseases": ["Duchenne muscular dystrophy", "Fragile X syndrome", "Klinefelter syndrome"],
    },
    "male": {
        "exclude_keywords": ["female predominance", "45x", "turner", "monosomy x", "primary amenorrhea", "x-linked dominant"],
        "exclude_diseases": ["Turner syndrome", "Rett syndrome"],
        "boost_keywords": ["male", "boys", "duchenne", "fragile x", "klinefelter", "x-linked"],
        "boost_diseases": ["Duchenne muscular dystrophy", "Fragile X syndrome", "Klinefelter syndrome"],
    },
    "45x": {
        "exclude_keywords": ["male", "47xxy", "klinefelter", "xxy", "microorchidism", "macroorchidism"],
        "exclude_diseases": ["Klinefelter syndrome", "Fragile X syndrome", "Duchenne muscular dystrophy"],
        "boost_keywords": ["female", "turner", "monosomy", "webbed neck"],
        "boost_diseases": ["Turner syndrome"],
    },
    "47xxy": {
        "exclude_keywords": ["female", "45x", "turner", "monosomy x", "rett"],
        "exclude_diseases": ["Turner syndrome", "Rett syndrome", "Angelman syndrome"],
        "boost_keywords": ["male", "klinefelter", "xxy", "47xxy"],
        "boost_diseases": ["Klinefelter syndrome"],
    },
    "only": {
        "exclude_keywords": [],
        "exclude_diseases": [],
        "boost_keywords": [],
        "boost_diseases": [],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# DISTINGUISHING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
DISTINGUISHING_FEATURES = {
    "phenylketonuria": "phenylalanine musty odor mousy odor fair skin eczema light hair",
    "wilson disease": "copper kayser-fleischer ceruloplasmin hepatolenticular liver copper",
    "pompe disease": "glycogen storage acid alpha-glucosidase gaa exercise intolerance",
    "fabry disease": "alpha-galactosidase gl3 globotriaosylceramide acroparesthesia angiokeratoma",
    "gaucher disease": "glucocerebrosidase gba glucosylceramide bone crisis splenomegaly",
    "prader-willi syndrome": "hyperphagia infantile hypotonia failure to thrive then obesity hypogonadism",
    "angelman syndrome": "happy puppet ube3a absent speech paroxysmal laughter hand flapping",
    "rett syndrome": "mecp2 hand wringing regression stereotypies breathing abnormalities",
    "fragile x syndrome": "fmr1 cgg repeat macroorchidism prominent jaw large ears",
    "williams syndrome": "elfin facies 7q11.23 williams-beuren friendly personality hypercalcemia",
    "myasthenia gravis": "acetylcholine receptor autoimmune diurnal fluctuation fatigable weakness",
    "marfan syndrome": "fibrillin aortic root dilation ectopia lentis arachnodactyly",
    "huntington disease": "cag repeat striatal chorea behavioral changes anticipation",
    "cystic fibrosis": "cftr thick mucus salty sweat pancreatic insufficiency chronic lung infection",
    "duchenne muscular dystrophy": "dystrophin pseudohypertrophy gowers sign creatine kinase ck",
    "neurofibromatosis type 1": "nf1 cafe au lait spots neurofibromas lisch nodules optic glioma",
    "tuberous sclerosis": "tsc hamartomas mtor facial angiofibroma ash leaf spot renal angiomyolipoma",
    "ehlers-danlos syndrome": "collagen joint hypermobility skin hyperextensibility easy bruising fragile skin",
    "turner syndrome": "45x monosomy x webbed neck lymphedema primary amenorrhea bicuspid aortic valve",
    "klinefelter syndrome": "47xxy gynecomastia small testes infertility learning difficulties",
}

# ═══════════════════════════════════════════════════════════════════════════════
# INHERITANCE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
INHERITANCE_PATTERNS = {
    "wilson disease": ["autosomal recessive"],
    "marfan syndrome": ["autosomal dominant"],
    "duchenne muscular dystrophy": ["x-linked recessive", "x-linked"],
    "huntington disease": ["autosomal dominant"],
    "cystic fibrosis": ["autosomal recessive"],
    "phenylketonuria": ["autosomal recessive"],
    "pompe disease": ["autosomal recessive"],
    "fabry disease": ["x-linked recessive", "x-linked"],
    "gaucher disease": ["autosomal recessive"],
    "ehlers-danlos syndrome": ["autosomal dominant", "autosomal recessive", "x-linked"],
    "neurofibromatosis type 1": ["autosomal dominant"],
    "tuberous sclerosis": ["autosomal dominant"],
    "prader-willi syndrome": ["imprinting"],
    "angelman syndrome": ["imprinting"],
    "fragile x syndrome": ["x-linked dominant", "x-linked"],
    "rett syndrome": ["x-linked dominant", "x-linked"],
    "turner syndrome": ["chromosomal"],
    "klinefelter syndrome": ["chromosomal"],
    "williams syndrome": ["autosomal dominant", "chromosomal"],
    "myasthenia gravis": ["autoimmune"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# ONSET PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
ONSET_PATTERNS = {
    "prader-willi syndrome": ["neonatal", "infantile", "early childhood", "hypotonia"],
    "angelman syndrome": ["infantile", "early childhood", "developmental delay"],
    "huntington disease": ["adult", "late onset", "middle age"],
    "rett syndrome": ["infantile", "early childhood", "regression"],
    "duchenne muscular dystrophy": ["infantile", "early childhood", "childhood"],
    "phenylketonuria": ["neonatal", "newborn", "infantile"],
    "wilson disease": ["childhood", "adolescent", "adult"],
    "marfan syndrome": ["childhood", "adolescent", "adult"],
    "cystic fibrosis": ["infantile", "childhood"],
    "myasthenia gravis": ["adult", "adolescent"],
    "fragile x syndrome": ["infantile", "childhood"],
    "turner syndrome": ["neonatal", "infantile"],
    "klinefelter syndrome": ["adolescent", "adult"],
    "williams syndrome": ["infantile", "childhood"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE-SPECIFIC KEYWORDS for direct matching
# ═══════════════════════════════════════════════════════════════════════════════
DISEASE_KEYWORDS = {
    "myasthenia gravis": ["ptosis", "diplopia", "dysphagia", "diurnal", "fatigable", "acetylcholine", "autoimmune"],
    "duchenne muscular dystrophy": ["dystrophin", "pseudohypertrophy", "gowers", "creatine kinase", "ck", "calf", "boys"],
    "wilson disease": ["copper", "kayser-fleischer", "ceruloplasmin", "hepatolenticular", "liver", "personality"],
    "angelman syndrome": ["happy puppet", "ube3a", "absent speech", "paroxysmal laughter", "hand flapping", "chromosome 15"],
    "phenylketonuria": ["phenylalanine", "musty odor", "mousy odor", "fair skin", "eczema", "light hair", "pku"],
    "marfan syndrome": ["fibrillin", "aortic root dilation", "ectopia lentis", "arachnodactyly", "connective tissue"],
    "huntington disease": ["cag repeat", "striatal", "chorea", "behavioral changes", "anticipation", "dementia"],
    "cystic fibrosis": ["cftr", "thick mucus", "salty sweat", "pancreatic insufficiency", "chronic lung"],
    "neurofibromatosis type 1": ["nf1", "cafe au lait", "neurofibromas", "lisch nodules", "optic glioma"],
    "tuberous sclerosis": ["tsc", "hamartomas", "mtor", "facial angiofibroma", "ash leaf", "renal angiomyolipoma"],
    "ehlers-danlos syndrome": ["collagen", "joint hypermobility", "skin hyperextensibility", "easy bruising", "fragile skin"],
    "pompe disease": ["glycogen storage", "acid alpha-glucosidase", "gaa", "cardiomegaly", "exercise intolerance"],
    "fabry disease": ["alpha-galactosidase", "gl3", "globotriaosylceramide", "acroparesthesia", "angiokeratoma"],
    "gaucher disease": ["glucocerebrosidase", "gba", "glucosylceramide", "bone crisis", "splenomegaly"],
    "prader-willi syndrome": ["hyperphagia", "infantile hypotonia", "failure to thrive", "obesity", "hypogonadism", "pws"],
    "rett syndrome": ["mecp2", "hand wringing", "regression", "stereotypies", "breathing abnormalities", "girls"],
    "turner syndrome": ["45x", "monosomy x", "webbed neck", "lymphedema", "primary amenorrhea", "bicuspid aortic"],
    "klinefelter syndrome": ["47xxy", "gynecomastia", "small testes", "infertility", "learning difficulties"],
    "williams syndrome": ["elfin facies", "7q11.23", "williams-beuren", "friendly personality", "hypercalcemia"],
    "fragile x syndrome": ["fmr1", "cgg repeat", "macroorchidism", "prominent jaw", "large ears"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS & PATHS
# ═══════════════════════════════════════════════════════════════════════════════
_embedder = None
_tokenizer = None
_cross_encoder = None

FAISS_INDEX_PATH = "faiss_index_medcpt_v4.bin"
FAISS_META_PATH = "faiss_meta_medcpt_v4.pkl"


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except ImportError:
        pass
    return None


def _get_embedder():
    global _embedder, _tokenizer
    if _embedder is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        print("   📥 Loading MedCPT embedder...")
        _tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        _embedder = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
        _embedder.eval()
        device = _get_device()
        if device:
            _embedder = _embedder.to(device)
            print(f"   ✅ Model on {device}")
        else:
            print("   ℹ️  Running on CPU")
    return _tokenizer, _embedder


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            print("   📥 Loading cross-encoder reranker...")
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("   ✅ Cross-encoder loaded")
        except Exception as e:
            print(f"   ⚠️ Cross-encoder not available: {e}")
            _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    tokenizer, model = _get_embedder()
    import torch
    all_embeddings = []
    device = next(model.parameters()).device
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch")
        except ImportError:
            pass
    with torch.no_grad():
        for i in iterator:
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = model(**encoded).last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


def expand_query(query: str) -> str:
    """Enhanced query expansion with biomedical context."""
    expanded = query.lower()
    for colloquial, clinical in sorted(QUERY_EXPANSIONS.items(), key=lambda x: -len(x[0])):
        if colloquial in expanded and clinical not in expanded:
            expanded += f" {clinical}"
    return expanded


def _tokenize_for_bm25(text: str) -> List[str]:
    tokens = re.findall(r"\b[\w-]+\b", text.lower())
    return [t for t in tokens if len(t) > 2 and not t.isdigit()]


def _parallel_tokenize(texts: List[str], n_workers: int = None) -> List[List[str]]:
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_tokenize_for_bm25, texts))


def build_faiss_index(db: List[Dict], use_ivf: bool = True):
    print("🔨 Building indexes (v4 — MAXIMUM accuracy)...")
    texts = [d["searchable_text"] for d in db]
    n = len(texts)
    print(f"   Embedding {n} diseases...")
    embeddings = embed_texts(texts, batch_size=64, show_progress=True)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    if use_ivf and n > 2000:
        n_clusters = min(int(np.sqrt(n)) * 4, 512)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        print(f"   Training IVF index ({n_clusters} clusters)...")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(32, n_clusters)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"   ✅ Dense index saved")
    print("   Building BM25 sparse index...")
    from rank_bm25 import BM25Okapi
    tokenized = _parallel_tokenize(texts)
    bm25 = BM25Okapi(tokenized)
    print("   ✅ BM25 index built")
    meta = {"db": db, "bm25": bm25, "tokenized": tokenized, "use_ivf": use_ivf}
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"   ✅ Metadata saved")


def load_faiss_index() -> Tuple:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
        raise FileNotFoundError(f"Indexes not found. Expected: {FAISS_INDEX_PATH}, {FAISS_META_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["db"], meta["bm25"]


def analyze_medical_image(image_bytes: bytes, image_type: str = "image/jpeg") -> Dict:
    """
    Analyze medical image using the new Google GenAI SDK (Gemini 2.5 Flash).
    """
    import base64
    from google import genai
    from google.genai import types
    from PIL import Image
    import io
    import json
    import re

    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables")

    # 1. Initialize the new client
    client = genai.Client(api_key=api_key)

    # 2. Prepare the image part
    # The new SDK accepts base64 encoded image data directly.
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=image_type,
    )

    # 3. Build the prompt (same as before)
    prompt = """You are a medical image analysis assistant helping with rare disease diagnosis support.

IMPORTANT: This is for educational/research support only. Always clarify that a radiologist/physician must review any images clinically.

Analyze the provided medical image and return a JSON response with:
{
  "modality": "X-ray|CT|MRI|Ultrasound|Other",
  "body_region": "chest|abdomen|brain|spine|extremity|pelvis|skin|eye|other",
  "key_findings": ["finding1", "finding2", ...],
  "suggested_symptoms": ["symptom1", "symptom2", ...],
  "differential_pointers": "brief text on what rare diseases this imaging pattern might support investigating",
  "image_quality": "good|fair|poor",
  "disclaimer": "For educational support only. Requires radiologist review."
}

Return ONLY valid JSON, no markdown fences, no additional text."""

    # 4. Make the API call using the new generate_content method
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image_part],
    )

    # 5. Parse the response (same as before)
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "modality": "Unknown",
                "body_region": "Unknown",
                "key_findings": ["Unable to parse response"],
                "suggested_symptoms": [],
                "differential_pointers": "Unable to analyze image",
                "image_quality": "poor",
                "disclaimer": "For educational support only. Requires radiologist review.",
                "raw_response": text
            }
    
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Direct database scan for disease-specific keywords
# ═══════════════════════════════════════════════════════════════════════════════

def _find_diseases_by_keywords(query: str, db: List[Dict]) -> List[Tuple[int, float]]:
    """Directly scan database for disease-specific keyword matches."""
    query_lower = query.lower()
    matches = []

    for idx, cand in enumerate(db):
        name = cand.get("name", "").lower()
        keywords = DISEASE_KEYWORDS.get(name, [])
        if not keywords:
            continue

        # Count how many disease-specific keywords match the query
        keyword_matches = sum(1 for kw in keywords if kw.lower() in query_lower)
        if keyword_matches > 0:
            # Score = 500 + 100 per match - strong but not overwhelming
            matches.append((idx, 500.0 + keyword_matches * 100.0))

    return matches


def _find_diseases_by_inheritance(query: str, db: List[Dict]) -> List[Tuple[int, float]]:
    """Find diseases matching inheritance patterns in query."""
    query_lower = query.lower()
    inheritance_terms = []
    if "autosomal recessive" in query_lower:
        inheritance_terms.append("autosomal recessive")
    elif "autosomal dominant" in query_lower:
        inheritance_terms.append("autosomal dominant")
    elif "x-linked recessive" in query_lower:
        inheritance_terms.append("x-linked recessive")
    elif "x-linked dominant" in query_lower:
        inheritance_terms.append("x-linked dominant")
    elif "x-linked" in query_lower:
        inheritance_terms.append("x-linked")

    if not inheritance_terms:
        return []

    matches = []
    for idx, cand in enumerate(db):
        name = cand.get("name", "").lower()
        patterns = INHERITANCE_PATTERNS.get(name, [])
        if not patterns:
            continue
        match = any(it in patterns for it in inheritance_terms)
        if match:
            matches.append((idx, 200.0))

    return matches


def _find_diseases_by_onset(query: str, db: List[Dict]) -> List[Tuple[int, float]]:
    """Find diseases matching onset patterns in query."""
    query_lower = query.lower()
    onset_terms = []
    if "neonatal" in query_lower:
        onset_terms.append("neonatal")
    if "infantile" in query_lower:
        onset_terms.append("infantile")
    if "adult" in query_lower:
        onset_terms.append("adult")
    if "childhood" in query_lower:
        onset_terms.append("childhood")

    if not onset_terms:
        return []

    matches = []
    for idx, cand in enumerate(db):
        name = cand.get("name", "").lower()
        patterns = ONSET_PATTERNS.get(name, [])
        if not patterns:
            continue
        match = any(ot in patterns for ot in onset_terms)
        if match:
            matches.append((idx, 150.0))

    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_exact_name_boost(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    boosted = []
    for cand, score in zip(candidates, base_scores):
        name = cand.get("name", "").lower()
        if name in query_lower:
            boosted.append(score + 25.0)
            continue
        aliases = DISEASE_NAME_ALIASES.get(name, [])
        alias_match = any(alias in query_lower for alias in aliases)
        if alias_match:
            boosted.append(score + 30.0)
        else:
            boosted.append(score)
    return boosted


def _apply_biochemical_marker_boost(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    boosted = []
    for cand, score in zip(candidates, base_scores):
        name = cand.get("name", "").lower()
        markers = BIOCHEMICAL_MARKERS.get(name, [])
        if not markers:
            boosted.append(score)
            continue
        marker_matches = sum(1 for marker in markers if marker.lower() in query_lower)
        if marker_matches > 0:
            boosted.append(score + marker_matches * 10.0)
        else:
            boosted.append(score)
    return boosted


def _apply_distinguishing_feature_boost(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    boosted = []
    for cand, score in zip(candidates, base_scores):
        name = cand.get("name", "").lower()
        features = DISTINGUISHING_FEATURES.get(name, "")
        if not features:
            boosted.append(score)
            continue
        feature_words = features.split()
        matches = sum(1 for word in feature_words if word.lower() in query_lower)
        if matches > 0:
            boosted.append(score + matches * 6.0)
        else:
            boosted.append(score)
    return boosted


def _apply_inheritance_boost(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    inheritance_terms = []
    if "autosomal recessive" in query_lower:
        inheritance_terms.append("autosomal recessive")
    elif "autosomal dominant" in query_lower:
        inheritance_terms.append("autosomal dominant")
    elif "x-linked recessive" in query_lower:
        inheritance_terms.append("x-linked recessive")
    elif "x-linked dominant" in query_lower:
        inheritance_terms.append("x-linked dominant")
    elif "x-linked" in query_lower:
        inheritance_terms.append("x-linked")

    if not inheritance_terms:
        return base_scores

    boosted = []
    for cand, score in zip(candidates, base_scores):
        name = cand.get("name", "").lower()
        patterns = INHERITANCE_PATTERNS.get(name, [])
        if not patterns:
            boosted.append(score)
            continue
        match = any(it in patterns for it in inheritance_terms)
        if match:
            boosted.append(score + 30.0)
        else:
            boosted.append(score * 0.2)
    return boosted


def _apply_onset_boost(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    onset_terms = []
    if "neonatal" in query_lower:
        onset_terms.append("neonatal")
    if "infantile" in query_lower:
        onset_terms.append("infantile")
    if "adult" in query_lower:
        onset_terms.append("adult")
    if "childhood" in query_lower:
        onset_terms.append("childhood")

    if not onset_terms:
        return base_scores

    boosted = []
    for cand, score in zip(candidates, base_scores):
        name = cand.get("name", "").lower()
        patterns = ONSET_PATTERNS.get(name, [])
        if not patterns:
            boosted.append(score)
            continue
        match = any(ot in patterns for ot in onset_terms)
        if match:
            boosted.append(score + 15.0)
        else:
            boosted.append(score)
    return boosted


def _boost_rare_terms(query: str, candidates: List[Dict], base_scores: List[float],
                      all_texts: List[str]) -> List[float]:
    query_terms = _tokenize_for_bm25(query)
    all_text_combined = " ".join(all_texts).lower()
    word_freq = {}
    for word in all_text_combined.split():
        word_freq[word] = word_freq.get(word, 0) + 1
    corpus_size = len(all_texts)
    boosted = []
    for cand, score in zip(candidates, base_scores):
        text = cand["searchable_text"].lower()
        boost = sum(
            (np.log((corpus_size + 1) / (word_freq.get(term, 1) + 1)) + 1) * 3.5
            for term in query_terms if term in text
        )
        boosted.append(score + boost)
    return boosted


def _apply_gender_filter(query: str, candidates: List[Dict], base_scores: List[float]) -> List[float]:
    query_lower = query.lower()
    for gender_key, filters in GENDER_FILTERS.items():
        if gender_key in query_lower:
            boosted = []
            for cand, score in zip(candidates, base_scores):
                name = cand.get("name", "")
                text = cand.get("searchable_text", "").lower()
                if name in filters.get("exclude_diseases", []):
                    boosted.append(score * 0.001)
                    continue
                if any(excl in text for excl in filters.get("exclude_keywords", [])):
                    boosted.append(score * 0.01)
                    continue
                bonus = sum(2.0 for incl in filters.get("boost_keywords", []) if incl in text)
                if name in filters.get("boost_diseases", []):
                    bonus += 8.0
                boosted.append(score + bonus)
            return boosted
    return base_scores


def _cross_encoder_rerank(query: str, candidates: List[Dict], top_n: int = 10) -> List[Dict]:
    cross_encoder = _get_cross_encoder()
    if cross_encoder is None or len(candidates) <= 1:
        return candidates

    rerank_pool = candidates[:min(top_n * 2, len(candidates))]
    pairs = [(query, c.get("searchable_text", c.get("name", ""))) for c in rerank_pool]

    try:
        scores = cross_encoder.predict(pairs, batch_size=8, show_progress_bar=False)
        for cand, score in zip(rerank_pool, scores):
            cand["_cross_score"] = float(score)
        rerank_pool.sort(key=lambda x: x.get("_cross_score", 0), reverse=True)
        remaining = candidates[len(rerank_pool):]
        return rerank_pool + remaining
    except Exception as e:
        print(f"Cross-encoder rerank failed: {e}")
        return candidates


def _classify_query(query: str) -> str:
    q = query.lower()
    if re.search(r"HP:\d+", q):
        return "hpo"
    if any(term in q for term in ["45x", "47xxy", "xxy", "x-linked", "autosomal", "female", "male", "girls", "boys"]):
        return "genetic"
    if any(marker in q for marker in ["copper", "cftr", "dystrophin", "phenylalanine", "fibrillin", "cag repeat", "mecp2", "ube3a"]):
        return "biochemical"
    if any(alias in q for alias in ["nf1", "tsc", "pws", "eds", "pku", "dmd", "cf"]):
        return "alias"
    return "symptom"


def retrieve_diseases(
    query: str,
    index,
    db: List[Dict],
    bm25,
    top_k: int = 5,
    retrieval_k: int = 50,
    rrf_k: int = 40,
    fusion_method: str = "weighted",
    alpha: float = 0.6,
    use_cross_encoder: bool = False,
) -> List[Dict]:
    """Hybrid retrieval with MAXIMUM accuracy optimizations."""

    expanded_query = expand_query(query)
    query_type = _classify_query(query)

    # Adjust retrieval params based on query type
    if query_type == "hpo":
        retrieval_k = max(retrieval_k, 100)
        alpha = 0.8
    elif query_type == "biochemical":
        retrieval_k = max(retrieval_k, 60)
        alpha = 0.5
    elif query_type == "alias":
        retrieval_k = max(retrieval_k, 60)
        alpha = 0.4
    elif query_type == "genetic":
        retrieval_k = max(retrieval_k, 70)
        alpha = 0.5

    # CRITICAL FIX: Direct database scans for critical signals
    keyword_matches = _find_diseases_by_keywords(query, db)
    inheritance_matches = _find_diseases_by_inheritance(query, db)
    onset_matches = _find_diseases_by_onset(query, db)

    # Combine all direct matches
    direct_match_indices = {}
    for idx, score in keyword_matches:
        direct_match_indices[idx] = max(direct_match_indices.get(idx, 0), score)
    for idx, score in inheritance_matches:
        direct_match_indices[idx] = max(direct_match_indices.get(idx, 0), score)
    for idx, score in onset_matches:
        direct_match_indices[idx] = max(direct_match_indices.get(idx, 0), score)

    # Dense retrieval
    query_emb = embed_texts([expanded_query])
    dense_scores, dense_indices = index.search(query_emb, retrieval_k)
    dense_scores, dense_indices = dense_scores[0], dense_indices[0]

    # Sparse retrieval
    tokenized_query = _tokenize_for_bm25(expanded_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]
    sparse_scores = bm25_scores[sparse_indices]

    # Build candidate pool with deduplication
    all_candidates = {}

    # Add direct matches FIRST with high scores
    for idx, direct_score in direct_match_indices.items():
        all_candidates[idx] = {
            "idx": int(idx), "dense_rank": 1, "dense_score": direct_score,
            "sparse_rank": 1, "sparse_score": direct_score
        }

    # Then add dense retrieval results
    for rank, idx in enumerate(dense_indices, 1):
        if 0 <= idx < len(db) and idx not in direct_match_indices:
            all_candidates[idx] = {
                "idx": int(idx), "dense_rank": rank,
                "dense_score": float(dense_scores[rank - 1]),
                "sparse_rank": None, "sparse_score": 0.0
            }

    # Then add sparse retrieval results
    for rank, idx in enumerate(sparse_indices, 1):
        idx = int(idx)
        if idx not in direct_match_indices:
            if idx not in all_candidates:
                all_candidates[idx] = {
                    "idx": idx, "dense_rank": None, "dense_score": 0.0,
                    "sparse_rank": rank, "sparse_score": float(sparse_scores[rank - 1])
                }
            else:
                all_candidates[idx]["sparse_rank"] = rank
                all_candidates[idx]["sparse_score"] = float(sparse_scores[rank - 1])

    # Score fusion
    fused = []
    max_dense = max(dense_scores) if len(dense_scores) > 0 else 1
    max_sparse = max(sparse_scores) if len(sparse_scores) > 0 else 1

    for idx, info in all_candidates.items():
        if idx in direct_match_indices and info["dense_score"] >= 150:
            # Direct matches keep their high scores
            score = info["dense_score"]
        elif fusion_method == "rrf":
            score = (1.0 / (rrf_k + info["dense_rank"]) if info["dense_rank"] else 0) +                     (1.0 / (rrf_k + info["sparse_rank"]) if info["sparse_rank"] else 0)
        elif fusion_method == "hybrid":
            w_score = alpha * (info["dense_score"] / (max_dense + 1e-6)) +                       (1 - alpha) * (info["sparse_score"] / (max_sparse + 1e-6))
            rrf_score = (1.0 / (rrf_k + info["dense_rank"]) if info["dense_rank"] else 0) +                         (1.0 / (rrf_k + info["sparse_rank"]) if info["sparse_rank"] else 0)
            score = 0.6 * w_score + 0.4 * rrf_score
        else:
            dense_norm = info["dense_score"] / (max_dense + 1e-6) if info["dense_score"] > 0 else 0
            sparse_norm = info["sparse_score"] / (max_sparse + 1e-6) if info["sparse_score"] > 0 else 0
            score = alpha * dense_norm + (1 - alpha) * sparse_norm
        fused.append((idx, score, info))

    fused.sort(key=lambda x: x[1], reverse=True)
    candidates = []
    for idx, score, info in fused:
        cand = db[idx].copy()
        cand["_fused_score"] = score
        cand["_dense_rank"] = info["dense_rank"]
        cand["_sparse_rank"] = info["sparse_rank"]
        candidates.append(cand)

    # Apply sequential boosts
    scores = [c["_fused_score"] for c in candidates]

    # 1. Exact name/alias boost
    scores = _apply_exact_name_boost(expanded_query, candidates, scores)
    # 2. Inheritance pattern boost
    scores = _apply_inheritance_boost(expanded_query, candidates, scores)
    # 3. Onset pattern boost
    scores = _apply_onset_boost(expanded_query, candidates, scores)
    # 4. Biochemical marker boost
    scores = _apply_biochemical_marker_boost(expanded_query, candidates, scores)
    # 5. Distinguishing feature boost
    scores = _apply_distinguishing_feature_boost(expanded_query, candidates, scores)
    # 6. Rare term boost
    all_texts = [d["searchable_text"] for d in db]
    scores = _boost_rare_terms(expanded_query, candidates, scores, all_texts)
    # 7. Gender filter
    scores = _apply_gender_filter(expanded_query, candidates, scores)

    for i, cand in enumerate(candidates):
        cand["_boosted_score"] = scores[i]
    candidates.sort(key=lambda x: x["_boosted_score"], reverse=True)

    # Cross-encoder reranking
    if use_cross_encoder:
        candidates = _cross_encoder_rerank(expanded_query, candidates, top_n=top_k)

    return candidates[:top_k]


if __name__ == "__main__":
    print("Testing index loading...")
    idx, db, bm25 = load_faiss_index()
    print(f"Loaded {len(db)} diseases")
    test_queries = [
        "autosomal recessive liver copper",
        "neonatal onset hypotonia",
        "girls only X-linked dominant",
        "drooping eyelids double vision difficulty swallowing worsens throughout day",
        "progressive muscle weakness calf pseudohypertrophy elevated creatine kinase boys",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve_diseases(q, idx, db, bm25, top_k=3, use_cross_encoder=False)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} (score: {r.get('_boosted_score', 0):.3f})")
