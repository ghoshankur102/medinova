"""
evaluation_config.py - Configuration for MediNova Evaluation Suite
Uses MedCPT for retrieval (existing) and Gemini Pro for image analysis
"""

import os
from typing import Dict, List, Optional

class ModelConfig:
    """Configuration for model APIs"""
    def __init__(self):
        # Gemini Pro for computer vision / image analysis
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model = "gemini-1.5-pro"
        self.gemini_temperature = 0.2
        
        # MedCPT (existing) for embeddings - no API key needed
        self.medcpt_model = "ncbi/MedCPT-Query-Encoder"


class EvalConfig:
    """Evaluation configuration"""
    def __init__(self):
        # Test parameters
        self.test_size = 150
        self.top_k = 5
        self.retrieval_k = 50
        self.fusion_method = "weighted"
        self.alpha = 0.6
        self.benchmark_queries = 100
        
        # Output paths
        self.output_dir = "evaluation_output"
        self.report_path = "evaluation_output/eval_report.json"
        self.html_report = "evaluation_output/eval_report.html"
        self.csv_results = "evaluation_output/results.csv"
        
        # Categories to test
        self.categories = [
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
        ]


# Known disease names for verification (from your database)
KNOWN_DISEASES = [
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