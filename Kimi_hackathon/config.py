"""
config.py - Configuration for MediNova
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ModelConfig:
    """Configuration for model APIs"""
    def __init__(self):
        # Google Gemini 2.5 Flash for Vision
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.gemini_model = "gemini-2.5-flash"
        self.gemini_temperature = 0.2
        
        # Hugging Face Token (for MedCPT)
        self.hf_token = os.environ.get("HF_TOKEN", "")
        
        # MedCPT for embeddings
        self.medcpt_model = "ncbi/MedCPT-Query-Encoder"
        
        # Check if API keys are set
        if not self.google_api_key:
            print("⚠️ GOOGLE_API_KEY not set in .env file!")
            print("   Please copy .env.example to .env and add your API key")
            print("   Get your key from: https://ai.google.dev/")
        
        if not self.hf_token:
            print("⚠️ HF_TOKEN not set in .env file!")
            print("   Please add HF_TOKEN to .env for MedCPT embeddings")
            print("   Get your token from: https://huggingface.co/settings/tokens")


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
        
        # Data paths
        self.data_dir = "data"
        self.results_dir = "results"
        
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


# Known disease names for verification
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

# Constants
DATA_DIR = "data"
RESULTS_DIR = "results"
EVAL_DIR = "evaluation_output"
NIH_DIR = "nih_chest_xray"

# File paths
PRODUCT1_PATH = os.path.join(DATA_DIR, "en_product1.xml")
PRODUCT4_PATH = os.path.join(DATA_DIR, "en_product4.xml")
DB_PATH = os.path.join(DATA_DIR, "hpo_disease_db_v3.pkl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index_medcpt_v4.bin")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss_meta_medcpt_v4.pkl")
HPO_OBO_PATH = os.path.join(DATA_DIR, "hp.obo")
TEST_DATASET_PATH = os.path.join(DATA_DIR, "test_dataset.json")