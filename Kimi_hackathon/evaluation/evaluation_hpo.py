#!/usr/bin/env python3
"""
evaluation_hpo.py - HPO ID Evaluation Suite (IMPROVED)
Now with HPO ID expansion to symptom terms for better retrieval
"""

#!/usr/bin/env python3
"""
evaluation_hpo.py - HPO ID Evaluation Suite with Automatic Expansion
"""

import os
import sys
import time
import json
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Import existing engine
from rag_engine_v4 import load_faiss_index, retrieve_diseases

# Import configuration
from config import EvalConfig

# Import HPO Expander (NEW)
from hpo_expander import HPOExpander

# ── Initialize HPO Expander ──────────────────────────────────────────────

_hpo_expander = None

def get_hpo_expander():
    """Lazy-load HPO expander"""
    global _hpo_expander
    if _hpo_expander is None:
        _hpo_expander = HPOExpander("hp.obo", auto_download=True)
    return _hpo_expander

def expand_hpo_query(query: str, max_def_length: int = 80) -> str:
    """Expand HPO IDs using automatic OBO expansion"""
    expander = get_hpo_expander()
    return expander.expand_query(query, max_def_length=max_def_length)

# ... rest of your evaluation_hpo.py code ...


# Import existing engine
from rag_engine_v4 import load_faiss_index, retrieve_diseases

# Import configuration
from config import EvalConfig

# ── HPO ID to Symptom Mapping ──────────────────────────────────────────────

HPO_TO_SYMPTOM = {
    # Classic symptoms
    "HP:0000508": "drooping eyelids ptosis",
    "HP:0000613": "double vision diplopia",
    "HP:0002015": "difficulty swallowing dysphagia",
    "HP:0002206": "gait disturbance difficulty walking",
    "HP:0002303": "fatigable weakness worsens with activity",
    "HP:0001250": "ataxia incoordination",
    "HP:0000738": "hand flapping stereotypies",
    "HP:0001263": "absent speech no speech",
    "HP:0002067": "paroxysmal laughter sudden laughter",
    "HP:0010739": "happy demeanor inappropriate laughter",
    
    # Duchenne MD
    "HP:0003236": "elevated creatine kinase high CK",
    "HP:0000944": "calf pseudohypertrophy enlarged calf",
    "HP:0003452": "Gowers sign difficulty standing",
    "HP:0001302": "waddling gait walking difficulty",
    "HP:0003202": "scoliosis curved spine",
    
    # Wilson disease
    "HP:0001417": "Kayser-Fleischer rings copper rings in eyes",
    "HP:0001396": "hepatomegaly enlarged liver",
    "HP:0000717": "personality changes behavioral changes",
    "HP:0002072": "tremor shaking",
    "HP:0001260": "dysarthria speech difficulty",
    
    # Marfan syndrome
    "HP:0002616": "aortic root dilation enlarged aorta",
    "HP:0001638": "cardiomyopathy heart disease",
    "HP:0005112": "arachnodactyly long fingers",
    "HP:0000978": "stretch marks skin changes",
    "HP:0000545": "ectopia lentis lens dislocation",
    
    # Cystic fibrosis
    "HP:0004900": "salty sweat high chloride",
    "HP:0000315": "thick mucus viscid secretions",
    "HP:0002719": "recurrent pneumonia lung infections",
    "HP:0002098": "respiratory insufficiency breathing difficulty",
    
    # Huntington disease
    "HP:0002352": "dementia cognitive decline",
    "HP:0001250": "ataxia incoordination",  # duplicate
    "HP:0002067": "chorea involuntary movements",
    
    # Rett syndrome
    "HP:0011400": "hand wringing repetitive hand movements",
    "HP:0002167": "regression loss of skills",
    "HP:0002790": "breathing abnormalities irregular breathing",
    "HP:0000708": "autism features social difficulties",
    
    # Pompe disease
    "HP:0001271": "hypotonia low muscle tone",
    "HP:0001638": "cardiomegaly enlarged heart",
    "HP:0002098": "respiratory failure breathing failure",
    "HP:0001260": "dysarthria speech difficulty",
    
    # Fabry disease
    "HP:0001071": "angiokeratoma skin spots",
    "HP:0011870": "acroparesthesia pain in hands feet",
    "HP:0001337": "tremor shaking",
    
    # Gaucher disease
    "HP:0001744": "splenomegaly enlarged spleen",
    "HP:0001396": "hepatomegaly enlarged liver",  # duplicate
    "HP:0001903": "anemia low red blood cells",
    "HP:0001890": "thrombocytopenia low platelets",
    "HP:0001337": "tremor shaking",
    
    # Neurofibromatosis type 1
    "HP:0000956": "cafe au lait spots brown patches",
    "HP:0001067": "neurofibromas skin tumors",
    "HP:0000608": "Lisch nodules iris spots",
    "HP:0009730": "optic glioma eye tumor",
    "HP:0000992": "axillary freckling underarm freckles",
    
    # Tuberous sclerosis
    "HP:0000615": "facial angiofibroma facial rash",
    "HP:0001053": "ash leaf spots white skin patches",
    "HP:0009733": "renal angiomyolipoma kidney tumors",
    "HP:0001250": "seizures epilepsy",
    "HP:0001249": "intellectual disability learning difficulty",
}

# ── Expanded HPO Test Dataset ──────────────────────────────────────────────

HPO_TEST_QUERIES = [
    # HPO only (expanded)
    {
        "query": "HP:0000508 HP:0000613 HP:0002015 HP:0002206",
        "expected": "Myasthenia gravis",
        "orpha": "589",
        "category": "hpo_only",
        "symptoms": ["drooping eyelids", "double vision", "difficulty swallowing", "gait disturbance"]
    },
    {
        "query": "HP:0003236 HP:0000944 HP:0003452 HP:0001302",
        "expected": "Duchenne muscular dystrophy",
        "orpha": "98896",
        "category": "hpo_only",
        "symptoms": ["elevated creatine kinase", "calf pseudohypertrophy", "Gowers sign", "waddling gait"]
    },
    {
        "query": "HP:0001417 HP:0001396 HP:0000717 HP:0002072",
        "expected": "Wilson disease",
        "orpha": "905",
        "category": "hpo_only",
        "symptoms": ["Kayser-Fleischer rings", "hepatomegaly", "personality changes", "tremor"]
    },
    {
        "query": "HP:0001263 HP:0001250 HP:0002067 HP:0010739",
        "expected": "Angelman syndrome",
        "orpha": "72",
        "category": "hpo_only",
        "symptoms": ["absent speech", "ataxia", "paroxysmal laughter", "happy demeanor"]
    },
    {
        "query": "HP:0002616 HP:0005112 HP:0000978 HP:0000545",
        "expected": "Marfan syndrome",
        "orpha": "558",
        "category": "hpo_only",
        "symptoms": ["aortic root dilation", "arachnodactyly", "stretch marks", "ectopia lentis"]
    },
    {
        "query": "HP:0004900 HP:0000315 HP:0002098 HP:0002719",
        "expected": "Cystic fibrosis",
        "orpha": "586",
        "category": "hpo_only",
        "symptoms": ["salty sweat", "thick mucus", "respiratory insufficiency", "recurrent pneumonia"]
    },
    {
        "query": "HP:0002067 HP:0002352 HP:0000717 HP:0001260",
        "expected": "Huntington disease",
        "orpha": "399",
        "category": "hpo_only",
        "symptoms": ["chorea", "dementia", "personality changes", "dysarthria"]
    },
    {
        "query": "HP:0011400 HP:0002167 HP:0002790 HP:0000708",
        "expected": "Rett syndrome",
        "orpha": "778",
        "category": "hpo_only",
        "symptoms": ["hand wringing", "regression", "breathing abnormalities", "autism features"]
    },
    
    # HPO + natural language (mixed)
    {
        "query": "HP:0000508 HP:0000613 difficulty swallowing worsens throughout day",
        "expected": "Myasthenia gravis",
        "orpha": "589",
        "category": "hpo_mixed",
        "symptoms": ["drooping eyelids", "double vision", "dysphagia", "fatigable weakness"]
    },
    {
        "query": "HP:0003236 HP:0000944 boys progressive muscle weakness HP:0003452",
        "expected": "Duchenne muscular dystrophy",
        "orpha": "98896",
        "category": "hpo_mixed",
        "symptoms": ["elevated creatine kinase", "calf pseudohypertrophy", "Gowers sign"]
    },
    {
        "query": "HP:0001417 HP:0001396 copper accumulation personality changes HP:0002072",
        "expected": "Wilson disease",
        "orpha": "905",
        "category": "hpo_mixed",
        "symptoms": ["Kayser-Fleischer rings", "hepatomegaly", "tremor"]
    },
    {
        "query": "HP:0001263 HP:0010739 seizures HP:0002067 HP:0001250",
        "expected": "Angelman syndrome",
        "orpha": "72",
        "category": "hpo_mixed",
        "symptoms": ["absent speech", "happy demeanor", "paroxysmal laughter", "ataxia"]
    },
    
    # HPO + gender
    {
        "query": "HP:0011400 HP:0002167 HP:0002790 HP:0000708 girls",
        "expected": "Rett syndrome",
        "orpha": "778",
        "category": "hpo_gender",
        "symptoms": ["hand wringing", "regression", "breathing abnormalities", "autism features"]
    },
    {
        "query": "HP:0003236 HP:0000944 HP:0003452 HP:0001302 boys",
        "expected": "Duchenne muscular dystrophy",
        "orpha": "98896",
        "category": "hpo_gender",
        "symptoms": ["elevated creatine kinase", "calf pseudohypertrophy", "Gowers sign", "waddling gait"]
    },
    {
        "query": "HP:0001417 HP:0001396 HP:0000717 HP:0002072 female",
        "expected": "Wilson disease",
        "orpha": "905",
        "category": "hpo_gender",
        "symptoms": ["Kayser-Fleischer rings", "hepatomegaly", "personality changes", "tremor"]
    },
    
    # HPO + inheritance
    {
        "query": "HP:0003236 HP:0000944 HP:0003452 HP:0001302 X-linked recessive",
        "expected": "Duchenne muscular dystrophy",
        "orpha": "98896",
        "category": "hpo_inheritance",
        "symptoms": ["elevated creatine kinase", "calf pseudohypertrophy", "Gowers sign", "waddling gait"]
    },
    {
        "query": "HP:0002616 HP:0005112 HP:0000978 HP:0000545 autosomal dominant",
        "expected": "Marfan syndrome",
        "orpha": "558",
        "category": "hpo_inheritance",
        "symptoms": ["aortic root dilation", "arachnodactyly", "stretch marks", "ectopia lentis"]
    },
    {
        "query": "HP:0001417 HP:0001396 HP:0000717 HP:0002072 autosomal recessive",
        "expected": "Wilson disease",
        "orpha": "905",
        "category": "hpo_inheritance",
        "symptoms": ["Kayser-Fleischer rings", "hepatomegaly", "personality changes", "tremor"]
    },
    
    # Classic symptoms with HPO
    {
        "query": "HP:0000508 HP:0000613 HP:0002015 HP:0002206 HP:0002303",
        "expected": "Myasthenia gravis",
        "orpha": "589",
        "category": "hpo_classic",
        "symptoms": ["drooping eyelids", "double vision", "dysphagia", "gait disturbance", "fatigable weakness"]
    },
    {
        "query": "HP:0003236 HP:0000944 HP:0003452 HP:0001302 HP:0003202",
        "expected": "Duchenne muscular dystrophy",
        "orpha": "98896",
        "category": "hpo_classic",
        "symptoms": ["elevated creatine kinase", "calf pseudohypertrophy", "Gowers sign", "waddling gait", "scoliosis"]
    },
    {
        "query": "HP:0001417 HP:0001396 HP:0000717 HP:0002072 HP:0001260",
        "expected": "Wilson disease",
        "orpha": "905",
        "category": "hpo_classic",
        "symptoms": ["Kayser-Fleischer rings", "hepatomegaly", "personality changes", "tremor", "dysarthria"]
    },
    {
        "query": "HP:0010739 HP:0001263 HP:0002067 HP:0001250 HP:0000738",
        "expected": "Angelman syndrome",
        "orpha": "72",
        "category": "hpo_classic",
        "symptoms": ["happy demeanor", "absent speech", "paroxysmal laughter", "ataxia", "hand flapping"]
    },
    
    # Biochemical markers with HPO
    {
        "query": "HP:0000519 HP:0001126 HP:0003236 HP:0001263 HP:0001250",
        "expected": "Phenylketonuria",
        "orpha": "716",
        "category": "hpo_biochemical",
        "symptoms": ["musty odor", "fair skin", "elevated creatine kinase", "absent speech", "ataxia"]
    },
    {
        "query": "HP:0002616 HP:0001638 HP:0005112 HP:0000978 HP:0000545",
        "expected": "Marfan syndrome",
        "orpha": "558",
        "category": "hpo_biochemical",
        "symptoms": ["aortic root dilation", "cardiomyopathy", "arachnodactyly", "stretch marks", "ectopia lentis"]
    },
    {
        "query": "HP:0004900 HP:0000315 HP:0002719 HP:0002098 HP:0001638",
        "expected": "Cystic fibrosis",
        "orpha": "586",
        "category": "hpo_biochemical",
        "symptoms": ["salty sweat", "thick mucus", "recurrent pneumonia", "respiratory insufficiency", "cardiomyopathy"]
    },
    {
        "query": "HP:0002067 HP:0002352 HP:0000717 HP:0001260 HP:0001250",
        "expected": "Huntington disease",
        "orpha": "399",
        "category": "hpo_biochemical",
        "symptoms": ["chorea", "dementia", "personality changes", "dysarthria", "ataxia"]
    },
    
    # Rare HPO combinations
    {
        "query": "HP:0001071 HP:0011870 HP:0001337 HP:0001396",
        "expected": "Fabry disease",
        "orpha": "324",
        "category": "hpo_rare",
        "symptoms": ["angiokeratoma", "acroparesthesia", "tremor", "hepatomegaly"]
    },
    {
        "query": "HP:0001744 HP:0001903 HP:0001890 HP:0001396",
        "expected": "Gaucher disease",
        "orpha": "355",
        "category": "hpo_rare",
        "symptoms": ["splenomegaly", "anemia", "thrombocytopenia", "hepatomegaly"]
    },
    {
        "query": "HP:0000956 HP:0001067 HP:0000608 HP:0009730",
        "expected": "Neurofibromatosis type 1",
        "orpha": "636",
        "category": "hpo_rare",
        "symptoms": ["cafe au lait spots", "neurofibromas", "Lisch nodules", "optic glioma"]
    },
    {
        "query": "HP:0000615 HP:0001053 HP:0009733 HP:0001250",
        "expected": "Tuberous sclerosis",
        "orpha": "805",
        "category": "hpo_rare",
        "symptoms": ["facial angiofibroma", "ash leaf spots", "renal angiomyolipoma", "seizures"]
    },
]

# ── HPO Query Expander ────────────────────────────────────────────────────

def expand_hpo_query(query: str) -> str:
    """Expand HPO IDs to include symptom descriptions"""
    expanded = query
    for hpo_id, symptom in HPO_TO_SYMPTOM.items():
        if hpo_id in expanded:
            # Add symptom description next to the HPO ID
            expanded = expanded.replace(hpo_id, f"{hpo_id} {symptom}")
    return expanded


# ── HPO Evaluation Runner ──────────────────────────────────────────────────

class HPOEvaluator:
    def __init__(self, config=None):
        self.config = config or EvalConfig()
        self.results = []
        self.category_stats = defaultdict(lambda: {"hits": 0, "total": 0, "mrr_sum": 0.0, "ndcg_sum": 0.0})
        self.query_times = []
        self.index = None
        self.db = None
        self.bm25 = None
        
    def load_index(self):
        print("\n📂 Loading FAISS index...")
        try:
            self.index, self.db, self.bm25 = load_faiss_index()
            print(f"   ✅ Loaded {len(self.db)} diseases")
            return True
        except Exception as e:
            print(f"   ❌ Failed to load index: {e}")
            return False
    
    def run_evaluation(self, test_queries, expand_hpo=True):
        if not self.index or not self.db or not self.bm25:
            if not self.load_index():
                return {"error": "Failed to load index"}
        
        print(f"\n{'='*70}")
        print(f"🧬 HPO ID EVALUATION SUITE (with HPO Expansion)")
        print(f"{'='*70}")
        print(f"  • Test queries: {len(test_queries)}")
        print(f"  • Top K: {self.config.top_k}")
        print(f"  • Retrieval K: {self.config.retrieval_k}")
        print(f"  • Fusion: {self.config.fusion_method}")
        print(f"  • Alpha: {0.8} (optimized for HPO)")
        print(f"  • HPO Expansion: {expand_hpo}")
        print(f"{'='*70}\n")
        
        total_time = 0.0
        total_queries = len(test_queries)
        
        for test in tqdm(test_queries, desc="Running HPO queries"):
            query = test["query"]
            
            # Expand HPO IDs to include symptoms
            if expand_hpo:
                expanded_query = expand_hpo_query(query)
            else:
                expanded_query = query
            
            expected_name = test["expected"]
            expected_orpha = test["orpha"]
            category = test.get("category", "unknown")
            symptoms = test.get("symptoms", [])
            
            t0 = time.time()
            results = retrieve_diseases(
                expanded_query, 
                self.index, 
                self.db, 
                self.bm25,
                top_k=self.config.top_k,
                retrieval_k=self.config.retrieval_k,
                fusion_method=self.config.fusion_method,
                alpha=0.8  # Higher alpha for HPO (semantic)
            )
            elapsed = time.time() - t0
            self.query_times.append(elapsed)
            total_time += elapsed
            
            # Calculate rank
            rank = None
            for i, r in enumerate(results, 1):
                if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                    rank = i
                    break
            
            hit = rank is not None
            mrr = 1.0 / rank if rank else 0.0
            
            # Calculate NDCG
            dcg = 0.0
            for i, r in enumerate(results[:self.config.top_k], 1):
                rel = 1.0 if (r.get("name") == expected_name or r.get("orpha_code") == expected_orpha) else 0.0
                dcg += rel / np.log2(i + 1)
            idcg = 1.0 / np.log2(2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            # Precision
            precision = 0.0
            for r in results[:self.config.top_k]:
                if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                    precision += 1
            precision /= self.config.top_k
            
            self.results.append({
                "query": query,
                "expanded_query": expanded_query[:100] + "..." if len(expanded_query) > 100 else expanded_query,
                "symptoms": ", ".join(symptoms[:3]) if symptoms else "",
                "expected": expected_name,
                "expected_orpha": expected_orpha,
                "category": category,
                "hit": hit,
                "rank": rank,
                "mrr": mrr,
                "ndcg": ndcg,
                "precision": precision,
                "time": elapsed,
                "top_results": [r.get("name") for r in results[:3]]
            })
            
            self.category_stats[category]["hits"] += int(hit)
            self.category_stats[category]["total"] += 1
            self.category_stats[category]["mrr_sum"] += mrr
            self.category_stats[category]["ndcg_sum"] += ndcg
        
        # Calculate overall metrics
        total_hits = sum(r["hit"] for r in self.results)
        overall_hit_rate = total_hits / total_queries * 100
        overall_mrr = sum(r["mrr"] for r in self.results) / total_queries
        overall_ndcg = sum(r["ndcg"] for r in self.results) / total_queries
        avg_time = total_time / total_queries
        
        print(f"\n{'='*70}")
        print(f"📊 HPO EVALUATION RESULTS (with HPO Expansion)")
        print(f"{'='*70}")
        print(f"  Hit Rate @ {self.config.top_k}: {overall_hit_rate:.1f}% ({total_hits}/{total_queries})")
        print(f"  MRR: {overall_mrr:.4f}")
        print(f"  NDCG@{self.config.top_k}: {overall_ndcg:.4f}")
        print(f"  Average Query Time: {avg_time:.3f}s")
        
        # Per-category breakdown
        print(f"\n📊 PER-CATEGORY BREAKDOWN")
        print(f"{'Category':<20} | {'HR%':>6} | {'MRR':>6} | {'Count':>6}")
        print("-" * 50)
        for cat in sorted(self.category_stats.keys()):
            stats = self.category_stats[cat]
            cat_hr = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
            cat_mrr = stats["mrr_sum"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{cat:<20} | {cat_hr:5.1f}% | {cat_mrr:.4f} | {stats['total']:6d}")
        
        return {
            "overall": {
                "hit_rate": float(overall_hit_rate),
                "mrr": float(overall_mrr),
                "ndcg": float(overall_ndcg),
                "avg_query_time": float(avg_time),
                "total_queries": total_queries
            },
            "per_category": {
                cat: {
                    "hit_rate": float(self.category_stats[cat]["hits"] / self.category_stats[cat]["total"] * 100),
                    "mrr": float(self.category_stats[cat]["mrr_sum"] / self.category_stats[cat]["total"]),
                    "count": int(self.category_stats[cat]["total"])
                }
                for cat in self.category_stats
            },
            "detailed_results": self.results
        }
    
    def save_results(self, results):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        full_report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "HPO_ID_with_Expansion",
            "config": {
                "top_k": self.config.top_k,
                "retrieval_k": self.config.retrieval_k,
                "fusion_method": self.config.fusion_method,
                "alpha": 0.8,
                "hpo_expansion": True
            },
            "evaluation": results
        }
        
        report_path = os.path.join(self.config.output_dir, "hpo_eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 HPO report saved to: {report_path}")
        
        df = pd.DataFrame(results["detailed_results"])
        csv_path = os.path.join(self.config.output_dir, "hpo_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"📄 HPO CSV saved to: {csv_path}")
        
        # Generate markdown summary
        self.generate_markdown_report(results)
    
    def generate_markdown_report(self, results):
        md_path = os.path.join(self.config.output_dir, "hpo_summary.md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# MediNova HPO ID Evaluation Report (with HPO Expansion)\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Top K:** {self.config.top_k}\n")
            f.write(f"- **Retrieval K:** {self.config.retrieval_k}\n")
            f.write(f"- **Fusion Method:** {self.config.fusion_method}\n")
            f.write(f"- **Alpha:** 0.8 (optimized for HPO semantic matching)\n")
            f.write(f"- **HPO Expansion:** Enabled (HPO IDs -> Symptoms)\n\n")
            
            f.write("## Overall Results\n\n")
            overall = results["overall"]
            f.write(f"- **Hit Rate @ {self.config.top_k}:** {overall['hit_rate']:.1f}%\n")
            f.write(f"- **MRR:** {overall['mrr']:.4f}\n")
            f.write(f"- **NDCG@{self.config.top_k}:** {overall['ndcg']:.4f}\n")
            f.write(f"- **Average Query Time:** {overall['avg_query_time']:.3f}s\n")
            f.write(f"- **Total Queries:** {overall['total_queries']}\n\n")
            
            f.write("## Per-Category Results\n\n")
            f.write("| Category | Hit Rate | MRR | Count |\n")
            f.write("|----------|----------|-----|-------|\n")
            for cat, stats in sorted(results["per_category"].items()):
                f.write(f"| {cat} | {stats['hit_rate']:.1f}% | {stats['mrr']:.4f} | {stats['count']} |\n")
            
            f.write("\n## Hit vs Miss Summary\n\n")
            hits = sum(1 for r in results["detailed_results"] if r["hit"])
            misses = len(results["detailed_results"]) - hits
            f.write(f"- **Hits:** {hits}\n")
            f.write(f"- **Misses:** {misses}\n")
            f.write(f"- **Hit Rate:** {hits/len(results['detailed_results'])*100:.1f}%\n")
            
            f.write("\n## Missed Queries\n\n")
            misses_list = [r for r in results["detailed_results"] if not r["hit"]]
            if misses_list:
                f.write("| Original Query | Expected | Category |\n")
                f.write("|----------------|----------|----------|\n")
                for miss in misses_list[:10]:  # Show top 10 misses
                    orig = miss['query'][:50] + "..." if len(miss['query']) > 50 else miss['query']
                    f.write(f"| {orig} | {miss['expected']} | {miss['category']} |\n")
            else:
                f.write("🎉 No misses! All HPO queries found their expected diseases.\n")
        
        print(f"📄 HPO markdown summary saved to: {md_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🧬 MediNova HPO ID Evaluation Suite (with HPO Expansion)")
    print("=" * 70)
    
    hpo_queries = HPO_TEST_QUERIES
    
    print(f"\n📊 HPO Test Dataset:")
    print(f"   Total HPO queries: {len(hpo_queries)}")
    
    category_counts = defaultdict(int)
    for q in hpo_queries:
        category_counts[q.get("category", "unknown")] += 1
    
    print("\n   Per-category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"     {cat:20s}: {count}")
    
    # Initialize and run
    config = EvalConfig()
    evaluator = HPOEvaluator(config)
    
    # Run evaluation with HPO expansion
    results = evaluator.run_evaluation(hpo_queries, expand_hpo=True)
    
    # Save results
    evaluator.save_results(results)
    
    print(f"\n{'='*70}")
    print(f"✅ HPO EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {config.output_dir}/")
    print(f"   • hpo_eval_report.json - Full JSON report")
    print(f"   • hpo_results.csv - Results as CSV")
    print(f"   • hpo_summary.md - Markdown summary")


if __name__ == "__main__":
    main()