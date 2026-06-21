#!/usr/bin/env python3
"""
evaluation_runner.py - Complete evaluation suite for MediNova
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

# Import configuration from config.py (NEW)
from config import EvalConfig

# Import dataset generator
from dataset_generator_v2 import generate_test_dataset

# ── Metrics Calculator ──────────────────────────────────────────────────────

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def hit_rate(results, expected_name, expected_orpha, k):
        for r in results[:k]:
            if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                return 1.0
        return 0.0
    
    @staticmethod
    def reciprocal_rank(results, expected_name, expected_orpha):
        for i, r in enumerate(results, 1):
            if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(results, expected_name, expected_orpha, k):
        dcg = 0.0
        for i, r in enumerate(results[:k], 1):
            rel = 1.0 if (r.get("name") == expected_name or r.get("orpha_code") == expected_orpha) else 0.0
            dcg += rel / np.log2(i + 1)
        idcg = 1.0 / np.log2(2)
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(results, expected_name, expected_orpha, k):
        relevant = 0
        for r in results[:k]:
            if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                relevant += 1
        return relevant / k
    
    @staticmethod
    def calculate_rank(results, expected_name, expected_orpha):
        for i, r in enumerate(results, 1):
            if r.get("name") == expected_name or r.get("orpha_code") == expected_orpha:
                return i
        return None


# ── Evaluation Runner ──────────────────────────────────────────────────────

class EvaluationRunner:
    def __init__(self, config=None):
        self.config = config or EvalConfig()
        self.calculator = MetricsCalculator()
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
    
    def run_evaluation(self, test_queries):
        if not self.index or not self.db or not self.bm25:
            if not self.load_index():
                return {"error": "Failed to load index"}
        
        print(f"\n{'='*70}")
        print(f"🧪 MEDINOVA EVALUATION SUITE v2.0")
        print(f"{'='*70}")
        print(f"  • Test queries: {len(test_queries)}")
        print(f"  • Top K: {self.config.top_k}")
        print(f"  • Retrieval K: {self.config.retrieval_k}")
        print(f"  • Fusion: {self.config.fusion_method}")
        print(f"  • Alpha: {self.config.alpha}")
        print(f"{'='*70}\n")
        
        total_time = 0.0
        total_queries = len(test_queries)
        
        for test in tqdm(test_queries, desc="Running queries"):
            query = test["query"]
            expected_name = test["expected_disease"]
            expected_orpha = test["orpha_code"]
            category = test.get("category", "unknown")
            
            t0 = time.time()
            results = retrieve_diseases(
                query, self.index, self.db, self.bm25,
                top_k=self.config.top_k,
                retrieval_k=self.config.retrieval_k,
                fusion_method=self.config.fusion_method,
                alpha=self.config.alpha
            )
            elapsed = time.time() - t0
            self.query_times.append(elapsed)
            total_time += elapsed
            
            rank = self.calculator.calculate_rank(results, expected_name, expected_orpha)
            hit = rank is not None
            mrr = 1.0 / rank if rank else 0.0
            ndcg = self.calculator.ndcg_at_k(results, expected_name, expected_orpha, self.config.top_k)
            precision = self.calculator.precision_at_k(results, expected_name, expected_orpha, self.config.top_k)
            
            self.results.append({
                "query": query,
                "expected": expected_name,
                "expected_orpha": expected_orpha,
                "category": category,
                "hit": hit,
                "rank": rank,
                "mrr": mrr,
                "ndcg": ndcg,
                "precision": precision,
                "time": elapsed
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
        print(f"📊 OVERALL RESULTS")
        print(f"{'='*70}")
        print(f"  Hit Rate @ {self.config.top_k}: {overall_hit_rate:.1f}% ({total_hits}/{total_queries})")
        print(f"  MRR: {overall_mrr:.4f}")
        print(f"  NDCG@{self.config.top_k}: {overall_ndcg:.4f}")
        print(f"  Average Query Time: {avg_time:.3f}s")
        
        print(f"\n📊 PER-CATEGORY BREAKDOWN")
        print(f"{'Category':<25} | {'HR%':>6} | {'MRR':>6} | {'Count':>6}")
        print("-" * 55)
        for cat in sorted(self.category_stats.keys()):
            stats = self.category_stats[cat]
            cat_hr = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
            cat_mrr = stats["mrr_sum"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{cat:<25} | {cat_hr:5.1f}% | {cat_mrr:.4f} | {stats['total']:6d}")
        
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
    
    def run_benchmark(self, n_queries=100):
        print(f"\n{'='*70}")
        print(f"⚡ PERFORMANCE BENCHMARK")
        print(f"{'='*70}")
        
        if self.results:
            base_queries = [r["query"] for r in self.results]
        else:
            base_queries = [
                "progressive muscle weakness calf pseudohypertrophy",
                "drooping eyelids double vision difficulty swallowing",
                "copper accumulation liver disease personality changes"
            ]
        
        test_queries = [random.choice(base_queries) for _ in range(n_queries)]
        latencies = []
        t_start = time.time()
        
        for query in tqdm(test_queries, desc="Benchmarking"):
            t0 = time.time()
            _ = retrieve_diseases(query, self.index, self.db, self.bm25, top_k=self.config.top_k)
            latencies.append(time.time() - t0)
        
        total_time = time.time() - t_start
        latencies = np.array(latencies)
        
        print(f"\n  Queries executed: {n_queries}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {n_queries/total_time:.1f} queries/sec")
        print(f"  Mean latency: {np.mean(latencies):.4f}s")
        print(f"  P95 latency: {np.percentile(latencies, 95):.4f}s")
        print(f"  P99 latency: {np.percentile(latencies, 99):.4f}s")
        
        return {
            "queries": n_queries,
            "total_time": float(total_time),
            "throughput": float(n_queries / total_time),
            "mean_latency": float(np.mean(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99))
        }
    
    def save_results(self, results, benchmark=None):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        full_report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "top_k": self.config.top_k,
                "retrieval_k": self.config.retrieval_k,
                "fusion_method": self.config.fusion_method,
                "alpha": self.config.alpha
            },
            "evaluation": results,
            "benchmark": benchmark
        }
        
        report_path = os.path.join(self.config.output_dir, "eval_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\n📄 Full report saved to: {report_path}")
        
        df = pd.DataFrame(results["detailed_results"])
        csv_path = os.path.join(self.config.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV results saved to: {csv_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🧬 MediNova Evaluation Suite v2.0")
    print("=" * 70)
    
    dataset_path = "test_dataset.json"
    
    if os.path.exists(dataset_path):
        print(f"\n📂 Loading existing dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            data = json.load(f)
            test_queries = data["queries"]
            print(f"   ✅ Loaded {len(test_queries)} queries")
    else:
        print(f"\n📊 Generating new test dataset (150 queries)...")
        test_queries, metadata = generate_test_dataset(150)
        with open(dataset_path, "w") as f:
            json.dump({"metadata": metadata, "queries": test_queries}, f, indent=2)
        print(f"   ✅ Generated and saved {len(test_queries)} queries")
    
    config = EvalConfig()
    runner = EvaluationRunner(config)
    results = runner.run_evaluation(test_queries)
    benchmark = runner.run_benchmark(n_queries=config.benchmark_queries)
    runner.save_results(results, benchmark)
    
    print(f"\n{'='*70}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Results saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()