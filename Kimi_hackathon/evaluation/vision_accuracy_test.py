#!/usr/bin/env python3
"""
vision_accuracy_test.py - Calculate Vision Model Accuracy
Tests Gemini 2.5 Flash on real NIH Chest X-ray images
"""

import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
load_dotenv()

from rag_engine_v4 import analyze_medical_image

# ── Configuration ──────────────────────────────────────────────────────────

NIH_DIR = "nih_chest_xray"
CSV_FILE = os.path.join(NIH_DIR, "Data_Entry_2017_v2020.csv")
IMAGE_DIR = os.path.join(NIH_DIR, "images")
OUTPUT_DIR = "evaluation_output"

# ── Load Metadata ──────────────────────────────────────────────────────────

def load_metadata():
    """Load NIH dataset metadata"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV not found: {CSV_FILE}")
        return None
    
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df)} images")
    return df

# ── Accuracy Calculator ──────────────────────────────────────────────────

class VisionAccuracyCalculator:
    def __init__(self):
        self.results = []
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.disease_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
    def extract_diseases(self, labels):
        """Extract disease labels from string"""
        if labels == 'No Finding':
            return ['Normal']
        return labels.split('|')
    
    def calculate_metrics(self, predicted_findings, expected_labels):
        """
        Calculate accuracy metrics between predicted and expected
        """
        # Parse expected diseases
        expected_diseases = set(self.extract_diseases(expected_labels))
        predicted_diseases = set([f.lower() for f in predicted_findings])
        
        # Map common variations
        disease_mapping = {
            'cardiomegaly': 'cardiomegaly',
            'effusion': 'effusion',
            'atelectasis': 'atelectasis',
            'infiltrate': 'infiltrate',
            'nodule': 'nodule',
            'pneumonia': 'pneumonia',
            'pneumothorax': 'pneumothorax',
            'edema': 'edema',
            'fibrosis': 'fibrosis',
            'hernia': 'hernia',
            'normal': 'normal',
            'no finding': 'normal'
        }
        
        # Map predicted to standard names
        mapped_predicted = set()
        for pred in predicted_diseases:
            for key, value in disease_mapping.items():
                if key in pred or pred in key:
                    mapped_predicted.add(value)
                    break
        
        # Calculate metrics
        true_positives = len(expected_diseases.intersection(mapped_predicted))
        false_positives = len(mapped_predicted - expected_diseases)
        false_negatives = len(expected_diseases - mapped_predicted)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / len(expected_diseases) if len(expected_diseases) > 0 else 0
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "expected": list(expected_diseases),
            "predicted": list(mapped_predicted)[:5]
        }
    
    def evaluate_image(self, image_path, expected_labels):
        """Evaluate a single image"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Get Gemini analysis
            result = analyze_medical_image(image_bytes, "image/png")
            predicted_findings = result.get('key_findings', [])
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_findings, expected_labels)
            
            # Store results
            self.results.append({
                "image": os.path.basename(image_path),
                "expected": expected_labels,
                "predicted": predicted_findings[:5],
                "metrics": metrics
            })
            
            # Update confusion matrix
            for expected in metrics["expected"]:
                if expected in metrics["predicted"]:
                    self.confusion_matrix[expected][expected] += 1
                    self.disease_stats[expected]["correct"] += 1
                else:
                    for pred in metrics["predicted"]:
                        self.confusion_matrix[expected][pred] += 1
                self.disease_stats[expected]["total"] += 1
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_summary(self):
        """Calculate overall summary metrics"""
        total = len(self.results)
        if total == 0:
            return {}
        
        avg_precision = np.mean([r["metrics"]["precision"] for r in self.results if r.get("metrics")])
        avg_recall = np.mean([r["metrics"]["recall"] for r in self.results if r.get("metrics")])
        avg_f1 = np.mean([r["metrics"]["f1"] for r in self.results if r.get("metrics")])
        avg_accuracy = np.mean([r["metrics"]["accuracy"] for r in self.results if r.get("metrics")])
        
        # Per-disease accuracy
        per_disease = {}
        for disease, stats in self.disease_stats.items():
            if stats["total"] > 0:
                per_disease[disease] = {
                    "accuracy": stats["correct"] / stats["total"] * 100,
                    "total": stats["total"]
                }
        
        return {
            "total_images": total,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_accuracy": avg_accuracy,
            "per_disease": per_disease
        }

# ── Main Evaluation ──────────────────────────────────────────────────────

def main():
    print("="*70)
    print("🩻 VISION MODEL ACCURACY EVALUATION")
    print("="*70)
    
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return
    
    # Load metadata
    df = load_metadata()
    if df is None:
        return
    
    # Get images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    print(f"📸 Found {len(image_files)} images")
    
    if not image_files:
        print("❌ No images found! Please download NIH images.")
        return
    
    # Sample images (mix of normal and diseased)
    sample_df = df[df['Image Index'].isin(image_files)]
    normal = sample_df[sample_df['Finding Labels'] == 'No Finding'].head(5)
    diseased = sample_df[sample_df['Finding Labels'] != 'No Finding'].head(10)
    test_df = pd.concat([normal, diseased])
    
    print(f"📊 Testing {len(test_df)} images (5 normal, 10 diseased)")
    print("-"*70)
    
    # Initialize calculator
    calculator = VisionAccuracyCalculator()
    
    # Evaluate each image
    for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df)), 1):
        image_file = row['Image Index']
        expected_labels = row['Finding Labels']
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        if not os.path.exists(image_path):
            print(f"⚠️ {image_file} not found")
            continue
        
        print(f"\n📤 [{idx}] {image_file}")
        print(f"   Expected: {expected_labels}")
        
        metrics = calculator.evaluate_image(image_path, expected_labels)
        
        if metrics:
            print(f"   ✅ Accuracy: {metrics['accuracy']*100:.1f}%")
            print(f"   Precision: {metrics['precision']:.2f}")
            print(f"   Recall: {metrics['recall']:.2f}")
            print(f"   F1: {metrics['f1']:.2f}")
            if metrics['predicted']:
                print(f"   Predicted: {', '.join(metrics['predicted'])}")
    
    # ── Summary ────────────────────────────────────────────────────────────
    summary = calculator.get_summary()
    
    print("\n" + "="*70)
    print("📊 VISION MODEL ACCURACY SUMMARY")
    print("="*70)
    
    print(f"  Total Images: {summary.get('total_images', 0)}")
    print(f"  Average Accuracy: {summary.get('avg_accuracy', 0)*100:.1f}%")
    print(f"  Average Precision: {summary.get('avg_precision', 0):.3f}")
    print(f"  Average Recall: {summary.get('avg_recall', 0):.3f}")
    print(f"  Average F1 Score: {summary.get('avg_f1', 0):.3f}")
    
    print("\n📊 PER-DISEASE ACCURACY:")
    print(f"{'Disease':<20} | {'Accuracy':>10} | {'Count':>6}")
    print("-"*40)
    for disease, stats in sorted(summary.get('per_disease', {}).items(), key=lambda x: -x[1]['accuracy']):
        print(f"{disease:<20} | {stats['accuracy']:>9.1f}% | {stats['total']:>6d}")
    
    # ── Save Results ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save full results
    output_path = os.path.join(OUTPUT_DIR, "vision_accuracy_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": summary,
            "detailed_results": calculator.results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_path}")
    
    # Save CSV
    df_results = pd.DataFrame(calculator.results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "vision_accuracy_results.csv"), index=False)
    print(f"📄 CSV saved to: {OUTPUT_DIR}/vision_accuracy_results.csv")
    
    # ── Generate Markdown Report ──────────────────────────────────────────
    md_path = os.path.join(OUTPUT_DIR, "vision_accuracy_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Vision Model Accuracy Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Images:** {summary.get('total_images', 0)}\n")
        f.write(f"- **Average Accuracy:** {summary.get('avg_accuracy', 0)*100:.1f}%\n")
        f.write(f"- **Average Precision:** {summary.get('avg_precision', 0):.3f}\n")
        f.write(f"- **Average Recall:** {summary.get('avg_recall', 0):.3f}\n")
        f.write(f"- **Average F1 Score:** {summary.get('avg_f1', 0):.3f}\n\n")
        
        f.write("## Per-Disease Accuracy\n\n")
        f.write("| Disease | Accuracy | Count |\n")
        f.write("|---------|----------|-------|\n")
        for disease, stats in sorted(summary.get('per_disease', {}).items(), key=lambda x: -x[1]['accuracy']):
            f.write(f"| {disease} | {stats['accuracy']:.1f}% | {stats['total']} |\n")
        
        f.write("\n## Detailed Results\n\n")
        f.write("| Image | Expected | Predicted | Accuracy |\n")
        f.write("|-------|----------|-----------|----------|\n")
        for r in calculator.results:
            expected = r.get('expected', 'N/A')[:20]
            predicted = ', '.join(r.get('metrics', {}).get('predicted', []))[:20]
            acc = r.get('metrics', {}).get('accuracy', 0) * 100
            f.write(f"| {r.get('image', 'N/A')} | {expected} | {predicted} | {acc:.1f}% |\n")
    
    print(f"📄 Report saved to: {md_path}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()