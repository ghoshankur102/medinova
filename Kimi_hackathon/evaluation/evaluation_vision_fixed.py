#!/usr/bin/env python3
"""
evaluation_vision_fixed.py - Fixed Vision Evaluation with Progress Output
"""

import os
import sys
import time
import json
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import io

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Check API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY not found in .env file!")
    sys.exit(1)

print(f"✅ Google API Key loaded: {GOOGLE_API_KEY[:10]}...")

# Import Gemini function
from rag_engine_v4 import analyze_medical_image

# ── Test Cases ─────────────────────────────────────────────────────────────

VISION_TEST_CASES = [
    {
        "id": "chest_xray_001",
        "modality": "X-ray",
        "body_region": "chest",
        "expected_findings": ["cardiomegaly", "infiltrates"],
        "expected_disease_indicators": ["heart failure", "pneumonia"],
        "hard": False
    },
    {
        "id": "chest_xray_002",
        "modality": "X-ray",
        "body_region": "chest",
        "expected_findings": ["aortic root dilation"],
        "expected_disease_indicators": ["Marfan syndrome"],
        "hard": False
    },
    {
        "id": "brain_mri_001",
        "modality": "MRI",
        "body_region": "brain",
        "expected_findings": ["cortical tubers"],
        "expected_disease_indicators": ["Tuberous sclerosis"],
        "hard": False
    },
    {
        "id": "bone_xray_001",
        "modality": "X-ray",
        "body_region": "extremity",
        "expected_findings": ["arachnodactyly"],
        "expected_disease_indicators": ["Marfan syndrome"],
        "hard": False
    },
    {
        "id": "abdominal_ultrasound_001",
        "modality": "Ultrasound",
        "body_region": "abdomen",
        "expected_findings": ["hepatomegaly", "splenomegaly"],
        "expected_disease_indicators": ["Gaucher disease"],
        "hard": False
    },
    {
        "id": "skin_photo_001",
        "modality": "Other",
        "body_region": "skin",
        "expected_findings": ["cafe au lait spots"],
        "expected_disease_indicators": ["Neurofibromatosis type 1"],
        "hard": False
    },
]

# ── Image Generator ──────────────────────────────────────────────────────

def create_synthetic_image(test_case: Dict) -> bytes:
    """Create synthetic image based on test case"""
    img = Image.new('RGB', (224, 224), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)
    
    # Draw shapes to simulate findings
    draw.ellipse([50, 50, 174, 174], outline='white', width=4)
    draw.rectangle([80, 80, 144, 144], outline='white', width=3)
    draw.text((10, 10), test_case['id'], fill='white')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

# ── Metrics ──────────────────────────────────────────────────────────────

def calculate_metrics(predicted: Dict, expected: Dict) -> Dict:
    """Calculate accuracy metrics"""
    expected_findings = set([f.lower() for f in expected.get('expected_findings', [])])
    predicted_findings = set([f.lower() for f in predicted.get('key_findings', [])])
    
    overlap = expected_findings.intersection(predicted_findings)
    precision = len(overlap) / len(predicted_findings) if predicted_findings else 0
    recall = len(overlap) / len(expected_findings) if expected_findings else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "findings_found": len(overlap) > 0,
        "overlap": list(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ── Main Evaluation ──────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("🩻 GEMINI 2.5 FLASH VISION EVALUATION")
    print("="*70)
    print(f"  • Test cases: {len(VISION_TEST_CASES)}")
    print(f"  • Model: Gemini 2.5 Flash")
    print("="*70 + "\n")
    
    results = []
    total_time = 0
    
    for test in tqdm(VISION_TEST_CASES, desc="Processing images"):
        test_id = test['id']
        print(f"\n📤 {test_id}...", end=" ", flush=True)
        
        # Create image
        image_bytes = create_synthetic_image(test)
        
        # Analyze
        t0 = time.time()
        try:
            analysis = analyze_medical_image(image_bytes, "image/png")
            elapsed = time.time() - t0
            total_time += elapsed
            
            # Calculate metrics
            metrics = calculate_metrics(analysis, test)
            
            # Store result
            result = {
                "id": test_id,
                "modality": test['modality'],
                "region": test['body_region'],
                "expected_findings": test['expected_findings'],
                "predicted_findings": analysis.get('key_findings', [])[:3],
                "findings_found": metrics['findings_found'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1": metrics['f1'],
                "time": elapsed
            }
            results.append(result)
            
            # Print result
            status = "✅" if metrics['findings_found'] else "❌"
            print(f"{status} F1: {metrics['f1']:.2f} ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "id": test_id,
                "findings_found": False,
                "error": str(e),
                "time": 0
            })
    
    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r.get('findings_found', False))
    total = len(results)
    
    print(f"  Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if results:
        valid = [r for r in results if 'precision' in r]
        if valid:
            avg_precision = np.mean([r['precision'] for r in valid])
            avg_recall = np.mean([r['recall'] for r in valid])
            avg_f1 = np.mean([r['f1'] for r in valid])
            avg_time = total_time / total
            
            print(f"  Avg Precision: {avg_precision:.3f}")
            print(f"  Avg Recall: {avg_recall:.3f}")
            print(f"  Avg F1 Score: {avg_f1:.3f}")
            print(f"  Avg Time: {avg_time:.2f}s")
    
    # ── Save Results ──────────────────────────────────────────────────────
    os.makedirs("evaluation_output", exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "gemini-2.5-flash",
        "test_cases": total,
        "successful": successful,
        "results": results
    }
    
    with open("evaluation_output/vision_fixed_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Results saved to: evaluation_output/vision_fixed_report.json")
    
    # CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_output/vision_fixed_results.csv", index=False)
    print(f"📄 Results saved to: evaluation_output/vision_fixed_results.csv")
    
    print("\n" + "="*70)
    print("✅ VISION EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()