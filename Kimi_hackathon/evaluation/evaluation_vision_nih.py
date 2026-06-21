#!/usr/bin/env python3
"""
evaluation_vision_nih.py - Test Gemini with REAL NIH Chest X-ray images
"""

import os
import sys
import time
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from rag_engine_v4 import analyze_medical_image

# ── Configuration ──────────────────────────────────────────────────────────

NIH_DIR = "nih_chest_xray"
CSV_FILE = os.path.join(NIH_DIR, "Data_Entry_2017_v2020.csv")
IMAGE_DIR = os.path.join(NIH_DIR, "images")
OUTPUT_DIR = "evaluation_output"

# ── Load Metadata ──────────────────────────────────────────────────────────

def load_metadata():
    """Load the NIH dataset metadata"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        print("   Please download Data_Entry_2017_v2020.csv from NIH Box")
        return None
    
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df)} image entries")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Finding labels: {df['Finding Labels'].unique()[:5]}...")
    return df

# ── Analyze Images ────────────────────────────────────────────────────────

def analyze_nih_images(df, num_images=10):
    """Analyze NIH images with Gemini"""
    
    print("\n" + "="*70)
    print("🩻 GEMINI 2.5 FLASH - NIH CHEST X-RAY EVALUATION")
    print("="*70)
    
    # Check image directory
    if not os.path.exists(IMAGE_DIR):
        print(f"\n❌ Image folder not found: {IMAGE_DIR}")
        print("   Please download the 'images' folder from NIH Box")
        return
    
    # Select images to test
    # Get a mix of normal and diseased images
    normal_df = df[df['Finding Labels'] == 'No Finding']
    disease_df = df[df['Finding Labels'] != 'No Finding']
    
    print(f"\n📊 Dataset stats:")
    print(f"   Normal images: {len(normal_df)}")
    print(f"   Diseased images: {len(disease_df)}")
    
    # Sample 5 normal and 5 diseased
    normal_samples = normal_df.sample(min(5, len(normal_df)))
    disease_samples = disease_df.sample(min(5, len(disease_df)))
    
    test_samples = pd.concat([normal_samples, disease_samples])
    
    print(f"\n📤 Testing {len(test_samples)} images...")
    print("-"*70)
    
    results = []
    total_time = 0
    
    for idx, row in test_samples.iterrows():
        image_file = row['Image Index']
        expected_labels = row['Finding Labels']
        
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"\n⚠️ Image not found: {image_file}")
            continue
        
        print(f"\n📤 [{idx+1}] {image_file}")
        print(f"   Expected: {expected_labels}")
        
        try:
            # Load and analyze image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            t0 = time.time()
            result = analyze_medical_image(image_bytes, "image/png")
            elapsed = time.time() - t0
            total_time += elapsed
            
            # Display results
            print(f"   ⏱️  Time: {elapsed:.2f}s")
            print(f"   📊 Modality: {result.get('modality', 'Unknown')}")
            print(f"   📍 Region: {result.get('body_region', 'Unknown')}")
            print(f"   🔍 Findings:")
            for finding in result.get('key_findings', [])[:3]:
                print(f"      • {finding}")
            
            # Store
            results.append({
                "image": image_file,
                "expected_labels": expected_labels,
                "modality": result.get('modality'),
                "region": result.get('body_region'),
                "findings": result.get('key_findings', []),
                "symptoms": result.get('suggested_symptoms', []),
                "quality": result.get('image_quality', 'unknown'),
                "time": elapsed
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "image": image_file,
                "expected_labels": expected_labels,
                "error": str(e)
            })
    
    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if 'error' not in r)
    total = len(results)
    
    print(f"  ✅ Successful: {successful}/{total}")
    
    if results:
        times = [r.get('time', 0) for r in results if r.get('time', 0) > 0]
        if times:
            print(f"  ⏱️  Avg time: {sum(times)/len(times):.2f}s")
            print(f"  ⏱️  Total time: {sum(times):.2f}s")
    
    # ── Save Results ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "gemini-2.5-flash",
        "dataset": "NIH Chest X-ray",
        "total_images": total,
        "successful": successful,
        "results": results
    }
    
    output_path = os.path.join(OUTPUT_DIR, "nih_vision_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Results saved to: {output_path}")
    
    # Save CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "nih_vision_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"📄 CSV saved to: {csv_path}")
    
    # Markdown summary
    md_path = os.path.join(OUTPUT_DIR, "nih_vision_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NIH Chest X-ray - Gemini 2.5 Flash Evaluation\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Images analyzed:** {total}\n")
        f.write(f"- **Successful:** {successful}\n")
        if times:
            f.write(f"- **Average time:** {sum(times)/len(times):.2f}s\n\n")
        
        f.write("## Results by Image\n\n")
        f.write("| Image | Expected Labels | Modality | Region | Quality |\n")
        f.write("|-------|-----------------|----------|--------|---------|\n")
        for r in results:
            expected = r.get('expected_labels', 'N/A')[:30]
            modality = r.get('modality', 'N/A')
            region = r.get('region', 'N/A')
            quality = r.get('quality', 'N/A')
            status = "✅" if 'error' not in r else "❌"
            f.write(f"| {status} {r['image']} | {expected} | {modality} | {region} | {quality} |\n")
        
        f.write("\n## Sample Findings\n\n")
        for r in results[:3]:
            if 'error' not in r:
                f.write(f"### {r['image']}\n")
                f.write(f"- **Expected:** {r['expected_labels']}\n")
                f.write(f"- **Gemini Findings:**\n")
                for finding in r.get('findings', [])[:3]:
                    f.write(f"  - {finding}\n")
                f.write("\n")
    
    print(f"📄 Summary saved to: {md_path}")
    print("\n" + "="*70)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        print("   Please create .env with: GOOGLE_API_KEY=your_key")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Load metadata
    df = load_metadata()
    if df is None:
        return
    
    # Analyze images
    analyze_nih_images(df, num_images=10)

if __name__ == "__main__":
    main()