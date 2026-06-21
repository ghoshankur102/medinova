#!/usr/bin/env python3
"""
test_nih_vision.py - Test Gemini Vision on NIH Chest X-ray Images
"""

import os
import sys
import time
import json
from dotenv import load_dotenv
load_dotenv()

from rag_engine_v4 import analyze_medical_image

# ── Configuration ──────────────────────────────────────────────────────────

IMAGE_DIR = "nih_chest_xray/images"
OUTPUT_DIR = "evaluation_output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "nih_vision_test_results.json")

# ── Check Setup ───────────────────────────────────────────────────────────

def check_setup():
    """Check if everything is set up correctly"""
    
    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file!")
        print("   Please create .env with: GOOGLE_API_KEY=your_key")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Check image directory
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Image folder not found: {IMAGE_DIR}")
        print("   Please download NIH images to this folder")
        return False
    
    # Check for images
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"❌ No images found in {IMAGE_DIR}")
        return False
    
    print(f"✅ Found {len(images)} images in {IMAGE_DIR}")
    return True

# ── Test Single Image ─────────────────────────────────────────────────────

def test_single_image(image_path):
    """Test a single image with Gemini"""
    
    print(f"\n📤 Testing: {os.path.basename(image_path)}")
    print("-" * 40)
    
    try:
        # Load image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"   📦 Size: {len(image_bytes)/1024:.1f} KB")
        
        # Analyze
        t0 = time.time()
        result = analyze_medical_image(image_bytes, "image/png")
        elapsed = time.time() - t0
        
        # Display results
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Modality: {result.get('modality', 'Unknown')}")
        print(f"   📍 Region: {result.get('body_region', 'Unknown')}")
        print(f"   🔍 Findings:")
        for finding in result.get('key_findings', [])[:5]:
            print(f"      • {finding}")
        print(f"   🏷️  Quality: {result.get('image_quality', 'Unknown')}")
        
        return result, elapsed
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, 0

# ── Test Multiple Images ──────────────────────────────────────────────────

def test_multiple_images(image_files, max_images=10):
    """Test multiple images"""
    
    results = []
    total_time = 0
    
    # Limit number of images
    if len(image_files) > max_images:
        print(f"\n📊 Testing first {max_images} of {len(image_files)} images")
        image_files = image_files[:max_images]
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(IMAGE_DIR, img_file)
        print(f"\n[{i}/{len(image_files)}] {img_file}")
        
        result, elapsed = test_single_image(img_path)
        
        if result:
            results.append({
                "image": img_file,
                "modality": result.get('modality', 'Unknown'),
                "region": result.get('body_region', 'Unknown'),
                "findings": result.get('key_findings', []),
                "symptoms": result.get('suggested_symptoms', []),
                "quality": result.get('image_quality', 'Unknown'),
                "time": elapsed
            })
            total_time += elapsed
    
    return results, total_time

# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(results, total_time):
    """Print summary of results"""
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total = len(results)
    print(f"  Total Images: {total}")
    
    # Modality distribution
    modalities = {}
    for r in results:
        m = r.get('modality', 'Unknown')
        modalities[m] = modalities.get(m, 0) + 1
    
    print("\n  📊 Modality Distribution:")
    for m, count in sorted(modalities.items(), key=lambda x: -x[1]):
        print(f"     {m}: {count} ({count/total*100:.1f}%)")
    
    # Region distribution
    regions = {}
    for r in results:
        reg = r.get('region', 'Unknown')
        regions[reg] = regions.get(reg, 0) + 1
    
    print("\n  📍 Region Distribution:")
    for reg, count in sorted(regions.items(), key=lambda x: -x[1]):
        print(f"     {reg}: {count} ({count/total*100:.1f}%)")
    
    # Quality distribution
    qualities = {}
    for r in results:
        q = r.get('quality', 'Unknown')
        qualities[q] = qualities.get(q, 0) + 1
    
    print("\n  🏷️  Quality Distribution:")
    for q, count in sorted(qualities.items(), key=lambda x: -x[1]):
        print(f"     {q}: {count} ({count/total*100:.1f}%)")
    
    # Average time
    times = [r['time'] for r in results if r.get('time', 0) > 0]
    if times:
        print(f"\n  ⏱️  Average Time: {sum(times)/len(times):.2f}s")
        print(f"  ⏱️  Total Time: {sum(times):.2f}s")
    
    # Sample findings
    print("\n  🔍 Sample Findings:")
    for r in results[:3]:
        img = r['image'][:20]
        findings = r.get('findings', [])[:2]
        print(f"     {img}: {', '.join(findings) if findings else 'No findings'}")

# ── Save Results ──────────────────────────────────────────────────────────

def save_results(results, total_time):
    """Save results to JSON and CSV"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "gemini-2.5-flash",
        "dataset": "NIH Chest X-ray",
        "total_images": len(results),
        "total_time": total_time,
        "results": results
    }
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Results saved to: {RESULTS_FILE}")
    
    # CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "nih_vision_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 CSV saved to: {csv_path}")
    
    # Markdown
    md_path = os.path.join(OUTPUT_DIR, "nih_vision_test_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# NIH Chest X-ray - Vision Test Results\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Images:** {len(results)}\n")
        f.write(f"- **Total Time:** {total_time:.2f}s\n")
        if results and results[0].get('time'):
            avg_time = sum(r['time'] for r in results) / len(results)
            f.write(f"- **Average Time:** {avg_time:.2f}s\n")
        
        f.write("\n## Results Table\n\n")
        f.write("| Image | Modality | Region | Quality | Time |\n")
        f.write("|-------|----------|--------|---------|------|\n")
        for r in results:
            f.write(f"| {r['image']} | {r.get('modality', 'N/A')} | {r.get('region', 'N/A')} | {r.get('quality', 'N/A')} | {r.get('time', 0):.2f}s |\n")
        
        f.write("\n## Sample Findings\n\n")
        for r in results[:5]:
            f.write(f"### {r['image']}\n")
            f.write(f"- **Modality:** {r.get('modality', 'N/A')}\n")
            f.write(f"- **Region:** {r.get('region', 'N/A')}\n")
            f.write(f"- **Quality:** {r.get('quality', 'N/A')}\n")
            f.write(f"- **Findings:**\n")
            for finding in r.get('findings', [])[:3]:
                f.write(f"  - {finding}\n")
            f.write("\n")
    
    print(f"📄 Summary saved to: {md_path}")

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("🩻 NIH CHEST X-RAY - VISION TEST")
    print("="*60)
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return
    
    # Get images
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📸 Found {len(images)} images:")
    for img in images[:5]:
        print(f"   - {img}")
    if len(images) > 5:
        print(f"   ... and {len(images) - 5} more")
    
    # Ask user how many to test
    print("\n" + "-"*60)
    print("Options:")
    print("  1. Test ALL images")
    print("  2. Test first 5 images")
    print("  3. Test first 10 images")
    print("  4. Test specific image")
    print("  5. Test first 3 images (quick)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        max_images = len(images)
    elif choice == "2":
        max_images = 5
    elif choice == "3":
        max_images = 10
    elif choice == "4":
        # Show images and let user pick
        print("\nAvailable images:")
        for i, img in enumerate(images[:20], 1):
            print(f"  {i}. {img}")
        if len(images) > 20:
            print(f"  ... and {len(images) - 20} more")
        
        try:
            idx = int(input("\nEnter image number: ")) - 1
            if 0 <= idx < len(images):
                result, elapsed = test_single_image(os.path.join(IMAGE_DIR, images[idx]))
                if result:
                    save_results([{
                        "image": images[idx],
                        "modality": result.get('modality', 'Unknown'),
                        "region": result.get('body_region', 'Unknown'),
                        "findings": result.get('key_findings', []),
                        "symptoms": result.get('suggested_symptoms', []),
                        "quality": result.get('image_quality', 'Unknown'),
                        "time": elapsed
                    }], elapsed)
                return
        except ValueError:
            print("❌ Invalid input")
            return
    else:
        max_images = 3
        print("\n📊 Testing first 3 images (quick test)...")
    
    # Run tests
    results, total_time = test_multiple_images(images, max_images)
    
    # Print summary
    print_summary(results, total_time)
    
    # Save results
    save_results(results, total_time)
    
    print("\n" + "="*60)
    print("✅ VISION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()