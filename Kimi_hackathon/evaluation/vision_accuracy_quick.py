#!/usr/bin/env python3
"""
vision_accuracy_quick.py - Quick vision accuracy test
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()

from rag_engine_v4 import analyze_medical_image
from PIL import Image
import io

def quick_test():
    print("="*60)
    print("🩻 QUICK VISION ACCURACY TEST")
    print("="*60)
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color=(50, 50, 70))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 174, 174], outline='white', width=3)
    draw.text((10, 10), "X-RAY TEST", fill='white')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    print("📤 Analyzing test image...")
    result = analyze_medical_image(img_bytes.getvalue(), "image/png")
    
    print("\n📊 Results:")
    print(f"  Modality: {result.get('modality', 'Unknown')}")
    print(f"  Region: {result.get('body_region', 'Unknown')}")
    print(f"  Findings: {result.get('key_findings', [])}")
    print(f"  Quality: {result.get('image_quality', 'Unknown')}")
    
    return result

if __name__ == "__main__":
    quick_test()