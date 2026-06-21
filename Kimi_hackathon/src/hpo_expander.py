#!/usr/bin/env python3
"""
hpo_expander.py - Automatic HPO ID Expansion using Official HPO OBO File
NO MANUAL MAPPING REQUIRED!
"""

import re
import json
import os
from typing import Dict, List, Optional, Set

# Try to import requests, but provide fallback
try:
    import requests
except ImportError:
    requests = None
    print("⚠️ 'requests' not installed. Run: pip install requests")

# ── Download Official HPO OBO File ──────────────────────────────────────

def download_hpo_obo(output_path: str = "hp.obo"):
    """Download the latest HPO OBO file from official source"""
    url = "http://purl.obolibrary.org/obo/hp.obo"
    
    if requests is None:
        print("❌ Cannot download: 'requests' module not installed")
        print("   Run: pip install requests")
        return False
    
    print(f"📥 Downloading HPO OBO file from {url}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        size_mb = len(response.content) / 1024 / 1024
        print(f"✅ Downloaded HPO OBO file to {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed to download HPO OBO: {e}")
        return False


# ── Parse HPO OBO File ──────────────────────────────────────────────────

def parse_hpo_obo(filepath: str) -> Dict[str, Dict]:
    """
    Parse HPO OBO file and extract ID -> {name, definition, synonyms}
    Returns dictionary with all HPO terms
    """
    print(f"📂 Parsing HPO OBO file: {filepath}")
    
    hpo_terms = {}
    current = {}
    term_count = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("[Term]"):
                    # Save previous term
                    if current.get("id"):
                        hpo_terms[current["id"]] = {
                            "name": current.get("name", ""),
                            "definition": current.get("def", ""),
                            "synonyms": current.get("synonyms", []),
                            "is_a": current.get("is_a", [])
                        }
                        term_count += 1
                    current = {"synonyms": [], "is_a": []}
                    
                elif line.startswith("id: "):
                    current["id"] = line.replace("id: ", "")
                    
                elif line.startswith("name: "):
                    current["name"] = line.replace("name: ", "")
                    
                elif line.startswith("def: "):
                    # Extract definition text between quotes
                    def_match = re.search(r'"([^"]+)"', line)
                    if def_match:
                        current["def"] = def_match.group(1)
                        
                elif line.startswith("synonym: "):
                    # Extract synonym text between quotes
                    syn_match = re.search(r'"([^"]+)"', line)
                    if syn_match:
                        current["synonyms"].append(syn_match.group(1))
                        
                elif line.startswith("is_a: "):
                    # Extract parent HPO ID
                    is_a_match = re.search(r"HP:\d+", line)
                    if is_a_match:
                        current["is_a"].append(is_a_match.group())
        
        # Save last term
        if current.get("id"):
            hpo_terms[current["id"]] = {
                "name": current.get("name", ""),
                "definition": current.get("def", ""),
                "synonyms": current.get("synonyms", []),
                "is_a": current.get("is_a", [])
            }
            term_count += 1
        
        print(f"✅ Parsed {len(hpo_terms)} HPO terms")
        return hpo_terms
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return {}
    except Exception as e:
        print(f"❌ Error parsing OBO file: {e}")
        return {}


# ── Build Searchable Text for Each HPO ─────────────────────────────────

def build_hpo_text_map(hpo_terms: Dict[str, Dict]) -> Dict[str, str]:
    """
    Build a mapping from HPO ID to searchable text
    Combines: name + definition + synonyms
    """
    hpo_text_map = {}
    
    for hpo_id, term in hpo_terms.items():
        parts = []
        
        # Add name
        if term.get("name"):
            parts.append(term["name"])
        
        # Add definition (first 200 chars max)
        if term.get("definition"):
            def_text = term["definition"][:200]
            parts.append(def_text)
        
        # Add synonyms (up to 5)
        if term.get("synonyms"):
            parts.extend(term["synonyms"][:5])
        
        # Combine
        hpo_text_map[hpo_id] = " ".join(parts)
    
    return hpo_text_map


# ── HPO Expander Class ──────────────────────────────────────────────────

class HPOExpander:
    """Automatic HPO expansion using official HPO OBO file"""
    
    def __init__(self, obo_path: str = "hp.obo", auto_download: bool = True):
        self.hpo_terms = {}
        self.hpo_text_map = {}
        self.is_loaded = False
        
        # Check if OBO exists
        if os.path.exists(obo_path):
            self.hpo_terms = parse_hpo_obo(obo_path)
            if self.hpo_terms:
                self.hpo_text_map = build_hpo_text_map(self.hpo_terms)
                self.is_loaded = True
                print(f"✅ Loaded {len(self.hpo_terms)} HPO terms")
        elif auto_download:
            print(f"⚠️ OBO file not found at {obo_path}")
            print("   Downloading now...")
            if download_hpo_obo(obo_path):
                self.hpo_terms = parse_hpo_obo(obo_path)
                if self.hpo_terms:
                    self.hpo_text_map = build_hpo_text_map(self.hpo_terms)
                    self.is_loaded = True
                    print(f"✅ Loaded {len(self.hpo_terms)} HPO terms")
    
    def expand_hpo_id(self, hpo_id: str, max_def_length: int = 100) -> str:
        """Expand a single HPO ID to text"""
        if not self.is_loaded:
            return hpo_id
        
        # Normalize HPO ID format
        hpo_id_clean = hpo_id.upper().strip()
        
        if hpo_id_clean in self.hpo_text_map:
            text = self.hpo_text_map[hpo_id_clean]
            # Truncate definition if too long
            if max_def_length > 0 and len(text) > max_def_length:
                # Try to keep name and synonyms, truncate definition
                term = self.hpo_terms.get(hpo_id_clean, {})
                name = term.get("name", "")
                synonyms = term.get("synonyms", [])
                combined = name + " " + " ".join(synonyms[:3])
                if len(combined) > max_def_length:
                    return combined[:max_def_length]
                return combined
            return text
        
        return hpo_id
    
    def expand_query(self, query: str, max_def_length: int = 100) -> str:
        """Expand all HPO IDs in a query"""
        # Extract HPO IDs
        hpo_ids = re.findall(r"HP:\d{7}", query, re.IGNORECASE)
        
        if not hpo_ids:
            return query
        
        expanded_query = query
        for hpo_id in hpo_ids:
            text = self.expand_hpo_id(hpo_id, max_def_length)
            if text and text != hpo_id:
                # Replace with ID + text
                expanded_query = expanded_query.replace(hpo_id, f"{hpo_id} {text}")
        
        return expanded_query
    
    def get_term_info(self, hpo_id: str) -> Optional[Dict]:
        """Get full term information for an HPO ID"""
        hpo_id_clean = hpo_id.upper().strip()
        if hpo_id_clean in self.hpo_terms:
            return self.hpo_terms[hpo_id_clean]
        return None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("🧬 HPO Expander - Automatic OBO-based Expansion")
    print("=" * 60)
    
    # Download and parse OBO
    expander = HPOExpander("hp.obo", auto_download=True)
    
    if not expander.is_loaded:
        print("❌ Failed to load HPO data")
        print("   Please install 'requests' and try again:")
        print("   pip install requests")
        return
    
    # Test queries
    test_queries = [
        "HP:0000508 HP:0000613 HP:0002015 HP:0002206",
        "HP:0003236 HP:0000944 HP:0003452 HP:0001302",
        "HP:0001417 HP:0001396 HP:0000717 HP:0002072",
        "HP:0001263 HP:0001250 HP:0002067 HP:0010739",
        "HP:0000956 HP:0001067 HP:0000608 HP:0009730",
        "HP:0000615 HP:0001053 HP:0009733 HP:0001250",
    ]
    
    print("\n📊 Expansion Results (with truncated definitions):")
    print("-" * 80)
    
    for query in test_queries:
        expanded = expander.expand_query(query, max_def_length=80)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded[:200]}...")
    
    # Show some term info
    print("\n" + "=" * 60)
    print("📖 Sample HPO Term Information:")
    print("-" * 60)
    
    sample_ids = ["HP:0001250", "HP:0001417", "HP:0003236"]
    for hpo_id in sample_ids:
        info = expander.get_term_info(hpo_id)
        if info:
            print(f"\n{hpo_id}:")
            print(f"  Name: {info.get('name', 'N/A')}")
            def_text = info.get('definition', 'N/A')
            print(f"  Definition: {def_text[:80]}..." if len(def_text) > 80 else f"  Definition: {def_text}")
            print(f"  Synonyms: {', '.join(info.get('synonyms', [])[:3])}")
    
    print("\n" + "=" * 60)
    print("✅ HPO expansion ready!")
    print(f"   Total terms loaded: {len(expander.hpo_terms)}")
    print(f"   Example: {test_queries[0]}")
    expanded_example = expander.expand_query(test_queries[0], max_def_length=80)
    print(f"   -> {expanded_example[:200]}...")


if __name__ == "__main__":
    main()