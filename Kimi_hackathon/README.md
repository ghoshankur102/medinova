# 🧬 MediNova - Rare Disease Diagnosis Support System

**AI-powered rare disease diagnosis support with hybrid RAG + medical image analysis.**

[![Streamlit App](https://img.shields.io/badge/🚀-Live_Demo-blue?style=for-the-badge&logo=streamlit)](https://your-app-url.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Evaluation Results](#-evaluation-results)
- [Project Structure](#-project-structure)
- [Target Audience](#-target-audience)
- [Future Work](#-future-work)
- [Team](#-team)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**MediNova** is an intelligent rare disease diagnosis support system that combines:

- **Hybrid Retrieval**: Semantic (dense) + keyword (sparse) search over 10,000+ rare diseases
- **Natural Language Understanding**: Converts colloquial symptoms to clinical terms
- **Medical Image Analysis**: AI-powered analysis of X-rays, MRIs, and CT scans
- **HPO Integration**: Structured search using Human Phenotype Ontology codes

> ⚠️ **Disclaimer**: This system is for **educational and research support only**. It does NOT replace a qualified physician or radiologist. Never use this for clinical decisions.

---

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔍 **Symptom Search** | Natural language querying with 73.6% accuracy | ✅ Production |
| 🧬 **HPO Code Support** | Structured search with 91.7% accuracy | ✅ Production |
| 🩻 **Image Analysis** | Gemini 2.5 Flash for medical image interpretation | ✅ Experimental |
| 📊 **Hybrid Retrieval** | FAISS (dense) + BM25 (sparse) fusion | ✅ Production |
| 💬 **Colloquial Support** | Converts patient-friendly language to clinical terms | ✅ Production |
| 🎯 **Gender Filtering** | Gender-aware disease filtering | ✅ Production |
| 🧠 **Inheritance Detection** | Recognizes genetic inheritance patterns | ✅ Production |
| ⚡ **Fast Performance** | 4.5 queries/sec with < 300ms latency | ✅ Production |

---

## 🛠️ Technology Stack

### Core Framework

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive web UI |
| **Language** | Python 3.10+ | Main application language |

### Retrieval Engine

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Dense Retrieval** | [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| **Sparse Retrieval** | [BM25](https://github.com/dorianbrown/rank_bm25) | Keyword-based search |
| **Embedding Model** | [MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | Biomedical text embeddings |

### AI/ML Models

| Model | Provider | Purpose |
|-------|----------|---------|
| **Gemini 2.5 Flash** | Google | Medical image analysis |
| **MedCPT** | NCBI | Text embeddings for disease retrieval |

### Data Sources

| Source | Description | Size |
|--------|-------------|------|
| **Orphanet** | Rare disease database | 10,000+ diseases |
| **HPO OBO** | Human Phenotype Ontology | 20,000+ terms |
| **NIH Chest X-ray** | Real medical images for testing | 112,120 images |

## 🔑 Environment Setup

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env

### Key Libraries

```python
# Core
streamlit          # Web UI
pandas             # Data manipulation
numpy              # Numerical computing

# Retrieval
faiss-cpu          # Vector search
rank-bm25          # BM25 search
transformers       # MedCPT model loading
torch              # Deep learning backend

# Vision
google-genai       # Gemini API client
Pillow             # Image processing

# Utilities
python-dotenv      # Environment variables
requests           # HTTP requests
tqdm               # Progress bars