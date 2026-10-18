"""Streamlit Page: Natural Language Processing (Classical to LLMs).

Guides analysts, engineers, and researchers through sentiment analysis, negation scope resolution,
TF-IDF keyword extraction, supervised incident classification (Naive Bayes / Linear SVM),
unsupervised topic discovery (LDA), dense semantic embeddings, Named Entity Recognition (NER),
and LLM zero-shot classification with Retrieval-Augmented Generation (RAG).
"""

import os
import sys
import streamlit as st

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.ui_components.sidebar_navigation import (
    render_custom_sidebar_navigation,
    render_roundtrip_navigation_banner
)
from src.ui_components.nlp_widgets import (
    render_nlp_mode_selector,
    render_nlp_method_module
)
from src.nlp.nlp_schema import NLP_METHODS

st.set_page_config(
    page_title="Natural Language Processing | MethodFinder",
    page_icon="🗣️",
    layout="wide"
)

# Render Custom Grouped Sidebar Navigation
render_custom_sidebar_navigation("pages/6_🗣️_Natural_Language_Processing.py")

# Top Header Banner
st.title("🗣️ Natural Language Processing (Classical to LLMs)")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All textual narratives, customer feedback entries, IT support tickets, entity annotations, and policy documents across this module are completely fictional and fabricated solely for education and illustration purposes.")

st.markdown("""
Navigate the methods below to analyze unstructured text across **5 progressive evolutionary levels**:
from **Classical & Rule-Based NLP** (lexicon sentiment, negation scope resolution) and **Statistical Vectorization** (TF-IDF, Naive Bayes / SVM),
to **Thematic Discovery** (LDA topic modelling), **Dense Semantic Embeddings** (sentence vectors, semantic search),
and **State-of-the-Art Generative AI** (Named Entity Recognition, LLM Zero-Shot classification, and Retrieval-Augmented Generation).
""")

# ----------------- Implementation Progress Tracker (Gated behind debug_mode) -----------------
if st.session_state.get("debug_mode", False) or st.session_state.get("debug_model", False):
    with st.expander("📌 NLP Module Implementation Tracker", expanded=False):
        st.markdown("""
        | Level | Method / Architecture | Mathematical Formulation | Excel Implementation | Python Pipeline | Synthetic Generator | Status |
        | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
        | **1. Rule-Based** | Sentiment Analysis & Negation Scope | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **2. Vectorization** | TF-IDF & Keyword Extraction | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **2. Vectorization** | Supervised Text Classification (NB / SVM) | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **3. Topics** | Latent Dirichlet Allocation (LDA) | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **4. Dense Vectors** | Semantic Embeddings & Vector Search | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **5. GenAI & LLMs** | Named Entity Recognition (NER) & Extraction | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        | **5. GenAI & LLMs** | LLM Zero-Shot & RAG Architecture | ✅ | ✅ | ✅ | ✅ | 🟢 Completed |
        """)

st.markdown("---")

# Navigation Mode & Method Execution
nav_mode, selected_method_key = render_nlp_mode_selector()
render_nlp_method_module(selected_method_key)

