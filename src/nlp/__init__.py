"""Natural Language Processing (NLP) Module for MethodFinder.

Provides end-to-end analytical architectures spanning Classical & Rule-Based NLP,
Statistical Vectorization (TF-IDF & Naive Bayes/SVM), Thematic Topic Discovery (LDA),
Dense Semantic Vector Embeddings, Clinical Named Entity Recognition (NER),
and Modern LLM Zero-Shot Classification with Retrieval-Augmented Generation (RAG).
"""

from src.nlp.nlp_schema import NLP_METHODS, NLP_CLUSTERS
from src.nlp.nlp_engine import (
    calculate_sentiment_negex,
    calculate_tfidf,
    calculate_naive_bayes_svm,
    calculate_lda_topics,
    calculate_dense_embeddings_similarity,
    calculate_clinical_ner,
    calculate_rag_metrics,
    generate_synthetic_nlp_data
)

__all__ = [
    "NLP_METHODS",
    "NLP_CLUSTERS",
    "calculate_sentiment_negex",
    "calculate_tfidf",
    "calculate_naive_bayes_svm",
    "calculate_lda_topics",
    "calculate_dense_embeddings_similarity",
    "calculate_clinical_ner",
    "calculate_rag_metrics",
    "generate_synthetic_nlp_data"
]

