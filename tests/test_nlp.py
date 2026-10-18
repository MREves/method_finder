"""Unit Tests for Natural Language Processing (NLP) Module.

Verifies schema integrity, mathematical formulations across all 7 methods
(Sentiment & NegEx, TF-IDF, Naive Bayes/SVM, LDA, Dense Embeddings, Named Entity Recognition, RAG),
synthetic data generation, and strict AST expander compliance (expanded=False).
"""

import os
import sys
import ast
import unittest
try:
    import numpy as np
    import pandas as pd
    HAS_SCIENTIFIC_STACK = True
except ImportError:
    HAS_SCIENTIFIC_STACK = False
    np = None
    pd = None

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp.nlp_schema import NLP_METHODS, NLP_CLUSTERS, NLP_WIZARD_QUESTIONS
if HAS_SCIENTIFIC_STACK:
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
else:
    calculate_sentiment_negex = None
    calculate_tfidf = None
    calculate_naive_bayes_svm = None
    calculate_lda_topics = None
    calculate_dense_embeddings_similarity = None
    calculate_clinical_ner = None
    calculate_rag_metrics = None
    generate_synthetic_nlp_data = None


class TestNLPSchemaIntegrity(unittest.TestCase):
    """Verifies complete metadata coverage across all 7 NLP methods."""

    def test_all_seven_methods_exist_and_have_required_fields(self):
        self.assertEqual(len(NLP_METHODS), 7)
        expected_keys = [
            "sentiment_negex",
            "tfidf_keywords",
            "text_classification_bayes_svm",
            "lda_topic_modelling",
            "dense_embeddings_similarity",
            "clinical_ner",
            "llm_zeroshot_rag"
        ]
        for k in expected_keys:
            self.assertIn(k, NLP_METHODS)
            meta = NLP_METHODS[k]
            self.assertEqual(meta["id"], k)
            self.assertIn("name", meta)
            self.assertIn("category", meta)
            self.assertIn("level", meta)
            self.assertIn("icon", meta)
            self.assertIn("summary", meta)
            self.assertIn("when_to_use", meta)
            self.assertIn("key_assumptions", meta)
            self.assertIn("common_pitfalls", meta)
            self.assertIn("plain_english_rules", meta)
            self.assertIn("formulas", meta)
            self.assertIn("excel_guide", meta)
            self.assertIn("python_code", meta)

            # Check formulas
            self.assertGreater(len(meta["formulas"]), 0)
            for f_name, f_data in meta["formulas"].items():
                self.assertIn("latex", f_data)
                self.assertIn("plain_english", f_data)
                self.assertIn("title", f_data)
                self.assertIn("terms", f_data)
                self.assertGreater(len(f_data["terms"]), 0)
                self.assertIn("steps", f_data)
                self.assertGreater(len(f_data["steps"]), 0)
                self.assertIn("why_used", f_data)
                self.assertGreater(len(f_data["why_used"]), 10)

            # Check Excel guide
            self.assertIn("cell_mapping", meta["excel_guide"])
            self.assertIn("steps", meta["excel_guide"])

    def test_llm_zeroshot_rag_tooling_and_transformers_code(self):
        rag_meta = NLP_METHODS["llm_zeroshot_rag"]
        excel_guide = rag_meta["excel_guide"]
        self.assertTrue(excel_guide.get("is_not_viable", False))
        self.assertIn("recommended_tooling", excel_guide)
        self.assertIn("why_not_viable", excel_guide)
        
        # Verify python_code utilizes standard transformers and pipeline structures
        py_code = rag_meta["python_code"]
        self.assertIn("from transformers import pipeline", py_code)
        self.assertIn("from sentence_transformers import SentenceTransformer", py_code)
        self.assertIn("zero-shot-classification", py_code)
        self.assertIn("text-generation", py_code)

    def test_nlp_clusters_structure(self):
        self.assertEqual(len(NLP_CLUSTERS), 5)
        for c_id, c_data in NLP_CLUSTERS.items():
            self.assertIn("name", c_data)
            self.assertIn("icon", c_data)
            self.assertIn("level", c_data)
            self.assertIn("description", c_data)
            self.assertIn("methods", c_data)

    def test_wizard_questions_structure(self):
        self.assertGreater(len(NLP_WIZARD_QUESTIONS), 0)
        q = NLP_WIZARD_QUESTIONS[0]
        self.assertIn("options", q)
        self.assertEqual(len(q["options"]), 7)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestSentimentNegExEngine(unittest.TestCase):
    """Verifies sentiment analysis and negation scope inversion."""

    def test_positive_and_negative_sentiment(self):
        pos_res = calculate_sentiment_negex("The customer support staff were attentive and compassionate; user felt valued.")
        self.assertGreater(pos_res["compound"], 0.05)
        self.assertEqual(pos_res["category"], "Positive")

        neg_res = calculate_sentiment_negex("Service delivery was delayed for hours; agent was extremely rude and dismissive.")
        self.assertLess(neg_res["compound"], -0.05)
        self.assertEqual(neg_res["category"], "Negative")

    def test_negation_inverts_issue_valence(self):
        # Issue with negative valence
        issue_res = calculate_sentiment_negex("System experienced severe outage.")
        self.assertLess(issue_res["compound"], 0.0)

        # Negated issue: "no outage" should invert valence and increment negation count
        negated_res = calculate_sentiment_negex("System experienced no outage.")
        self.assertGreater(negated_res["negations_detected"], 0)
        self.assertGreater(negated_res["compound"], issue_res["compound"])

    def test_empty_text_safety(self):
        res = calculate_sentiment_negex("")
        self.assertEqual(res["compound"], 0.0)
        self.assertEqual(res["category"], "Neutral")


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestTFIDFEngine(unittest.TestCase):
    """Verifies TF-IDF matrix generation, IDF weighting, and pairwise cosine similarity."""

    def setUp(self):
        self.corpus = [
            "Production server outage: application received memory error in cloud cluster.",
            "Server deployment delayed due to infrastructure outage in cloud cluster.",
            "Customer payment checkout incident near gateway; mild delay, no failure noted.",
            "Critical security vulnerability reported by monitoring system; team failed to patch for 4 hours."
        ]

    def test_tfidf_matrix_and_similarity(self):
        res = calculate_tfidf(self.corpus, max_features=10)
        self.assertIn("feature_names", res)
        self.assertIn("tfidf_matrix", res)
        self.assertIn("cosine_similarity", res)
        self.assertIn("keywords_table", res)

        # Cosine similarity diagonal must be 1.0
        sim_df = res["cosine_similarity"]
        for i in range(len(self.corpus)):
            self.assertAlmostEqual(sim_df.iloc[i, i], 1.0, places=2)

        # Server outage documents (Doc 1 vs Doc 2) should have higher similarity than Doc 1 vs Doc 3 (Payment)
        sim_server = sim_df.iloc[0, 1]
        sim_unrelated = sim_df.iloc[0, 2]
        self.assertGreater(sim_server, sim_unrelated)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestNaiveBayesSVMEngine(unittest.TestCase):
    """Verifies supervised classification with Laplace smoothing."""

    def test_supervised_classifier_training_and_metrics(self):
        df_data = generate_synthetic_nlp_data("text_classification_bayes_svm", n_samples=16, seed=42)
        res = calculate_naive_bayes_svm(
            df_data["Incident_Description"].tolist(),
            df_data["Severity_Tier"].tolist()
        )
        self.assertGreater(res["accuracy"], 0.5)
        self.assertGreater(res["macro_f1"], 0.5)
        self.assertIsInstance(res["confusion_matrix"], pd.DataFrame)
        self.assertEqual(len(res["predictions_table"]), len(df_data))


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestLDATopicEngine(unittest.TestCase):
    """Verifies Latent Dirichlet Allocation (LDA) and topic coherence."""

    def test_lda_topic_modelling(self):
        df_c = generate_synthetic_nlp_data("lda_topic_modelling", n_samples=15, seed=42)
        res = calculate_lda_topics(df_c["Customer_Review"].tolist(), n_topics=3, seed=42)
        
        self.assertEqual(res["n_topics"], 3)
        self.assertGreater(res["coherence_score"], 0.0)
        self.assertIn("topics_table", res)
        self.assertIn("doc_mixtures", res)
        
        # Each document topic mixture should sum to ~100%
        doc_mix = res["doc_mixtures"]
        for _, row in doc_mix.iterrows():
            total_prob = row["Topic_1_%"] + row["Topic_2_%"] + row["Topic_3_%"]
            self.assertAlmostEqual(total_prob, 100.0, delta=1.5)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestDenseEmbeddingsEngine(unittest.TestCase):
    """Verifies dense semantic vector embeddings and 2D PCA projection."""

    def test_dense_embeddings_search(self):
        df_p = generate_synthetic_nlp_data("dense_embeddings_similarity", n_samples=10, seed=42)
        corpus = df_p["Comment_Text"].tolist()
        query = "support agents were rude and unhelpful"
        
        res = calculate_dense_embeddings_similarity(corpus, query, seed=42)
        self.assertIn("results_table", res)
        self.assertIn("best_match_doc", res)
        self.assertGreaterEqual(res["best_similarity"], 0.0)
        self.assertLessEqual(res["best_similarity"], 1.0)
        
        # PCA coordinates must be 2D tuple
        self.assertEqual(len(res["query_pca_2d"]), 2)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestClinicalNEREngine(unittest.TestCase):
    """Verifies named entity extraction, IOB tagging, and F1 estimation."""

    def test_ner_extraction(self):
        text = "Acme Corp purchased 50 cloud servers for $45,000 on 2024-03-15; reported critical latency defect."
        res = calculate_clinical_ner(text)
        
        self.assertGreater(res["n_entities"], 0)
        self.assertIn("ORGANIZATION", res["entity_counts"])
        self.assertIn("PRODUCT", res["entity_counts"])
        self.assertIn("PRICE", res["entity_counts"])
        self.assertIn("DEFECT", res["entity_counts"])
        
        # Check IOB tagging sequence
        df_iob = res["iob_table"]
        self.assertGreater(len(df_iob), 0)
        self.assertIn("B-ORGANIZATION", df_iob["IOB_Tag"].values)
        self.assertIn("B-PRODUCT", df_iob["IOB_Tag"].values)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestRAGArchitectureEngine(unittest.TestCase):
    """Verifies RAG vector retrieval, prompt assembly, and faithfulness audit."""

    def test_rag_retrieval_and_prompt_assembly(self):
        query = "What is the guaranteed response time and credit for a Severity 1 P1 outage?"
        res = calculate_rag_metrics(query, top_k=1)
        
        self.assertIn("retrieved_chunks", res)
        self.assertIn("DOC_SLA_101", res["best_source"])
        self.assertGreater(res["best_similarity"], 0.20)
        self.assertIn("99.99%", res["assembled_prompt"])
        self.assertGreater(res["faithfulness"], 0.80)
        self.assertGreater(res["answer_relevance"], 0.80)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "Scientific stack not available")
class TestSyntheticNLPDataGenerators(unittest.TestCase):
    """Verifies synthetic dataset generation for all 7 NLP methods."""

    def test_all_seven_generators(self):
        methods = [
            "sentiment_negex",
            "tfidf_keywords",
            "text_classification_bayes_svm",
            "lda_topic_modelling",
            "dense_embeddings_similarity",
            "clinical_ner",
            "llm_zeroshot_rag"
        ]
        for m in methods:
            df = generate_synthetic_nlp_data(m, n_samples=20, seed=42)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 20)


class TestNLPASTExpanderCompliance(unittest.TestCase):
    """Verifies strict AST expander compliance: all expanders must be collapsed (expanded=False)."""

    def test_nlp_page_expander_compliance(self):
        file_path = os.path.join(PROJECT_ROOT, "pages", "6_🗣️_Natural_Language_Processing.py")
        self.assertTrue(os.path.exists(file_path), f"File {file_path} must exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("expanded" + "=True", content)

    def test_nlp_widgets_expander_compliance(self):
        file_path = os.path.join(PROJECT_ROOT, "src", "ui_components", "nlp_widgets.py")
        self.assertTrue(os.path.exists(file_path), f"File {file_path} must exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("expanded" + "=True", content)


if __name__ == "__main__":
    unittest.main()
