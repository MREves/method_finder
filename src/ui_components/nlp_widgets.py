"""Interactive UI Components & Visual Diagnostic Playgrounds for Natural Language Processing (NLP).

Provides:
1. Mode Selector & Guided Problem-to-Method Wizard.
2. 7 Interactive Diagnostic Simulation Playgrounds with top controls, 2x2 wrapped metric grids,
   and full page-width Plotly visualizers.
3. Standard 4-Tab Methodological Blueprint containers (Theory, Excel Guide, Python Pipeline, CSV Download).
"""

from typing import Dict, Any, Tuple, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.nlp.nlp_schema import NLP_METHODS, NLP_CLUSTERS, NLP_WIZARD_QUESTIONS
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


# =====================================================================
# 1. Mode Selector & Navigation Router
# =====================================================================

def render_nlp_mode_selector() -> Tuple[str, str]:
    """Renders the top mode selector: Guided Problem-to-Method Wizard vs. Direct Method Explorer."""
    c_mode1, c_mode2 = st.columns([1.5, 3])
    with c_mode1:
        mode = st.radio(
            "Navigation Mode:",
            ["🎯 Guided Problem Wizard", "🔍 Direct Method Explorer"],
            index=0,
            horizontal=False,
            key="nlp_nav_mode_radio"
        )
        
    with c_mode2:
        if mode == "🎯 Guided Problem Wizard":
            q_info = NLP_WIZARD_QUESTIONS[0]
            options_dict = {opt["label"]: opt["recommendation"] for opt in q_info["options"]}
            selected_label = st.selectbox(
                f"Question: {q_info['question']}",
                options=list(options_dict.keys()),
                index=0,
                key="nlp_wizard_select"
            )
            selected_method_key = options_dict[selected_label]
            meta = NLP_METHODS[selected_method_key]
            st.success(f"**Recommended Method:** {meta['icon']} **{meta['name']}** — *{meta['category']}*")
        else:
            method_options = {f"{m['icon']} {m['name']} ({m['category'].split(':')[0]})": k for k, m in NLP_METHODS.items()}
            selected_display = st.selectbox(
                "Select NLP Method Directly:",
                options=list(method_options.keys()),
                index=0,
                key="nlp_direct_select"
            )
            selected_method_key = method_options[selected_display]
            
    st.markdown("---")
    return mode, selected_method_key


# =====================================================================
# 2. Main Method Execution Controller
# =====================================================================

def render_nlp_method_module(method_key: str) -> None:
    """Renders the complete pedagogical module for the selected NLP method."""
    if method_key not in NLP_METHODS:
        st.error(f"Unknown NLP method key: '{method_key}'")
        return

    meta = NLP_METHODS[method_key]
    
    # Method Title & Header Banner
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <span style="font-size: 1.35rem; font-weight: 800; color: #0f172a;">
                {meta['icon']} {meta['name']}
            </span>
            <span style="background-color: #eff6ff; color: #1e40af; font-size: 0.78rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid #bfdbfe;">
                {meta['category']}
            </span>
        </div>
        <div style="font-size: 0.92rem; color: #334155; margin-top: 0.4rem; line-height: 1.5;">
            {meta['summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Live Diagnostic Simulation & Visual Playground
    st.info("💡 **Educational Disclaimer:** All textual examples, customer feedback narratives, IT incident tickets, entity annotations, and policy documents across this module are entirely fictional and fabricated solely for educational and methodological illustration purposes.")
    st.markdown("### 🔬 Live Interactive Simulation & Diagnostic Visualizer")
    
    if method_key == "sentiment_negex":
        render_sentiment_negex_playground(meta)
    elif method_key == "tfidf_keywords":
        render_tfidf_keywords_playground(meta)
    elif method_key == "text_classification_bayes_svm":
        render_naive_bayes_svm_playground(meta)
    elif method_key == "lda_topic_modelling":
        render_lda_topics_playground(meta)
    elif method_key == "dense_embeddings_similarity":
        render_dense_embeddings_playground(meta)
    elif method_key == "clinical_ner":
        render_clinical_ner_playground(meta)
    elif method_key == "llm_zeroshot_rag":
        render_llm_zeroshot_rag_playground(meta)
        
    st.markdown("---")
    
    # Standard 4-Tab Pedagogical Container
    st.subheader(f"📚 Methodological Blueprint: {meta['name']}")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 1. Theory & Formulas",
        "📗 2. Excel Guide",
        "🐍 3. Python Pipeline",
        "💾 4. Download Synthetic Data"
    ])
    
    with tab1:
        render_nlp_tab1_theory(meta)
    with tab2:
        render_nlp_tab2_excel(meta)
    with tab3:
        render_nlp_tab3_python(meta)
    with tab4:
        render_nlp_tab4_data(meta)


# =====================================================================
# 3. Interactive Playgrounds per Method
# =====================================================================

def render_sentiment_negex_playground(meta: Dict[str, Any]) -> None:
    """Interactive Lexicon Sentiment with Negation Scope Inversion."""
    sample_texts = [
        "Customer service was prompt and helpful; no complaints about the onboarding process.",
        "Software update installed seamlessly; device running smoothly without any performance issues.",
        "Delivery was delayed by two weeks; support agent was extremely rude and dismissive.",
        "Monthly invoice was inaccurate and customer support repeatedly ignored ticket escalation.",
        "Routine account renewal completed; client without any complaints or billing discrepancies."
    ]
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_sample = st.selectbox(
            "Select Customer / User Feedback Sample:",
            options=sample_texts,
            index=0,
            key="sent_sample_select"
        )
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="sent_seed")
        
    custom_text = st.text_input("Or Enter Custom User Review / Feedback Text:", value=selected_sample, key="sent_custom_input")
    res = calculate_sentiment_negex(custom_text)
    
    # 2x2 Metric Layout with responsive text wrapping
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetricDelta"] {
        white-space: normal !important;
        word-break: break-word !important;
        font-size: 0.82rem !important;
        line-height: 1.3 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Compound Valence Score", f"{res['compound']:+.3f}")
    with m2:
        st.metric("Polarity Category", res["rating_tier"], delta=res["rating_detail"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Negation Scopes Detected", f"{res['negations_detected']} scope(s)")
    with m4:
        st.metric("Word Count Processed", f"{res['word_count']} words", delta=f"{int(res['pos']*100)}% Pos / {int(res['neg']*100)}% Neg")
        
    # Polarity distribution bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Negative", "Neutral", "Positive"],
        y=[res["neg"] * 100, res["neu"] * 100, res["pos"] * 100],
        marker_color=["#ef4444", "#94a3b8", "#16a34a"],
        text=[f"{res['neg']*100:.1f}%", f"{res['neu']*100:.1f}%", f"{res['pos']*100:.1f}%"],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"Sentence Polarity Distribution (% Share) — Normalized Compound: {res['compound']:+.3f}",
        yaxis=dict(title="% Proportion of Valence"),
        height=320,
        margin=dict(l=50, r=30, t=35, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🔬 Token-by-Token Valence & Negation Breakdown")
    st.dataframe(res["token_table"], use_container_width=True)


def render_tfidf_keywords_playground(meta: Dict[str, Any]) -> None:
    """Interactive TF-IDF keyword extraction and document similarity heatmap."""
    corpus = [
        "Server outage: Production database connection pool exhausted during payment gateway surge.",
        "Database connection timeout during checkout payment processing on production cluster.",
        "Warehouse shipment delayed due to inventory barcode scanner hardware failure.",
        "Payment gateway API rejected batch credit card transactions due to expired SSL certificate.",
        "Customer refund request unresolved after order cancellation in retail portal."
    ]
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        max_feat = st.slider("Top Vocabulary Features to Extract:", 5, 20, 10, 1, key="tfidf_feat_slider")
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="tfidf_seed")
    st.caption("Demonstrating smooth Inverse Document Frequency: terms frequent in one incident ticket but rare globally receive highest discriminative weights.")
        
    res = calculate_tfidf(corpus, max_features=max_feat)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Corpus Documents", f"{res['n_documents']} tickets")
    with m2:
        st.metric("Extracted Vocabulary", f"{res['vocab_size']} terms", delta="Domain stop words removed", delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Avg Pairwise Similarity", f"{res['avg_inter_doc_similarity']:.3f}")
    with m4:
        top_global_term = res["feature_names"][0] if res["feature_names"] else "None"
        st.metric("Highest Rank Corpus Term", top_global_term, delta="Maximum IDF Discriminator", delta_color="off")
        
    # Full Page-Width Cosine Similarity Heatmap
    fig_heat = px.imshow(
        res["cosine_similarity"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Pairwise Document Cosine Similarity Matrix (1.0 = Identical Vocabulary Distribution)"
    )
    fig_heat.update_layout(height=330, margin=dict(l=60, r=30, t=35, b=25))
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("#### 🔤 Top TF-IDF Keywords per Incident Narrative")
    st.dataframe(res["keywords_table"], use_container_width=True)


def render_naive_bayes_svm_playground(meta: Dict[str, Any]) -> None:
    """Interactive Supervised Incident Classification with Laplace Smoothing."""
    df_data = generate_synthetic_nlp_data("text_classification_bayes_svm", n_samples=16, seed=42)
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_incident = st.selectbox(
            "Select IT Support Ticket to Triage:",
            options=df_data["Incident_Description"].tolist(),
            index=0,
            key="nb_incident_select"
        )
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="nb_seed")
    st.caption("Multinomial Naive Bayes model evaluating word log-likelihood ratios to triage incident tickets.")
        
    res = calculate_naive_bayes_svm(
        df_data["Incident_Description"].tolist(),
        df_data["Severity_Tier"].tolist()
    )
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Model Training Accuracy", f"{int(res['accuracy'] * 100)}%")
    with m2:
        st.metric("Macro F1-Score", f"{res['macro_f1']:.3f}", delta=res["rating_detail"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Total Incident Corpus", f"{res['n_samples']} tickets")
    with m4:
        st.metric("Incident Classes", f"{len(res['classes'])} tiers", delta=" / ".join(res["classes"][:2]), delta_color="off")
        
    # Full Page-Width Confusion Matrix Heatmap
    fig_conf = px.imshow(
        res["confusion_matrix"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Greens",
        title="Supervised Ticket Triage Confusion Matrix (Actual vs Predicted Severity)"
    )
    fig_conf.update_layout(height=320, margin=dict(l=60, r=30, t=35, b=25))
    st.plotly_chart(fig_conf, use_container_width=True)
    
    st.markdown("#### 🏷️ Ticket Classification & Confidence Breakdown")
    st.dataframe(res["predictions_table"], use_container_width=True)


def render_lda_topics_playground(meta: Dict[str, Any]) -> None:
    """Interactive Latent Dirichlet Allocation (LDA) Topic Discovery."""
    df_reviews = generate_synthetic_nlp_data("lda_topic_modelling", n_samples=18, seed=42)
    corpus = df_reviews["Customer_Review"].tolist()
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        n_top = st.slider("Number of Latent Topics (k):", 2, 5, 3, 1, key="lda_k_slider")
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="lda_seed")
    st.caption("Unsupervised thematic discovery: models customer reviews as Dirichlet mixtures over latent topics.")
        
    res = calculate_lda_topics(corpus, n_topics=n_top, seed=int(seed_val))
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Latent Topics Discovered", f"k = {res['n_topics']}")
    with m2:
        st.metric("Topic Coherence (Cv)", f"{res['coherence_score']:.3f}", delta=res["coherence_label"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Corpus Size Evaluated", f"{len(corpus)} reviews")
    with m4:
        st.metric("Vocabulary Extracted", f"{res['vocab_size']} words", delta="Min document frequency = 1", delta_color="off")
        
    # Bar Chart of Topic Prevalences
    df_t = res["topics_table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_t["Topic"],
        y=df_t["Prevalence_Pct"],
        text=df_t["Top_Keywords"],
        textposition="inside",
        marker_color=["#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899"][:len(df_t)]
    ))
    fig.update_layout(
        title="Unsupervised Topic Prevalences across Customer Reviews (% Corpus Share)",
        yaxis=dict(title="% of Total Review Content"),
        height=320,
        margin=dict(l=50, r=30, t=35, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🧩 Review-by-Review Topic Mixture Proportions")
    st.dataframe(res["doc_mixtures"], use_container_width=True)


def render_dense_embeddings_playground(meta: Dict[str, Any]) -> None:
    """Interactive Dense Semantic Vector Embeddings & 2D PCA projection."""
    df_phrases = generate_synthetic_nlp_data("dense_embeddings_similarity", n_samples=12, seed=42)
    corpus = df_phrases["Comment_Text"].tolist()
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        query = st.text_input("Enter Semantic Search Inquiry:", value="application is sluggish and freezes frequently", key="emb_query_input")
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="emb_seed")
    st.caption("Dense embeddings map concepts into a continuous 384-dimensional geometric space, finding semantically synonymous feedback entries regardless of exact phrasing.")
        
    res = calculate_dense_embeddings_similarity(corpus, query, seed=int(seed_val))
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Top Semantic Match", f"{res['best_match_doc']}")
    with m2:
        st.metric("Maximum Cosine Similarity", f"{res['best_similarity']:.3f}", delta=res["rating_detail"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Vector Dimensions", "384 dense dims")
    with m4:
        st.metric("Search Space", f"{len(corpus)} feedback entries", delta="Continuous geometric manifold", delta_color="off")
        
    # 2D PCA Scatter Plot
    df_res = res["results_table"]
    fig = go.Figure()
    
    # Corpus documents
    fig.add_trace(go.Scatter(
        x=df_res["PCA_X"],
        y=df_res["PCA_Y"],
        mode="markers+text",
        text=df_res["Document"],
        textposition="top center",
        marker=dict(
            size=12,
            color=df_res["Cosine_Similarity"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Cosine Sim")
        ),
        name="Stored Feedback / Tickets"
    ))
    
    # Query point
    fig.add_trace(go.Scatter(
        x=[res["query_pca_2d"][0]],
        y=[res["query_pca_2d"][1]],
        mode="markers+text",
        text=["Your Query 🎯"],
        textposition="bottom center",
        marker=dict(size=16, color="#ef4444", symbol="star"),
        name="Query Vector"
    ))
    
    fig.update_layout(
        title="2D PCA Projection of Semantic Vector Manifold (Proximity Reflects Meaning)",
        xaxis=dict(title="Principal Component 1 (Semantic Latent Space)"),
        yaxis=dict(title="Principal Component 2"),
        height=340,
        margin=dict(l=50, r=30, t=35, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🔍 Semantic Similarity Ranking Table")
    st.dataframe(df_res[["Rank", "Document", "Text", "Cosine_Similarity", "Match_Tier"]], use_container_width=True)


def render_clinical_ner_playground(meta: Dict[str, Any]) -> None:
    """Interactive Named Entity Recognition (NER) & IOB tagger."""
    sample_notes = [
        "Acme Corp acquired 50 cloud servers for $45,000 on 2024-03-15; reported critical latency defect in production cluster.",
        "Customer ordered 12 units of Model X laptop with 64GB RAM for $2,400; hardware overheating defect observed.",
        "GlobalTech deployed 100 enterprise database licenses on 2023-11-20; experienced network timeout defect under peak load.",
        "BetaLabs renewed 5 server appliances with 128GB RAM costing $15,000; reported memory leak defect in storage tier."
    ]
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        selected_note = st.selectbox("Select Enterprise Ticket / Invoice Sample:", options=sample_notes, index=0, key="ner_sample_select")
    with col_ctrl2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="ner_seed")
        
    custom_note = st.text_input("Or Enter Custom Enterprise Narrative:", value=selected_note, key="ner_custom_input")
    res = calculate_clinical_ner(custom_note)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Named Entities Tagged", f"{res['n_entities']} entities")
    with m2:
        st.metric("Entity Density", f"{res['entity_density']}% of tokens", delta=res["rating_detail"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Strict Entity F1 Benchmark", f"{res['strict_f1']:.3f}")
    with m4:
        st.metric("Ontological Types Found", f"{len(res['entity_counts'])} categories", delta="Org / Product / Spec / Defect / Price", delta_color="off")
        
    # Interactive Highlighted Text Markup
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 6px; padding: 0.9rem 1.1rem; line-height: 1.8; margin-top: 0.4rem; margin-bottom: 0.8rem; font-size: 0.95rem;">
        {res['highlighted_html']}
    </div>
    """, unsafe_allow_html=True)
    
    # Entity Counts Bar Chart
    df_counts = pd.DataFrame(list(res["entity_counts"].items()), columns=["Entity_Type", "Count"]) if res["entity_counts"] else pd.DataFrame(columns=["Entity_Type", "Count"])
    if not df_counts.empty:
        fig = px.bar(
            df_counts,
            x="Entity_Type",
            y="Count",
            color="Entity_Type",
            title="Distribution of Extracted Named Entity Classes",
            text="Count"
        )
        fig.update_layout(height=280, margin=dict(l=40, r=30, t=35, b=25))
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### 🏷️ IOB Token Span Diagnostic Table")
    st.dataframe(res["iob_table"], use_container_width=True)


def render_llm_zeroshot_rag_playground(meta: Dict[str, Any]) -> None:
    """Interactive LLM Zero-Shot & Retrieval-Augmented Generation (RAG) Architecture."""
    sample_queries = [
        "What is the guaranteed response time and credit for a Severity 1 P1 outage?",
        "What is the return policy window and restocking fee for opened hardware?",
        "What are the API rate limits and retry headers for enterprise tier endpoints?",
        "What are the data encryption standards and retention protocols for customer records?"
    ]
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        query = st.selectbox("Select Enterprise Policy Inquiry for RAG Decision Support:", options=sample_queries, index=0, key="rag_q_select")
    with col_ctrl2:
        top_k = st.slider("Top Chunks to Retrieve (k):", 1, 3, 1, 1, key="rag_k_slider")
        
    res = calculate_rag_metrics(query, top_k=top_k)
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("RAG Faithfulness Score", f"{int(res['faithfulness'] * 100)}%")
    with m2:
        st.metric("Hallucination Risk Audit", res["rating_tier"], delta=res["rating_detail"], delta_color="off")
        
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Answer Relevance Score", f"{int(res['answer_relevance'] * 100)}%")
    with m4:
        st.metric("Context Precision", f"{res['context_precision']:.3f}", delta=f"Grounded in {res['best_source']}", delta_color="off")
        
    # Assembled RAG Prompt & Verified Answer Display
    st.markdown(f"""
    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid {res['badge_color']}; border-radius: 6px; padding: 0.8rem 1.1rem; margin-top: 0.4rem; margin-bottom: 0.8rem;">
        <span style="font-weight: 700; color: #166534; font-size: 0.9rem;">
            🤖 Verified RAG Grounded Answer ({res['verdict']}):
        </span>
        <div style="font-size: 0.88rem; color: #1e293b; margin-top: 0.3rem; line-height: 1.5;">
            {res['generated_answer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📚 Retrieved Policy Context Chunks")
    st.dataframe(res["retrieved_chunks"], use_container_width=True)


# =====================================================================
# 4. Standard 4-Tab Educational Containers
# =====================================================================

def render_nlp_tab1_theory(meta: Dict[str, Any]) -> None:
    """Tab 1: Mathematical Theory & Plain English Formula Breakdowns."""
    st.markdown("### 📖 Methodological Foundations & Mathematical Formulations")
    
    st.markdown("#### 🎯 Practical & Analytical When to Use")
    for item in meta["when_to_use"]:
        st.markdown(f"- {item}")
        
    st.markdown("#### 📐 Mathematical Derivations & Plain-English Explanations (Zero Prior Knowledge Assumed)")
    st.markdown("Below is a step-by-step mathematical and conceptual breakdown of every formula used in this analytical method:")
    
    for idx, (f_key, f_data) in enumerate(meta.get("formulas", {}).items(), 1):
        title = f_data.get("title", f_key.replace("_", " ").title())
        with st.expander(f"{idx}. {title}", expanded=False):
            st.markdown("#### Mathematical Representation:")
            st.latex(f_data["latex"])
            
            st.markdown("#### Plain English Breakdown of Every Term:")
            if "terms" in f_data and f_data["terms"]:
                for term, desc in f_data["terms"]:
                    term_str = term if term.startswith("$") else f"${term}$"
                    st.markdown(f"* **{term_str}**: {desc}")
            else:
                st.markdown(f"* **Summary:** {f_data['plain_english']}")
                
            if "steps" in f_data and f_data["steps"]:
                st.markdown("#### Step-by-Step Computational Procedure:")
                for step in f_data["steps"]:
                    st.markdown(f"{step}")
                    
            if "why_used" in f_data and f_data["why_used"]:
                st.markdown("#### Why This Calculation is Being Used & Practical Intuition:")
                st.markdown(f_data["why_used"])
                
    st.markdown("#### 📏 Plain English Interpretation Benchmarks & Rules of Thumb")
    for rule in meta.get("plain_english_rules", []):
        st.markdown(f"- {rule}")
        
    st.markdown("#### ⚖️ Key Assumptions & Common Pitfalls")
    c_a, c_p = st.columns(2)
    with c_a:
        st.markdown("**Core Methodological Assumptions:**")
        for a in meta.get("key_assumptions", []):
            st.markdown(f"- {a}")
    with c_p:
        st.markdown("**Common Pitfalls & Traps:**")
        for p in meta.get("common_pitfalls", []):
            st.markdown(f"- {p}")


def render_nlp_tab2_excel(meta: Dict[str, Any]) -> None:
    """Tab 2: Step-by-Step Microsoft Excel Guide & Tooling Assessment."""
    st.markdown("### 📗 Microsoft Excel Implementation Blueprint & Tooling Assessment")
    
    excel_info = meta.get("excel_guide", {})
    if excel_info.get("is_not_viable", False):
        st.warning("⚠️ **Architectural Feasibility Notice: RAG and Foundation LLMs are Not Viable for Native Execution in Microsoft Excel**")
        
        st.markdown(r"""
        #### 🛑 Why RAG & Large Language Models Cannot Run Natively in Excel:
        * **1. Massive Compute & Tensor Memory Scale:** Modern generative Large Language Models (7 Billion to 70+ Billion parameters) require billions of matrix multiplications, FP16/INT4 tensor quantization, and GPU acceleration (NVIDIA CUDA, Apple Metal MPS, TensorRT) that standalone Excel formula engines cannot execute.
        * **2. High-Dimensional Vector Embeddings & Indexing:** RAG relies on generating 384- to 1536-dimensional continuous floating-point vector representations and performing sub-millisecond Approximate Nearest Neighbor (ANN) vector lookups (HNSW graph indexing) across thousands of chunked documents—a task that exceeds spreadsheet calculation engines.
        * **3. Dynamic Subword Tokenization & KV-Caching:** Autoregressive LLM generation requires Byte-Pair Encoding (BPE) tokenizers and multi-turn Key-Value attention caching, which cannot be represented using standard spreadsheet cell formulas (`VLOOKUP`, `INDEX/MATCH`, etc.).
        """)
        
        st.markdown("#### 🛠️ Recommended Production Tooling Ecosystem for RAG & LLMs")
        st.markdown("""
        To build robust, production-grade RAG and LLM classification systems, data and engineering teams utilize the following dedicated tooling stack:
        
        | Architectural Layer | Recommended Industry Tooling | Role in RAG Pipeline |
        | :--- | :--- | :--- |
        | **Dense Embedding Models** | `sentence-transformers` (`all-MiniLM-L6-v2`, `bge-large-en`), OpenAI `text-embedding-3`, Cohere Embed | Converts document passages into dense 384–1536 dimensional semantic vectors. |
        | **Vector Databases (ANN Indexing)** | **Qdrant**, **ChromaDB**, **Pinecone**, **pgvector** (PostgreSQL), **FAISS** | Fast, scalable approximate nearest neighbor search across document chunks. |
        | **RAG Orchestration Frameworks** | **LangChain**, **LlamaIndex**, **Haystack**, **Semantic Kernel** | Manages document chunking, metadata filtering, retriever pipelines, and prompt injection. |
        | **Inference Runtimes & Serving** | **Hugging Face `transformers`**, **vLLM**, **Ollama**, **TensorRT-LLM**, **llama.cpp** | High-throughput, low-latency GPU/CPU token generation with quantized models. |
        | **Evaluation & Hallucination Auditing** | **RAGAS**, **TruLens**, **DeepEval** | Automated scoring of Faithfulness, Answer Relevance, and Context Precision. |
        | **Low-Code / Enterprise Interface** | **Langflow**, **Flowise**, **Microsoft Copilot Studio** | Visual workflow builders for teams requiring no-code or GUI-based RAG deployment. |
        """)
        
        with st.expander("🔌 How Excel Can Interface with External LLMs (Client-Side API Integration Pattern)", expanded=False):
            st.markdown("""
            While Excel **cannot run LLMs locally**, organizations requiring spreadsheet interfaces for business analysts can connect Excel as a **lightweight client** to an external LLM server:
            1. **Power Query REST API Calls:** Use `Web.Contents()` in Power Query M-code to send ticket text via `POST` requests to an internal vLLM or Azure OpenAI endpoint.
            2. **Python in Excel (Microsoft 365):** Use official Microsoft `PY()` worksheet functions to call cloud-hosted containerized Python scripts that query vector databases and return grounded answers directly into worksheet cells.
            3. **Custom VBA HTTP Requests:** Use `MSXML2.ServerXMLHTTP60` to dispatch user queries to a private company RAG endpoint (`https://internal-rag.corp/api/query`) and parse the JSON response into Excel columns.
            """)
    else:
        st.info(f"**Recommended Spreadsheet Layout:** {excel_info.get('cell_mapping', '')}")
        st.markdown("#### 📋 Step-by-Step Spreadsheet Procedure")
        for step in excel_info.get("steps", []):
            st.markdown(f"{step}")


def render_nlp_tab3_python(meta: Dict[str, Any]) -> None:
    """Tab 3: Production-Ready Python Pipeline."""
    st.markdown("### 🐍 Production Python Pipeline")
    st.code(meta["python_code"], language="python")


def render_nlp_tab4_data(meta: Dict[str, Any]) -> None:
    """Tab 4: Synthetic Dataset Preview & 1-Click CSV Download."""
    st.markdown("### 💾 Hands-On Practice Dataset")
    st.markdown("Download this realistic synthetic business/operational dataset to test the analytical steps in Microsoft Excel or Python.")
    st.caption("💡 *Educational Disclaimer: All records, feedback narratives, and ticket IDs are entirely simulated and fabricated solely for educational and illustrative purposes.*")
    
    df_data = generate_synthetic_nlp_data(meta["id"], n_samples=30, seed=42)
    st.dataframe(df_data, use_container_width=True)
    
    csv_bytes = df_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {meta['name']} Synthetic CSV",
        data=csv_bytes,
        file_name=f"{meta['id']}_synthetic_practice_data.csv",
        mime="text/csv",
        key=f"btn_download_nlp_{meta['id']}"
    )

