from __future__ import annotations

"""Survey & Questionnaire Analysis UI Components.

Provides interactive decision wizards, dynamic Plotly visualizations,
simulation controls, and the standard 4-tab pedagogical container
(Mathematical Formulation, Excel Guide, Python Pipeline, Synthetic Data Download).
"""

from typing import Dict, Any, List, Tuple

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_UI_STACK = True
except ImportError:
    HAS_UI_STACK = False
    st = None
    pd = None
    np = None
    px = None
    go = None

from src.survey_analysis.survey_schema import (
    SURVEY_METHODS,
    SURVEY_CATEGORIES,
    SURVEY_DECISION_NODES
)
from src.survey_analysis.survey_engine import (
    calculate_likert_metrics,
    calculate_cronbachs_alpha,
    calculate_mcdonalds_omega,
    calculate_cohens_kappa,
    calculate_factor_analysis,
    calculate_survey_sample_size_fpc,
    calculate_survey_weights_kish,
    generate_synthetic_survey_data
)
from src.ui_components.sample_size_badge import render_sample_size_deep_link_badge


# =====================================================================
# 1. Decision Wizard & Direct Explorer Selector
# =====================================================================

def render_survey_mode_selector() -> Tuple[str, str]:
    """Renders toggle between Guided Survey Wizard and Direct Method Explorer."""
    c_mode1, c_mode2 = st.columns([2.5, 1.5])
    with c_mode1:
        nav_mode = st.radio(
            "Navigation Mode:",
            options=["🎯 Guided Decision Wizard", "🔍 Direct Method Explorer"],
            horizontal=True,
            key="survey_nav_mode_radio"
        )
    with c_mode2:
        st.caption("Choose **Guided Decision Wizard** for problem-based recommendations or **Direct Explorer** to study a specific method.")
        
    if "Guided" in nav_mode:
        selected_method = render_survey_guided_wizard()
        return "wizard", selected_method
    else:
        selected_method = render_survey_direct_explorer()
        return "direct", selected_method


def render_survey_guided_wizard() -> str:
    """Renders an interactive step-by-step decision funnel for survey methodology."""
    st.markdown("### 🎯 Guided Survey Methodology Funnel")
    st.markdown("Select your primary research question or analytical bottleneck:")
    
    node = SURVEY_DECISION_NODES[0]
    options = node["options"]
    
    selected_option_label = st.selectbox(
        node["question"],
        options=[opt["label"] for opt in options],
        key="survey_wizard_goal_select"
    )
    
    # Map label to recommendation
    selected_opt = next(opt for opt in options if opt["label"] == selected_option_label)
    rec_method_key = selected_opt["recommendation"]
    method_meta = SURVEY_METHODS[rec_method_key]
    
    st.markdown(f"""
    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid #16a34a; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; margin-bottom: 1rem;">
        <span style="font-weight: 700; color: #166534; font-size: 0.95rem;">
            🎯 Recommended Survey Method: {method_meta['icon']} {method_meta['name']}
        </span>
        <div style="font-size: 0.85rem; color: #14532d; margin-top: 0.25rem; line-height: 1.4;">
            {method_meta['summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return rec_method_key


def render_survey_direct_explorer() -> str:
    """Renders direct selectbox categorized by the 5 survey clusters."""
    st.markdown("### 🔍 Direct Survey Method Explorer")
    
    method_choices = {
        f"{m['icon']} {m['name']}": k for k, m in SURVEY_METHODS.items()
    }
    
    selected_label = st.selectbox(
        "Select survey method or psychometric tool:",
        options=list(method_choices.keys()),
        index=0,
        key="survey_direct_method_select"
    )
    
    return method_choices[selected_label]


# =====================================================================
# 2. Main Method Module Renderer
# =====================================================================

def render_survey_method_module(method_key: str) -> None:
    """Renders the comprehensive 4-tab module for the selected survey method."""
    meta = SURVEY_METHODS.get(method_key)
    if not meta:
        st.error(f"Unknown survey method key: {method_key}")
        return
        
    cat_info = SURVEY_CATEGORIES.get(meta["category"], {})
    
    # Method Header Card
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">
                {meta['icon']} {meta['name']}
            </span>
            <span style="background-color: #eff6ff; color: #1e40af; font-size: 0.78rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid #bfdbfe;">
                {cat_info.get('name', 'Survey Methodology')}
            </span>
        </div>
        <div style="font-size: 0.92rem; color: #334155; margin-top: 0.4rem; line-height: 1.5;">
            {meta['summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Simulation & Visual Playground
    st.markdown("### 🔬 Live Interactive Simulation & Diagnostic Visualizer")
    
    if method_key == "likert_scale_analysis":
        render_likert_interactive_playground(meta)
    elif method_key == "cronbachs_alpha":
        render_cronbach_interactive_playground(meta)
    elif method_key == "mcdonalds_omega":
        render_omega_interactive_playground(meta)
    elif method_key == "cohens_kappa":
        render_kappa_interactive_playground(meta)
    elif method_key == "exploratory_factor_analysis":
        render_efa_interactive_playground(meta)
    elif method_key == "survey_sample_size":
        render_survey_sample_size_playground(meta)
    elif method_key == "survey_weighting":
        render_survey_weighting_playground(meta)
        
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
        render_survey_tab1_theory(meta)
    with tab2:
        render_survey_tab2_excel(meta)
    with tab3:
        render_survey_tab3_python(meta)
    with tab4:
        render_survey_tab4_data(meta)


# =====================================================================
# 3. Interactive Playgrounds per Method
# =====================================================================

def render_likert_interactive_playground(meta: Dict[str, Any]) -> None:
    """Interactive diverging stacked bar chart and Likert metrics."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Sample Size (Respondents N):", 50, 1000, 200, 25, key="likert_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="likert_seed")
    st.caption("Simulating a 5-item Patient Reported Experience Measure (PREM) across clinical communication, dignity, and overall care.")
        
    df = generate_synthetic_survey_data("likert_scale_analysis", n_samples=n_samples, seed=int(seed_val))
    item_cols = [c for c in df.columns if c != "Respondent_ID"]
    res = calculate_likert_metrics(df[item_cols])
    df_sum = res["item_summary"]
    comp = res["composite"]
    
    # Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Composite Mean", f"{comp['mean']:.2f}")
    with m2:
        st.metric("Composite SD", f"{comp['sd']:.2f}")
    with m3:
        st.metric("Composite Median", f"{comp['median']:.1f}")
    with m4:
        st.metric("Skewness (Normality)", f"{comp['skewness']:.2f}", delta="Parametric OK" if comp['is_approximately_normal'] else "Skewed")
        
    # Norman 2010 Consensus Card
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.3rem; margin-bottom: 0.75rem;">
        <span style="font-weight: 700; color: #1e293b; font-size: 0.88rem;">
            ⚖️ Norman (2010) Composite Score Consensus:
        </span>
        <div style="font-size: 0.83rem; color: #475569; margin-top: 0.2rem; line-height: 1.45;">
            The composite summated score across these 5 questions has <strong>Mean = {comp['mean']}</strong> (SD = {comp['sd']}) and <strong>Median = {comp['median']}</strong> (IQR = {comp['iqr']}) with skewness = {comp['skewness']}.<br>
            <em>Methodological Verdict:</em> {'✅ Composite score qualifies for parametric tests (t-test, ANOVA, OLS) under the Central Limit Theorem.' if comp['is_approximately_normal'] else '⚠️ Composite score exhibits notable skewness; consider non-parametric tests or ordinal logistic regression.'}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Full Page-Width Diverging Stacked Bar Chart
    categories = df_sum["Item"].tolist()
    fig = go.Figure()
    
    # Negative bars (Disagreement)
    fig.add_trace(go.Bar(
        y=categories,
        x=-df_sum["Pct_Strongly_Disagree"],
        name="Strongly Disagree",
        orientation='h',
        marker=dict(color="#dc2626")
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=-df_sum["Pct_Disagree"],
        name="Disagree",
        orientation='h',
        marker=dict(color="#f87171")
    ))
    # Neutral bar
    fig.add_trace(go.Bar(
        y=categories,
        x=df_sum["Pct_Neutral"],
        name="Neutral",
        orientation='h',
        marker=dict(color="#94a3b8")
    ))
    # Positive bars (Agreement)
    fig.add_trace(go.Bar(
        y=categories,
        x=df_sum["Pct_Agree"],
        name="Agree",
        orientation='h',
        marker=dict(color="#4ade80")
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=df_sum["Pct_Strongly_Agree"],
        name="Strongly Agree",
        orientation='h',
        marker=dict(color="#16a34a")
    ))
    
    fig.update_layout(
        barmode='relative',
        title="Diverging Likert Response Distribution (% Negative vs Neutral vs % Positive)",
        xaxis=dict(title="% of Respondents (Centered on Neutral)"),
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=180, r=30, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### 📊 Item-by-Item Summary Statistics")
    st.dataframe(df_sum, use_container_width=True)


def render_cronbach_interactive_playground(meta: Dict[str, Any]) -> None:
    """Interactive Cronbach's Alpha calculation, CI, and item-deleted table."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Sample Size (N):", 50, 500, 150, 25, key="cronbach_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="cronbach_seed")
    st.caption("Simulating a 6-item Clinician Burnout & Work Exhaustion Scale. Item 6 has lower correlation to demonstrate the 'Alpha If Deleted' purification rule.")
        
    df = generate_synthetic_survey_data("cronbachs_alpha", n_samples=n_samples, seed=int(seed_val))
    item_cols = [c for c in df.columns if c != "Clinician_ID"]
    res = calculate_cronbachs_alpha(df[item_cols])
    
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

    m_row1_c1, m_row1_c2 = st.columns(2)
    with m_row1_c1:
        st.metric("Cronbach's Alpha (α)", f"{res['alpha']:.3f}")
    with m_row1_c2:
        st.metric("Standardized Alpha", f"{res['alpha_standardized']:.3f}")

    m_row2_c1, m_row2_c2 = st.columns(2)
    with m_row2_c1:
        st.metric("95% CI (Feldt)", f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]")
    with m_row2_c2:
        st.metric(
            label="Reliability Rating",
            value=res.get("rating_tier", "Acceptable"),
            delta=res.get("rating_detail", "Standard academic research benchmark"),
            delta_color="off"
        )
        
    st.markdown(f"""
    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid {res['badge_color']}; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.3rem; margin-bottom: 0.75rem;">
        <strong>Internal Consistency Benchmark:</strong> {res['rating']} (George & Mallery standard: α ≥ 0.70 acceptable, ≥ 0.80 good)
    </div>
    """, unsafe_allow_html=True)
    
    # Full Page-Width Correlation Heatmap
    corr = res["correlation_matrix"]
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Inter-Item Pearson Correlation Matrix"
    )
    fig_heat.update_layout(height=340, margin=dict(l=60, r=30, t=35, b=25))
    st.plotly_chart(fig_heat, use_container_width=True)
        
    st.markdown("#### 🔬 Item-Total Reliability & 'Alpha If Item Deleted' Diagnostic Table")
    st.dataframe(res["item_analysis"], use_container_width=True)


def render_omega_interactive_playground(meta: Dict[str, Any]) -> None:
    """Interactive McDonald's Omega vs Cronbach's Alpha comparison."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Sample Size (N):", 50, 500, 150, 25, key="omega_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="omega_seed")
    st.caption("McDonald's Omega accounts for unequal factor loadings (λ), overcoming alpha's systematic underestimation bias.")
        
    df = generate_synthetic_survey_data("mcdonalds_omega", n_samples=n_samples, seed=int(seed_val))
    item_cols = [c for c in df.columns if c != "Clinician_ID"]
    res = calculate_mcdonalds_omega(df[item_cols])
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("McDonald's Omega (ω)", f"{res['omega_total']:.3f}")
    with m2:
        st.metric("Cronbach's Alpha (α)", f"{res['cronbach_alpha']:.3f}")
    with m3:
        st.metric("Discrepancy (ω - α)", f"+{res['discrepancy_omega_minus_alpha']:.3f}")
        
    tau_text = "⚠️ <strong>Tau-Equivalence Violated:</strong> Factor loadings vary significantly across items. Cronbach's alpha underestimates true reliability. McDonald's omega is the statistically sound estimate." if res['tau_equivalence_violated'] else "✅ <strong>Tau-Equivalence Holds:</strong> Factor loadings are relatively uniform. Alpha and Omega yield consistent estimates."
    st.markdown(f"""
    <div style="background-color: #fefce8; border: 1px solid #fef08a; border-left: 4px solid #ca8a04; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.3rem; margin-bottom: 0.75rem; font-size: 0.85rem; color: #713f12;">
        {tau_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Full Page-Width Bar Chart of Loadings vs Uniqueness Error
    df_l = res["loadings_table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_l["Item"],
        y=df_l["True_Variance_λ2"],
        name="True Factor Variance (λ²)",
        marker_color="#2563eb"
    ))
    fig.add_trace(go.Bar(
        x=df_l["Item"],
        y=df_l["Uniqueness_Error_u"],
        name="Uniqueness Error Variance (u)",
        marker_color="#cbd5e1"
    ))
    fig.update_layout(
        barmode="stack",
        title="Item Variance Decomposition: True Latent Variance vs Uniqueness Error",
        yaxis=dict(title="Proportion of Variance (Total = 1.0)"),
        height=330,
        margin=dict(l=60, r=30, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### ⚖️ Standardized Factor Loadings & Uniqueness Table")
    st.dataframe(res["loadings_table"], use_container_width=True)


def render_kappa_interactive_playground(meta: Dict[str, Any]) -> None:
    """Interactive Cohen's Kappa & Weighted Kappa confusion matrix."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Patients Evaluated (N):", 30, 300, 100, 10, key="kappa_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="kappa_seed")
    st.caption("Simulating clinical triage ratings by two independent clinicians: Nurse vs Senior Clinician across 3 urgency categories.")
        
    df = generate_synthetic_survey_data("cohens_kappa", n_samples=n_samples, seed=int(seed_val))
    r1 = df["Rater1_Triage_Nurse"].tolist()
    r2 = df["Rater2_Senior_Clinician"].tolist()
    res = calculate_cohens_kappa(r1, r2)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Quadratic Weighted κ", f"{res['kappa_quadratic_weighted']:.3f}")
    with m2:
        st.metric("Unweighted κ", f"{res['kappa_unweighted']:.3f}")
    with m3:
        st.metric("Observed Agreement", f"{res['observed_agreement_pct']}%", delta=f"{res['observed_agreement_pct'] - res['expected_agreement_pct']:.1f}% vs chance")
        
    st.markdown(f"""
    <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-left: 4px solid {res['badge_color']}; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.3rem; margin-bottom: 0.75rem;">
        <strong>Landis & Koch Benchmark:</strong> {res['benchmark']} (p = {res['p_value']:.4f})
    </div>
    """, unsafe_allow_html=True)
    
    # Full Page-Width Heatmap of Confusion Matrix
    conf = res["confusion_matrix"]
    fig = px.imshow(
        conf,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Cross-Classification Matrix (Rater 1 vs Rater 2)"
    )
    fig.update_layout(height=340, margin=dict(l=60, r=30, t=35, b=25))
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### 👥 Confusion Matrix & Marginal Totals")
    st.dataframe(res["confusion_matrix"], use_container_width=True)


def render_efa_interactive_playground(meta: Dict[str, Any]) -> None:
    """Interactive Exploratory Factor Analysis & Scree Plot."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Sample Size (N):", 50, 500, 200, 25, key="efa_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="efa_seed")
    st.caption("Simulating a 9-item Quality of Life (QoL) scale with 3 hypothesized latent constructs: Physical, Psychological, and Social.")
        
    df = generate_synthetic_survey_data("exploratory_factor_analysis", n_samples=n_samples, seed=int(seed_val))
    item_cols = [c for c in df.columns if c != "Patient_ID"]
    res = calculate_factor_analysis(df[item_cols])
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("KMO Sampling Adequacy", f"{res['kmo']:.3f}")
    with m2:
        st.metric("Bartlett's Test", f"p = {res['bartlett_p_value']:.4f}")
    with m3:
        st.metric("Retained Factors (λ ≥ 1.0)", f"{res['n_factors_kaiser']}")
        
    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid {res['kmo_color']}; border-radius: 6px; padding: 0.6rem 0.9rem; margin-top: 0.3rem; margin-bottom: 0.75rem;">
        <strong>KMO Verdict:</strong> {res['kmo_label']} (Bartlett's χ² = {res['bartlett_chi2']}, df = {res['bartlett_df']})
    </div>
    """, unsafe_allow_html=True)
    
    # Full Page-Width Scree Plot
    df_scree = res["scree_table"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_scree["Factor"],
        y=df_scree["Eigenvalue"],
        mode="lines+markers",
        name="Eigenvalue",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=8)
    ))
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Kaiser Criterion (λ = 1.0)",
        annotation_position="bottom right"
    )
    fig.update_layout(
        title="Cattell's Scree Plot & Kaiser Retention Boundary",
        yaxis=dict(title="Eigenvalue (Variance Explained)"),
        height=330,
        margin=dict(l=60, r=30, t=35, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### 🧩 Rotated Factor Loadings Matrix (Varimax)")
    st.dataframe(res["loadings_table"], use_container_width=True)


def render_survey_sample_size_playground(meta: Dict[str, Any]) -> None:
    """Interactive Survey Sample Size with Finite Population Correction."""
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        pop_n = st.number_input("Finite Population (N):", min_value=100, max_value=1000000, value=5000, step=500, key="survey_pop_n")
    with col_c2:
        moe = st.slider("Margin of Error (±e):", 0.01, 0.10, 0.05, 0.005, format="±%.3f", key="survey_moe_slider")
    with col_c3:
        conf = st.selectbox("Confidence Level:", [0.90, 0.95, 0.99], index=1, key="survey_conf_select")
    with col_c4:
        resp_rate = st.slider("Expected Response Rate:", 0.10, 1.00, 0.35, 0.05, format="%.2f", key="survey_rr_slider")
    st.caption("Cochran's formula adjusted with Finite Population Correction (FPC) and non-response invitation inflation.")
        
    res = calculate_survey_sample_size_fpc(
        population_size_n=int(pop_n),
        confidence_level=float(conf),
        margin_of_error=float(moe),
        proportion_p=0.50,
        response_rate=float(resp_rate)
    )
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Evaluable Sample Required", f"{res['n_evaluable_fpc']} surveys", delta=f"-{res['fpc_savings_pct']}% FPC savings")
    with m2:
        st.metric("Invitations to Dispatch", f"{res['n_invitations_required']}", help=f"Inflated for {int(resp_rate*100)}% expected response rate")
    with m3:
        st.metric("Infinite Baseline (n₀)", f"{res['n_infinite_n0']} surveys")
        
    # Full Page-Width Plotly curve: Population N vs Sample Size n
    pop_range = np.logspace(2, 5, 50).astype(int)
    sample_sizes = [calculate_survey_sample_size_fpc(p, conf, moe)["n_evaluable_fpc"] for p in pop_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pop_range,
        y=sample_sizes,
        mode="lines",
        name="Required n (FPC)",
        line=dict(color="#16a34a", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[pop_n],
        y=[res["n_evaluable_fpc"]],
        mode="markers",
        name="Your Selected Population",
        marker=dict(color="#dc2626", size=10)
    ))
    fig.update_layout(
        title=f"Sample Size vs Population Size (at ±{int(moe*100)}% Margin of Error)",
        xaxis=dict(title="Population Size N (Log Scale)", type="log"),
        yaxis=dict(title="Required Evaluable Sample n"),
        height=330,
        margin=dict(l=60, r=30, t=35, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### 📐 Sensitivity Matrix: Required Sample Size Across Margins of Error")
    st.dataframe(res["sensitivity_table"], use_container_width=True)


def render_survey_weighting_playground(meta: Dict[str, Any]) -> None:
    """Interactive Post-Stratification Weighting & Kish Effective Sample Size."""
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        n_samples = st.slider("Collected Sample Size (n):", 100, 1000, 300, 50, key="weight_n_slider")
    with col_c2:
        seed_val = st.number_input("Random Seed:", value=42, step=1, key="weight_seed")
    st.caption("Simulating patient feedback survey where Young Adults (18-34) are underrepresented and Older Adults (65+) are overrepresented.")
        
    df = generate_synthetic_survey_data("survey_weighting", n_samples=n_samples, seed=int(seed_val))
    target_pop = {
        "18-34 (Young Adult)": 0.30,
        "35-64 (Working Age)": 0.50,
        "65+ (Older Adult)": 0.20
    }
    res = calculate_survey_weights_kish(
        df_responses=df,
        group_col="Age_Group",
        target_pop_props=target_pop,
        score_col="Satisfaction_Score_1_to_10"
    )
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Kish Effective N (Neff)", f"{res['n_effective_kish']:.0f}", delta=f"{res['efficiency_pct'] - 100:.1f}% efficiency")
    with m2:
        st.metric("Kish Design Effect (Deff)", f"{res['design_effect_deff']:.3f}", help="Deff = 1 + CV(w)^2: variance inflation factor caused by unequal weighting")
    with m3:
        st.metric("Weighted Mean Score", f"{res['weighted_mean']:.2f}", delta=f"{res['weighted_mean'] - res['unweighted_mean']:.2f} vs unweighted")
        
    # Full Page-Width Comparison Bar Chart: Population % vs Sample %
    df_w = res["weight_table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_w["Demographic_Group"],
        y=df_w["Target_Population_%"],
        name="True Population %",
        marker_color="#2563eb"
    ))
    fig.add_trace(go.Bar(
        x=df_w["Demographic_Group"],
        y=df_w["Sample_%"],
        name="Sample Returns %",
        marker_color="#f59e0b"
    ))
    fig.update_layout(
        barmode="group",
        title="Demographic Skew: Population Benchmark vs Collected Sample",
        yaxis=dict(title="% of Total"),
        height=330,
        margin=dict(l=60, r=30, t=35, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("#### ⚖️ Post-Stratification Cell Weights Table")
    st.dataframe(res["weight_table"], use_container_width=True)


# =====================================================================
# 4. Standard 4-Tab Educational Renderers
# =====================================================================

def render_survey_tab1_theory(meta: Dict[str, Any]) -> None:
    """Tab 1: Conceptual Explanation, Formulations & Plain English Breakdowns."""
    st.markdown("### 📖 Mathematical Foundations & Biostatistical Logic")
    
    st.markdown("#### 🎯 When to Use")
    for item in meta["when_to_use"]:
        st.markdown(f"- {item}")
        
    st.markdown("#### 📐 Mathematical Formulations & Plain English Explanations")
    for f_name, f_data in meta.get("formulas", {}).items():
        st.latex(f_data["latex"])
        st.markdown(f"""
        <div style="background-color: #f1f5f9; border-left: 3px solid #3b82f6; border-radius: 4px; padding: 0.5rem 0.85rem; margin-bottom: 0.8rem; font-size: 0.86rem; color: #1e293b;">
            <strong>Plain English Explanation:</strong> {f_data['plain_english']}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### 📏 Plain English Interpretation Benchmarks & Rules of Thumb")
    for rule in meta.get("plain_english_rules", []):
        st.markdown(f"- {rule}")
        
    st.markdown("#### ⚠️ Key Assumptions & Common Pitfalls")
    c_a, c_p = st.columns(2)
    with c_a:
        st.markdown("**Core Assumptions:**")
        for a in meta.get("key_assumptions", []):
            st.markdown(f"- {a}")
    with c_p:
        st.markdown("**Common Pitfalls:**")
        for p in meta.get("common_pitfalls", []):
            st.markdown(f"- {p}")


def render_survey_tab2_excel(meta: Dict[str, Any]) -> None:
    """Tab 2: Step-by-Step Microsoft Excel Implementation Guide."""
    st.markdown("### 📗 Step-by-Step Microsoft Excel Implementation Guide")
    excel_info = meta.get("excel_guide", {})
    
    st.markdown(excel_info.get("description", ""))
    
    st.markdown("#### 📋 Recommended Cell Layout")
    cell_data = excel_info.get("cell_mapping", {})
    df_cells = pd.DataFrame(
        [{"Cell Address / Range": k, "Contents / Formula": v} for k, v in cell_data.items()]
    )
    st.dataframe(df_cells, use_container_width=True)
    
    st.markdown("#### 🛠️ Step-by-Step Instructions")
    for s in excel_info.get("steps", []):
        st.markdown(s)


def render_survey_tab3_python(meta: Dict[str, Any]) -> None:
    """Tab 3: Standalone Production Python Pipeline."""
    st.markdown("### 🐍 Production Python Implementation Pipeline")
    st.markdown("Copy, paste, and run this standalone Python script in your analytical workflow:")
    
    py_code = meta.get("python_code", "# Python implementation")
    st.code(py_code, language="python")


def render_survey_tab4_data(meta: Dict[str, Any]) -> None:
    """Tab 4: Synthetic Dataset Download & Data Preview."""
    st.markdown("### 💾 Synthetic Dataset Download")
    st.markdown("Explore this worked survey example using realistic simulated healthcare or operational data:")
    
    method_key = meta["id"]
    df_synth = generate_synthetic_survey_data(method_key, n_samples=200, seed=42)
    
    st.dataframe(df_synth.head(10), use_container_width=True)
    st.caption(f"Showing first 10 rows of {len(df_synth)} simulated survey records.")
    
    csv_bytes = df_synth.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {meta['name']} Synthetic Dataset (CSV)",
        data=csv_bytes,
        file_name=f"synthetic_survey_{method_key}.csv",
        mime="text/csv",
        key=f"btn_dl_survey_{method_key}"
    )
