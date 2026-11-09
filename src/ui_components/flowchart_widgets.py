"""Flowchart and Decision Tree UI Components.

Provides comprehensive visual decision trees, multi-pillar triage diagrams,
interactive path simulators, and educational triage rules across:
1. Master Cross-Pillar Triage (Statistics vs Epidemiology vs Machine Learning vs Sample Size vs NLP)
2. Classical Statistical Hypothesis Testing (18 Tests, Parametric & Non-Parametric)
3. Observational Epidemiology & Public Health (17 Methods, Risk, Rates & Screening)
4. Machine Learning & Predictive Modeling (16 Architectures, Rare Events & Clustering)
5. Biostatistical Sample Size & Power Planning (10 Frameworks, Attrition & Clusters)
6. Qualitative Analysis & Natural Language Processing (7 Methods, Rule-Based to LLMs)
"""

import html
from typing import Dict, Any, Optional
import streamlit as st


def render_html(html_str: str) -> None:
    """Safely renders HTML content, stripping excessive indentation to prevent

    CommonMark code-block misinterpretation, and falls back to st.markdown.
    """
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    cleaned_html = "".join(clean_lines)
    if hasattr(st, "html"):
        st.html(cleaned_html)
    else:
        st.markdown(cleaned_html, unsafe_allow_html=True)


# =====================================================================
# 1. Master Cross-Pillar Triage Flowchart & Guide
# =====================================================================

def render_master_triage_flowchart() -> None:
    """Renders the high-level triage flowchart mapping real-world research questions

    to the appropriate quantitative and qualitative analytical pillar.
    """
    st.markdown("### 🌐 Master Cross-Pillar Triage: Choosing Your Analytical Pillar")
    st.markdown("""
    When designing an investigation, the most fundamental decision is determining **which analytical paradigm** fits your research question, data collection process, and governance requirements:
    * **Pillar 1: Classical Statistics** — Hypothesis testing, $p$-values, effect sizes, and confidence intervals under controlled or randomized conditions.
    * **Pillar 2: Observational Epidemiology** — Quantifying disease risk, incidence rates, diagnostic accuracy, and controlling confounding without experimental randomization.
    * **Pillar 3: Machine Learning & AI** — Predictive scoring, handling high-dimensional complex patterns, rare event detection, and unsupervised phenotypic discovery.
    * **Pillar 4: Biostatistical Sample Size & Power** — Formal trial planning, minimal detectable differences, and ethical sample sizing.
    * **Pillar 5: Qualitative Analysis & NLP** — Extracting structured insights, sentiment, negation, topics, clinical entities, and factual synthesis from unstructured text narratives.
    * **Pillar 8: Causal Inference & Policy Evaluation** — Quasi-experimental architectures (DiD, ITS, PSM, RDD, Instrumental Variables) to infer counterfactual cause-and-effect when RCTs are unethical or unfeasible.
    * **Pillar 9: Statistical Process Control & QI** — Shewhart control charts, Laney overdispersion corrections (p', u'), and NHS Making Data Count icons to distinguish common cause noise from authentic special causes.
    * **Pillar 10: Health Economics & HTA** — Cost-effectiveness analysis, ICERs, QALYs, 3.5% NICE discounting, Markov state transitions, Monte Carlo PSA, and 5-year NHS Budget Impact Analysis.
    * **Pillar 11: Missing Data Mechanisms & Imputation** — Rubin's taxonomy (MCAR vs MAR vs MNAR), Little's test, MICE with Rubin's pooling, algorithmic k-NN/MissForest, and ICH E9 (R1) tipping point sensitivity audits.
    * **Pillar 12: Longitudinal & Hierarchical Mixed-Effects Models** — Laird & Ware LMM formulation (random intercepts & slopes), GLMM subject-specific vs marginal odds ratios, RM-ANOVA sphericity & missing data recovery, 3-level organizational nesting, and Nakagawa conditional $R^2$.
    """)

    # Visual Mermaid Diagram
    mermaid_code = """```mermaid
flowchart TD
    Start(["Research Question or Analytical Problem"]) --> Q1{"What is the Primary Analytical Goal?"}

    Q1 -->|"Hypothesis Testing & Causal Inference"| Q2{"Was Data Collected under Controlled / Randomized Conditions?"}
    Q2 -->|"Yes: Controlled / RCT / Pre-Post"| P1["📊 Pillar 1: Classical Statistical Tests<br/>(t-tests, ANOVA, Mann-Whitney, Chi-Square)"]
    Q2 -->|"No: Observational Data"| QCausal{"Evaluating Policy Shock, Cutoff Rule, or Counterfactual?"}
    QCausal -->|"Yes: Quasi-Experimental Policy Evaluation"| P8["🔍 Pillar 8: Causal Inference & Policy Evaluation<br/>(DiD, ITS, Propensity Matching, RDD, IV/2SLS)"]
    QCausal -->|"No: Observational Cohort / Survey"| P2["🏥 Pillar 2: Observational Epidemiology<br/>(Relative Risk, Odds Ratio, Standardization, Confounding)"]

    Q1 -->|"Predictive Scoring, Automation or Pattern Discovery"| Q3{"Are Ground-Truth Target Labels Available?"}
    Q3 -->|"Yes: Target Y Known"| ML_Sup["🤖 Pillar 3: Supervised ML<br/>(Regularized Regression, Cost-Sensitive Ensembles, PR-AUC)"]
    Q3 -->|"No: Unsupervised Discovery"| ML_Unsup["🧩 Pillar 3: Unsupervised ML<br/>(K-Means Clustering, DBSCAN, PCA, Stability Metrics)"]

    Q1 -->|"Prospective Study Sizing & Protocol Design"| P4["📐 Pillar 4: Sample Size & Power<br/>(Closed-Form Formulas, MCID, Attrition Cushion, VIF)"]

    Q1 -->|"Unstructured Text & Qualitative Language Insights"| P5["🗣️ Pillar 5: Qualitative Analysis & NLP<br/>(Sentiment, NegEx, TF-IDF, Topics, Embeddings, LLM RAG)"]

    Q1 -->|"Quality Improvement & Healthcare Process Monitoring"| P9["📈 Pillar 9: Statistical Process Control & QI<br/>(Run Charts, Shewhart XmR, Laney p' & u', NHS MDC Icons)"]

    Q1 -->|"Cost-Effectiveness, QALYs & Healthcare Payer Decision"| P10["💷 Pillar 10: Health Economics & HTA<br/>(ICER & CE Plane, Markov Models, NMB, PSA & BIA)"]

    Q1 -->|"Missing Data Audits & Imputation Strategy"| P11["🧩 Pillar 11: Missing Data & Imputation<br/>(Little's Test, Listwise Hazard, MICE Pooling, k-NN, Tipping Point)"]

    Q1 -->|"Repeated Measures / Clustered Hierarchies"| P12["🪜 Pillar 12: Longitudinal & Mixed Models<br/>(Random Slopes, GLMM, Sphericity Breakdown, 3-Level Nesting, Nakagawa R²)"]

    P1 -.->|"Requires Power Sizing"| P4
    P2 -.->|"Requires Power Sizing"| P4
    ML_Unsup -.->|"Requires Sample Sizing"| P4

    classDef primary fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef epi fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef causal fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46;
    classDef ml fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef power fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412;
    classDef nlp fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef spc fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef hta fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef missing fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,color:#5b21b6;
    classDef decision fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#0f172a;

    class P1 primary;
    class P2 epi;
    class P8 causal;
    class ML_Sup,ML_Unsup ml;
    class P4 power;
    class P5 nlp;
    class P9 spc;
    classDef mixed fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#312e81;
    class P10 hta;
    class P11 missing;
    class P12 mixed;
    class Q1,Q2,QCausal,Q3 decision;
```"""
    st.markdown(mermaid_code)

    # Master Triage Comparison Matrix
    with st.expander("📋 Master Triage Matrix: 5-Way Cross-Pillar Comparison", expanded=False):
        matrix_html = r"""
        <div style="overflow-x: auto; margin-top: 0.5rem; margin-bottom: 0.5rem;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem; text-align: left;">
                <thead>
                    <tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                        <th style="padding: 0.65rem 0.85rem; color: #0f172a; font-weight: 700;">Criterion</th>
                        <th style="padding: 0.65rem 0.85rem; color: #166534; font-weight: 700;">📊 Classical Statistics</th>
                        <th style="padding: 0.65rem 0.85rem; color: #1e40af; font-weight: 700;">🏥 Observational Epidemiology</th>
                        <th style="padding: 0.65rem 0.85rem; color: #6b21a8; font-weight: 700;">🤖 Machine Learning</th>
                        <th style="padding: 0.65rem 0.85rem; color: #9a3412; font-weight: 700;">📐 Sample Size & Power</th>
                        <th style="padding: 0.65rem 0.85rem; color: #0369a1; font-weight: 700;">🗣️ Qualitative NLP</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Primary Objective</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Test predefined null hypothesis ($H_0$), confirm intervention effects</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Quantify disease risk, incidence rates, and control observational confounding</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Maximize out-of-sample prediction accuracy or discover latent patient clusters</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Determine minimum sample size needed to detect effect size $\delta$ at $\ge 80\%$ power</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Extract structured meaning, clinical entities, sentiment, themes, or synthesize answers from unstructured text</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Data Architecture</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Controlled experiments, randomized trials, small-to-moderate samples ($N < 500$)</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Cohorts, case-control matching, surveys, demographic registries</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">High-dimensional tabular data, telemetry streams, $N \ge 500$, many features</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Pre-trial planning stage before data collection commences</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Clinical narratives, discharge summaries, incident reports (Datix), open-ended survey comments, customer reviews</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Key Output Metrics</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">$t$-stat, $F$-ratio, $p$-value, Cohen's $d$, 95% Confidence Intervals</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Relative Risk (RR), Odds Ratio (OR), DSR, Bedside Risk Scores (Sullivan Points), Sensitivity/Specificity, NNT</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">PR-AUC, ROC-AUC, Brier Score, Silhouette, Davies-Bouldin, Calinski-Harabasz</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Total Required $N$, Per-Group $n$, Power Curves, Attrition Cushion ($N / (1-L)$)</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Sentiment valence, TF-IDF weights, Topic coherence ($C_v$ / NPMI), Cosine similarity, Entity span F1, Faithfulness ratio</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Primary Governance</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">CONSORT 2010, ICH-E9, APA Guidelines</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">STROBE Guidelines, PRISMA 2020, STARD, TRIPOD (Prognosis)</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">TRIPOD-AI, NHS AI Ethics, Four-Fifths 80% Rule</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">NIHR, MRC, FDA Statistical Guidance, SPIRIT</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Caldicott Principles, HIPAA de-identification, STROBE qualitative criteria, RAGAS Hallucination auditing</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        render_html(matrix_html)

    # Interactive Step-by-Step Triage Explorer
    st.markdown("#### 🧭 Interactive Pillar Triage Explorer")
    st.caption("Select your research constraints below to see the recommended pillar, primary method, and direct launcher.")

    c1, c2, c3 = st.columns(3)
    with c1:
        triage_goal = st.selectbox(
            "1. Primary Research Objective:",
            options=[
                "Confirm hypothesis / compare clinical groups",
                "Measure observational disease risk / exposure",
                "Evaluate a screening swab or diagnostic test",
                "Develop clinical risk score / bedside stratification (TRIPOD)",
                "Predict rare clinical event (e.g. ICU Sepsis)",
                "Forecast sequential time series demand",
                "Discover unknown sub-phenotypes / patient clusters",
                "Determine required sample size for a grant proposal",
                "Analyze unstructured text, clinical notes, or qualitative comments (NLP)",
                "Evaluate policy shock or causal effect without RCT (Quasi-Experimental)",
                "Monitor longitudinal healthcare process / detect special cause variation (SPC & QI)",
                "Evaluate cost-effectiveness, QALYs, or NHS budget impact (Health Economics & HTA)",
                "Audit missing data mechanisms / multiple imputation & sensitivity (Rubin MICE & Tipping Point)",
                "Model longitudinal repeated measures / hierarchical clusters (Random slopes, GLMM & 3-level nesting)"
            ],
            key="triage_goal_select"
        )
    with c2:
        triage_data = st.selectbox(
            "2. Study Architecture / Data Collection:",
            options=[
                "Randomized Controlled Trial (Parallel Groups)",
                "Paired Pre-Post Study (Same Subjects)",
                "Prospective Cohort (Followed over time)",
                "Retrospective Case-Control (1:k matching)",
                "Cross-Sectional Diagnostic Accuracy Study",
                "High-Dimensional EHR / Telemetry Matrix",
                "Pre-Study Design Planning Stage",
                "Unstructured Text / Clinical Notes / Survey Free-Text",
                "Sequential Process Time Series (Monthly/Weekly QI Audits)"
            ],
            key="triage_data_select"
        )
    with c3:
        triage_scale = st.selectbox(
            "3. Expected Sample Magnitude ($N$):",
            options=[
                "Small Cohort (N < 30 participants)",
                "Moderate Sample (30 <= N < 500 participants)",
                "Large Registry / Telemetry (N >= 500 records)"
            ],
            key="triage_scale_select"
        )

    # Resolve recommendation
    resolved_pillar = "Classical Statistics"
    resolved_method = "Welch's Independent Two-Sample t-test"
    resolved_page = "pages/1_📊_Statistical_Tests.py"
    resolved_icon = "📊"
    resolved_color = "#166534"
    resolved_bg = "#f0fdf4"
    resolved_border = "#16a34a"
    resolved_rationale = "Testing differences between 2 experimental groups with continuous endpoint."

    if "longitudinal" in triage_goal.lower() or "mixed" in triage_goal.lower() or "multilevel" in triage_goal.lower() or "hierarchical" in triage_goal.lower():
        resolved_pillar = "Longitudinal & Hierarchical Mixed-Effects Models"
        resolved_method = "Linear Mixed Models (LMM) / GLMM / 3-Level Multilevel Models"
        resolved_page = "pages/12_🪜_Longitudinal_and_Mixed_Models.py"
        resolved_icon = "🪜"
        resolved_color = "#312e81"
        resolved_bg = "#eef2ff"
        resolved_border = "#4f46e5"
        resolved_rationale = "Modeling repeated patient measurements, personalized trajectories (random intercepts & slopes), non-collapsible GLMM odds ratios, and partitioning variance across organizational hierarchies."
    elif "missing data" in triage_goal.lower() or "imputation" in triage_goal.lower() or "tipping point" in triage_goal.lower():
        resolved_pillar = "Missing Data Mechanisms & Imputation"
        resolved_method = "Rubin's Taxonomy / MICE (Chained Equations) / Tipping Point Analysis"
        resolved_page = "pages/11_🧩_Missing_Data_and_Imputation.py"
        resolved_icon = "🧩"
        resolved_color = "#5b21b6"
        resolved_bg = "#f5f3ff"
        resolved_border = "#7c3aed"
        resolved_rationale = "Auditing MCAR/MAR/MNAR mechanisms, avoiding single imputation variance collapse, pooling m datasets via Rubin's rules, and conducting ICH E9 tipping point sensitivity audits."
    elif "cost-effectiveness" in triage_goal.lower() or "qaly" in triage_goal.lower() or "budget impact" in triage_goal.lower() or "health economics" in triage_goal.lower():
        resolved_pillar = "Health Economics & Health Technology Assessment (HTA / NICE)"
        resolved_method = "Cost-Effectiveness (ICER / NMB) / Markov Cohort / PSA / Budget Impact Analysis"
        resolved_page = "pages/10_💷_Health_Economics_and_HTA.py"
        resolved_icon = "💷"
        resolved_color = "#854d0e"
        resolved_bg = "#fefce8"
        resolved_border = "#ca8a04"
        resolved_rationale = "Evaluating lifetime cost per QALY gained, opportunity costs, Markov chronic disease transitions, and 5-year payer affordability for NHS reimbursement."
    elif "special cause" in triage_goal.lower() or "spc" in triage_goal.lower() or "qi audits" in triage_data.lower():
        resolved_pillar = "Statistical Process Control (SPC) & Quality Improvement"
        resolved_method = "Run Charts / Shewhart XmR / Laney p' & u' / NHS Making Data Count"
        resolved_page = "pages/9_📈_Statistical_Process_Control_and_Quality_Improvement.py"
        resolved_icon = "📈"
        resolved_color = "#047857"
        resolved_bg = "#ecfdf5"
        resolved_border = "#10b981"
        resolved_rationale = "Distinguishing common cause from special cause variation, avoiding tampering, and evaluating QI interventions using Shewhart and Laney control limits."
    elif "unstructured text" in triage_goal.lower() or "free-text" in triage_data.lower() or "clinical notes" in triage_data.lower():
        resolved_pillar = "Qualitative Analysis & NLP"
        resolved_method = "Clinical Text & Natural Language Processing Pipeline"
        resolved_page = "pages/6_🗣️_Natural_Language_Processing.py"
        resolved_icon = "🗣️"
        resolved_color = "#0284c7"
        resolved_bg = "#f0f9ff"
        resolved_border = "#0ea5e9"
        resolved_rationale = "Analyzing qualitative, unstructured textual narratives across rule-based, statistical, or foundation LLM architectures."
    elif "policy shock" in triage_goal.lower() or "quasi-experimental" in triage_goal.lower() or "causal" in triage_goal.lower():
        resolved_pillar = "Causal Inference & Policy Evaluation"
        resolved_method = "Difference-in-Differences / Interrupted Time Series / RDD / PSM / IV"
        resolved_page = "pages/8_🔍_Causal_Inference_and_Policy_Evaluation.py"
        resolved_icon = "🔍"
        resolved_color = "#047857"
        resolved_bg = "#ecfdf5"
        resolved_border = "#10b981"
        resolved_rationale = "Evaluating real-world policy shocks, clinical guidelines, and counterfactual effects when RCTs are unethical or unfeasible."
    elif "grant proposal" in triage_goal or "Pre-Study" in triage_data:
        resolved_pillar = "Sample Size & Power"
        resolved_method = "Two Independent Means Sizing (t-test)"
        resolved_page = "pages/4_📐_Sample_Size_and_Power.py"
        resolved_icon = "📐"
        resolved_color = "#9a3412"
        resolved_bg = "#fff7ed"
        resolved_border = "#ea580c"
        resolved_rationale = "Formal prospective statistical power sizing required before data collection begins."
    elif "observational disease risk" in triage_goal:
        resolved_pillar = "Observational Epidemiology"
        if "Case-Control" in triage_data:
            resolved_method = "Odds Ratio (OR) with Rare Disease Check"
        else:
            resolved_method = "Relative Risk (RR) / Cumulative Incidence"
        resolved_page = "pages/2_🏥_Epidemiology_Methods.py"
        resolved_icon = "🏥"
        resolved_color = "#1e40af"
        resolved_bg = "#eff6ff"
        resolved_border = "#2563eb"
        resolved_rationale = "Quantifying exposure-disease association in an observational cohort or case-control study."
    elif "diagnostic test" in triage_goal:
        resolved_pillar = "Observational Epidemiology"
        resolved_method = "Diagnostic Accuracy (Sensitivity, Specificity, ROC-AUC)"
        resolved_page = "pages/2_🏥_Epidemiology_Methods.py"
        resolved_icon = "🎯"
        resolved_color = "#1e40af"
        resolved_bg = "#eff6ff"
        resolved_border = "#2563eb"
        resolved_rationale = "Validating index test performance against a verified gold standard reference."
    elif "clinical risk score" in triage_goal.lower() or "risk stratification" in triage_goal.lower():
        resolved_pillar = "Observational Epidemiology"
        resolved_method = "Clinical Risk Stratification & Prognostic Models (Sullivan Points, Calibration, DCA)"
        resolved_page = "pages/2_🏥_Epidemiology_Methods.py"
        resolved_icon = "🗂️"
        resolved_color = "#1e40af"
        resolved_bg = "#eff6ff"
        resolved_border = "#2563eb"
        resolved_rationale = "Formulating bedside integer point scores, assessing calibration & discrimination (TRIPOD), and evaluating clinical Net Benefit via Decision Curve Analysis."
    elif "Forecast sequential" in triage_goal:
        resolved_pillar = "Time Series & Operational Forecasting"
        resolved_method = "Time Series Analysis, Decomposition & Forecasting Models"
        resolved_page = "pages/7_⏱️_Time_Series_and_Forecasting.py"
        resolved_icon = "⏱️"
        resolved_color = "#0369a1"
        resolved_bg = "#f0f9ff"
        resolved_border = "#0ea5e9"
        resolved_rationale = "Analyzing sequential temporal dependencies, classical trend/seasonal decomposition, and evaluating models with rolling-origin temporal validation."
    elif "Predict rare clinical event" in triage_goal:
        resolved_pillar = "Machine Learning & AI"
        resolved_method = "Cost-Sensitive Random Forest (PR-AUC / Brier Score)"
        resolved_page = "pages/3_🤖_Machine_Learning.py"
        resolved_icon = "🤖"
        resolved_color = "#6b21a8"
        resolved_bg = "#faf5ff"
        resolved_border = "#9333ea"
        resolved_rationale = "Optimizing predictive generalizability and handling severe class imbalance."
    elif "sub-phenotypes" in triage_goal:
        resolved_pillar = "Machine Learning & AI"
        resolved_method = "K-Means Clustering with Silhouette, DBI & CHI Validation"
        resolved_page = "pages/3_🤖_Machine_Learning.py"
        resolved_icon = "🧩"
        resolved_color = "#6b21a8"
        resolved_bg = "#faf5ff"
        resolved_border = "#9333ea"
        resolved_rationale = "Unsupervised cluster discovery evaluated with rigorous stability criteria."
    elif "Paired Pre-Post" in triage_data:
        resolved_pillar = "Classical Statistics"
        if "Small Cohort" in triage_scale:
            resolved_method = "Wilcoxon Signed-Rank Test (Non-Parametric Fallback)"
        else:
            resolved_method = "Paired Samples t-test"
        resolved_page = "pages/1_📊_Statistical_Tests.py"
        resolved_icon = "🔄"
        resolved_color = "#166534"
        resolved_bg = "#f0fdf4"
        resolved_border = "#16a34a"
        resolved_rationale = "Within-subject repeated measures evaluated before and after an intervention."
    elif "Small Cohort" in triage_scale:
        resolved_pillar = "Classical Statistics"
        resolved_method = "Mann-Whitney U Test (Non-Parametric Fallback)"
        resolved_page = "pages/1_📊_Statistical_Tests.py"
        resolved_icon = "⚡"
        resolved_color = "#166534"
        resolved_bg = "#f0fdf4"
        resolved_border = "#16a34a"
        resolved_rationale = "Small sample size ($N < 30$) prevents relying on Central Limit Theorem normality."

    res_card = f"""
    <div style="background-color: {resolved_bg}; border: 2px solid {resolved_border}; border-radius: 8px; padding: 1.1rem 1.35rem; margin-top: 1rem; margin-bottom: 0.8rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
            <span style="font-weight: 800; color: {resolved_color}; font-size: 1.1rem;">
                {resolved_icon} Triage Result: {resolved_pillar}
            </span>
            <span style="font-size: 0.8rem; font-weight: 700; color: {resolved_color}; background: rgba(255,255,255,0.7); padding: 0.2rem 0.55rem; border-radius: 4px; border: 1px solid {resolved_border};">
                Recommended Pathway
            </span>
        </div>
        <div style="font-size: 0.95rem; color: #1e293b; font-weight: 700; margin-bottom: 0.25rem;">
            Primary Method: {resolved_method}
        </div>
        <div style="font-size: 0.85rem; color: #475569; line-height: 1.45;">
            {resolved_rationale}
        </div>
    </div>
    """
    render_html(res_card)

    col_btn, _ = st.columns([1.5, 2.5])
    with col_btn:
        if st.button(f"Open {resolved_pillar} Module 👉", key="btn_triage_jump", use_container_width=True):
            if "clinical risk stratification" in resolved_method.lower():
                st.session_state["epi_nav_mode"] = "direct"
                st.session_state["epi_direct_method_key"] = "risk_stratification"
            st.switch_page(resolved_page)


# =====================================================================
# 2. Classical Statistics Decision Tree & Flowchart
# =====================================================================

def render_statistics_decision_tree() -> None:
    """Renders the comprehensive Classical Statistics decision tree covering

    all 18 parametric and non-parametric hypothesis tests with sample size gates.
    """
    st.markdown("### 📊 Classical Statistics Decision Tree (18 Tests)")
    st.markdown(r"""
    This decision tree guides you from your **Outcome Variable Scale** (Continuous vs Categorical vs Relationship) 
    through **Experimental Group Structure** (1, 2, or 3+ groups, Independent vs Paired) 
    and **Distributional / Sample Scale Gates** ($N < 30$ vs $N \ge 30$) to the optimal test.
    """)

    mermaid_code = """```mermaid
flowchart TD
    DataScale{"Outcome Scale & Objective"}

    %% Continuous Path
    DataScale -->|"Continuous Metric"| Groups{"How Many Groups / Conditions?"}
    
    Groups -->|"1 Sample vs Population Mean"| N_One{"Sample Size Gate"}
    N_One -->|"N >= 30: CLT Normality"| T_One["One-Sample t-test"]
    N_One -->|"N < 30: Skewed / Outliers"| W_One["Wilcoxon Signed-Rank Test"]

    Groups -->|"2 Groups"| Dep2{"Group Relationship"}
    Dep2 -->|"Independent Groups"| N_Two{"Sample Size Gate"}
    N_Two -->|"N >= 30 per group"| T_Welch["Welch's Independent t-test<br/>(Unequal Variance Default)"]
    N_Two -->|"N < 30: Skewed / Outliers"| U_Mann["Mann-Whitney U Test<br/>(Rank-Sum Fallback)"]

    Dep2 -->|"Paired / Repeated Measures"| N_Pair{"Sample Size Gate"}
    N_Pair -->|"N >= 30 pairs"| T_Pair["Paired Samples t-test"]
    N_Pair -->|"N < 30 pairs: Skewed"| W_Pair["Wilcoxon Signed-Rank (Paired)"]

    Groups -->|"3+ Groups"| Dep3{"Group Relationship"}
    Dep3 -->|"Independent Groups"| N_Anova{"Sample Size Gate"}
    N_Anova -->|"N >= 30 per group"| F_Anova["One-Way ANOVA (F-test)"]
    N_Anova -->|"N < 30: Non-Normal"| KW_Test["Kruskal-Wallis H Test"]

    Dep3 -->|"Repeated Measures (Same subjects)"| N_RM{"Sample Size Gate"}
    N_RM -->|"N >= 30: Sphericity Assumed"| RM_Anova["Repeated Measures ANOVA"]
    N_RM -->|"N < 30: Non-Normal"| Fried_Test["Friedman Test"]

    %% Categorical Path
    DataScale -->|"Categorical / Counts"| CatGroups{"Contingency Architecture"}
    CatGroups -->|"Independent Groups (2x2 or RxC)"| CellCounts{"Expected Cell Counts"}
    CellCounts -->|"All Expected >= 5"| Chi2["Pearson Chi-Square Test"]
    CellCounts -->|"Any Expected < 5"| Fisher["Fisher's Exact Test"]
    CatGroups -->|"Paired Binary Proportions"| McNemar["McNemar's Test (Discordant Pairs)"]

    %% Association Path
    DataScale -->|"Continuous Association"| RelType{"Linear vs Monotonic"}
    RelType -->|"Linear & Normal (N >= 30)"| Pearson["Pearson Correlation (r)"]
    RelType -->|"Monotonic / Ordinal / Skewed"| Spearman["Spearman Rank Correlation (rho)"]

    classDef parametric fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef nonparam fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class T_One,T_Welch,T_Pair,F_Anova,RM_Anova,Chi2,Pearson parametric;
    class W_One,U_Mann,W_Pair,KW_Test,Fried_Test,Fisher,McNemar,Spearman nonparam;
    class DataScale,Groups,N_One,Dep2,N_Two,N_Pair,Dep3,N_Anova,N_RM,CatGroups,CellCounts,RelType decision;
```"""
    st.markdown(mermaid_code)

    # CLT & Assumption Triage Box
    with st.expander("💡 Central Limit Theorem & Non-Parametric Triage Rules", expanded=False):
        clt_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #1d4ed8; margin-bottom: 0.35rem;">
                ⚖️ The $N = 30$ Heuristic & Why It Matters
            </div>
            • <strong>When $N \ge 30$ per group:</strong> Under the <strong>Central Limit Theorem (CLT)</strong>, the sampling distribution of the sample mean asymptotically approaches normality regardless of underlying population shape, provided there are no catastrophic outliers. Parametric tests ($t$-test, ANOVA) remain robust and powerful.<br>
            • <strong>When $N < 30$ per group:</strong> The CLT cannot be relied upon. If the data shows moderate skewness ($|\text{Skew}| > 1.0$) or extreme values beyond $3 \times \text{IQR}$, the test statistic does not follow Student's $t$ or Fisher's $F$ distribution, inflating Type I error rates. <strong>Switch directly to the rank-based non-parametric alternative.</strong><br>
            • <strong>Default to Welch's t-test over Student's:</strong> Student's $t$-test assumes homoscedasticity ($\sigma_1^2 = \sigma_2^2$). In real clinical trials, variances are rarely identical. Welch's $t$-test corrects degrees of freedom via the Satterthwaite equation, maintaining exact 5% error rates without sacrificing power.
        </div>
        """
        render_html(clt_html)

    # Interactive Test Finder
    st.markdown("#### ⚡ Quick Statistical Test Selector")
    s1, s2 = st.columns(2)
    with s1:
        test_objective = st.selectbox(
            "What is your research design?",
            options=[
                "Compare 2 Independent Groups (e.g. Treatment vs Control)",
                "Compare 2 Paired / Matched Measurements (Pre vs Post)",
                "Compare 3+ Independent Groups (Multi-Arm Trial)",
                "Compare 3+ Repeated Measurements (Time points)",
                "Categorical Contingency Table (2x2 or RxC)",
                "Paired Binary Changes (Discordant Pairs)",
                "Bivariate Continuous Association"
            ],
            key="stat_test_obj_select"
        )
    with s2:
        test_normality = st.radio(
            "Sample Size & Normality Status:",
            options=[
                "Normal distribution or N >= 30 (Parametric)",
                "Skewed, outliers, or N < 30 (Non-Parametric Fallback)"
            ],
            key="stat_test_norm_select"
        )

    # Map to test key
    is_param = "Parametric" in test_normality
    stat_mapping = {
        "Compare 2 Independent Groups (e.g. Treatment vs Control)": ("welch_t_test" if is_param else "mann_whitney_u", "Welch's Independent t-test" if is_param else "Mann-Whitney U Test"),
        "Compare 2 Paired / Matched Measurements (Pre vs Post)": ("paired_t_test" if is_param else "wilcoxon_paired", "Paired Samples t-test" if is_param else "Wilcoxon Signed-Rank Test"),
        "Compare 3+ Independent Groups (Multi-Arm Trial)": ("one_way_anova" if is_param else "kruskal_wallis", "One-Way ANOVA (F-test)" if is_param else "Kruskal-Wallis H Test"),
        "Compare 3+ Repeated Measurements (Time points)": ("repeated_measures_anova" if is_param else "friedman_test", "Repeated Measures ANOVA" if is_param else "Friedman Test"),
        "Categorical Contingency Table (2x2 or RxC)": ("chi_square_independence" if is_param else "fisher_exact", "Chi-Square Test of Independence" if is_param else "Fisher's Exact Test"),
        "Paired Binary Changes (Discordant Pairs)": ("mcnemar_test", "McNemar's Test"),
        "Bivariate Continuous Association": ("pearson_correlation" if is_param else "spearman_correlation", "Pearson Correlation (r)" if is_param else "Spearman Rank Correlation (rho)")
    }
    t_key, t_name = stat_mapping.get(test_objective, ("welch_t_test", "Welch's t-test"))

    st.success(f"🎯 **Recommended Method:** `{t_name}`")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {t_name} on Page 1 👉", key=f"btn_jump_stat_{t_key}", use_container_width=True):
            st.session_state["nav_mode"] = "direct"
            st.session_state["direct_test_key"] = t_key
            st.switch_page("pages/1_📊_Statistical_Tests.py")


# =====================================================================
# 3. Observational Epidemiology Decision Tree & Flowchart
# =====================================================================

def render_epidemiology_decision_tree() -> None:
    """Renders the comprehensive Observational Epidemiology decision tree covering

    Cohort, Case-Control, Cross-Sectional, Standardization, Health Inequalities, Screening, and Meta-Analysis.
    """
    st.markdown("### 🏥 Observational Epidemiology & Public Health Decision Tree (18 Methods)")
    st.markdown("""
    Observational studies cannot rely on experimental randomization. This decision tree routes based on 
    **Temporal Direction** (Prospective vs Retrospective vs Cross-Sectional), **Sampling Strategy** (Exposed vs Outcome-First), 
    **Socioeconomic Gradients**, and **Confounding Adjustment** to determine the valid epidemiological effect measure.
    """)

    mermaid_code = """```mermaid
flowchart TD
    EpiStart{"Epidemiological Objective & Study Architecture"}

    %% Prospective Cohort
    EpiStart -->|"Prospective: Exposure First, Follow Forward"| Cohort{"Follow-up Accounting"}
    Cohort -->|"Fixed Cohort: No Loss to Follow-up"| RR["Relative Risk (Risk Ratio, RR)<br/>Cumulative Incidence"]
    Cohort -->|"Variable Follow-up: Person-Time Accrual"| IRR["Incidence Rate Ratio (IRR)<br/>Person-Years Denominator"]
    Cohort -->|"Public Health Burden / Preventable Excess"| AR["Attributable Risk (AR) &<br/>Population Attributable Fraction (PAF)"]

    %% Retrospective Case-Control
    EpiStart -->|"Retrospective: Outcome First, Look Backwards"| CC{"Confounding & Stratification"}
    CC -->|"Unstratified Crude Analysis"| OR["Odds Ratio (OR)<br/>Rare Disease Gate: P < 0.05"]
    CC -->|"Confounder Present: Stratified Tables"| MH["Mantel-Haenszel Adjusted OR<br/>Woolf Test for Effect Modification"]
    CC -->|"Outbreak Investigation (Foodborne Illness)"| Outbreak["Attack Rates & Food-Specific Risk Differences"]

    %% Cross-Sectional & Screening
    EpiStart -->|"Snapshot: Exposure & Outcome Concurrent"| Cross{"Research Question"}
    Cross -->|"Population Disease Burden"| Prev["Prevalence & Prevalence Odds Ratio"]
    Cross -->|"Diagnostic Swab / Rapid Test vs Gold Standard"| Diag["Diagnostic Accuracy<br/>Sens, Spec, PPV, NPV, LR+, LR-, ROC-AUC"]

    %% Clinical Prognosis & Risk Stratification
    EpiStart -->|"Clinical Prognosis & Risk Stratification"| Prog{"Model Purpose"}
    Prog -->|"Bedside Points & Stratification (TRIPOD)"| Strat["Clinical Risk Stratification<br/>Sullivan Points, Calibration & DCA"]

    %% Standardization & Health Inequalities
    EpiStart -->|"Demographic & Socioeconomic Comparison"| PopHealth{"Analytical Task"}
    PopHealth -->|"Age-Specific Standardization across Regions"| DSR["Direct Age-Standardized Rate (DSR) & SMR<br/>European Standard Population"]
    PopHealth -->|"Social Gradient across Ranked IMD Strata"| Ineq["Slope Index (SII) & Relative Index (RII)<br/>Midpoint Ridits & Population-Weighted Slope"]

    %% Evidence Synthesis & Survival
    EpiStart -->|"Clinical Evidence Synthesis / Time-to-Event"| Synth{"Synthesis Type"}
    Synth -->|"Pooling Multiple Trials (PRISMA 2020)"| Meta["Random-Effects Meta-Analysis<br/>DerSimonian-Laird & I-squared"]
    Synth -->|"Time-to-Event Survival Analysis"| KM["Kaplan-Meier Curves & Log-Rank Test<br/>Cox Proportional Hazards"]

    classDef cohort fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef casecontrol fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#991b1b;
    classDef screening fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef riskstrat fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#86198f;
    classDef std fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef ineq fill:#ecfdf5,stroke:#059669,stroke-width:2px,color:#065f46;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class RR,IRR,AR cohort;
    class OR,MH,Outbreak casecontrol;
    class Prev,Diag screening;
    class Strat riskstrat;
    class DSR,Meta,KM std;
    class Ineq ineq;
    class EpiStart,Cohort,CC,Cross,Prog,PopHealth,Synth decision;
```"""
    st.markdown(mermaid_code)

    # Bias & Validity Triage Box
    with st.expander("🛡️ Core Epidemiological Validity & Bias Rules", expanded=False):
        epi_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #dc2626; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #b91c1c; margin-bottom: 0.35rem;">
                🚨 Methodological Traps in Observational Research
            </div>
            • <strong>Never Compute Relative Risk from Case-Control Data:</strong> In a case-control design, the investigator arbitrarily fixes the ratio of cases to controls (e.g. 1:2). As a result, the sample prevalence is artificial and does not reflect true population incidence. <strong>You must calculate the Odds Ratio (OR).</strong><br>
            • <strong>The Rare Disease Assumption ($P < 0.05$):</strong> When the outcome occurs in fewer than 5% of the exposed and unexposed populations, $1 - P \approx 1$, meaning $\text{Odds Ratio} \approx \text{Relative Risk}$. When the disease is common ($P > 10\%$), the Odds Ratio substantially exaggerates the Relative Risk!<br>
            • <strong>Reversal Paradox in Demography:</strong> Comparing crude death rates between an older rural district (e.g. Derbyshire Dales) and a younger urban center (Derby City) falsely makes the rural area appear less healthy. Direct Age-Standardization (DSR) removes confounding by age structure.
        </div>
        """
        render_html(epi_html)

    # Interactive Epi Method Finder
    st.markdown("#### ⚡ Quick Epidemiological Method Selector")
    e1, e2 = st.columns(2)
    with e1:
        epi_design = st.selectbox(
            "What is your observational study design?",
            options=[
                "Prospective Cohort (Exposed vs Unexposed followed over time)",
                "Retrospective Case-Control (Cases matched to Controls)",
                "Cross-Sectional Diagnostic Accuracy Study",
                "Clinical Prognosis & Risk Stratification (TRIPOD Standards)",
                "Regional Demographic Rate Standardization",
                "Socioeconomic Health Inequality Surveillance (e.g. IMD Deciles)",
                "Systematic Review / Evidence Synthesis"
            ],
            key="epi_tree_design_select"
        )
    with e2:
        epi_objective = st.selectbox(
            "What is your target metric / task?",
            options=[
                "Compute Relative Risk / Cumulative Incidence",
                "Compute Incidence Rate Ratio (Person-Years)",
                "Compute Odds Ratio with Rare Disease Check",
                "Adjust for Confounder via Mantel-Haenszel OR",
                "Diagnostic Sensitivity, Specificity & ROC-AUC",
                "Formulate Clinical Risk Score & Tiers (Sullivan Points, Calibration, DCA)",
                "Direct Age-Standardized Rate (DSR) with Derbyshire Map",
                "Slope Index (SII) & Relative Index (RII) across IMD Deciles",
                "DerSimonian-Laird Random-Effects Meta-Analysis"
            ],
            key="epi_tree_obj_select"
        )

    # Map to epi key
    epi_routing = {
        "Compute Relative Risk / Cumulative Incidence": ("relative_risk", "Relative Risk (Risk Ratio)"),
        "Compute Incidence Rate Ratio (Person-Years)": ("incidence_rate_ratio", "Incidence Rate Ratio (Person-Time)"),
        "Compute Odds Ratio with Rare Disease Check": ("odds_ratio", "Odds Ratio (Case-Control)"),
        "Adjust for Confounder via Mantel-Haenszel OR": ("mantel_haenszel", "Mantel-Haenszel Stratified Odds Ratio"),
        "Diagnostic Sensitivity, Specificity & ROC-AUC": ("diagnostic_accuracy", "Diagnostic Accuracy (Sens/Spec/ROC)"),
        "Formulate Clinical Risk Score & Tiers (Sullivan Points, Calibration, DCA)": ("risk_stratification", "Clinical Risk Stratification & Prognostic Models"),
        "Direct Age-Standardized Rate (DSR) with Derbyshire Map": ("standardization", "Direct Standardization (DSR & Map)"),
        "Slope Index (SII) & Relative Index (RII) across IMD Deciles": ("health_inequalities", "Health Inequalities (SII & RII)"),
        "DerSimonian-Laird Random-Effects Meta-Analysis": ("meta_analysis", "Random-Effects Meta-Analysis")
    }
    ekey, ename = epi_routing.get(epi_objective, ("relative_risk", "Relative Risk"))

    st.success(f"🎯 **Recommended Method:** `{ename}`")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {ename} on Page 2 👉", key=f"btn_jump_epi_{ekey}", use_container_width=True):
            st.session_state["epi_nav_mode"] = "direct"
            st.session_state["epi_direct_method_key"] = ekey
            st.switch_page("pages/2_🏥_Epidemiology_Methods.py")


# =====================================================================
# 4. Machine Learning & Predictive Modeling Decision Tree
# =====================================================================

def render_machine_learning_decision_tree() -> None:
    """Renders the Machine Learning decision tree covering Supervised, Rare Event,

    Equity Auditing, Time Series Forecasting, and Unsupervised Clustering with Validation.
    """
    st.markdown("### 🤖 Machine Learning & Operational Analytics Decision Tree (16 Architectures)")
    st.markdown(r"""
    Machine learning solves problems where classical parametric equations struggle: high-dimensional collinear predictors,
    non-linear interactions, severe class imbalance ($< 10\%$), sequential forecasting, and exploratory clustering.
    """)

    mermaid_code = """```mermaid
flowchart TD
    MLStart{"Machine Learning Paradigm"}

    %% Supervised Regression
    MLStart -->|"Supervised: Continuous Outcome Y"| RegGoal{"Model Interpretability vs Non-Linearity"}
    RegGoal -->|"High Interpretability / Linear Baseline"| Ridge["Ridge / Lasso / Elastic Net Regularized Regression"]
    RegGoal -->|"Complex Non-Linear Interactions"| RF_Reg["Random Forest Regressor / Gradient Boosting"]

    %% Supervised Classification
    MLStart -->|"Supervised: Binary / Multi-Class Y"| ImbalGate{"Class Distribution & Target Prevalence"}
    ImbalGate -->|"Balanced (Positive Class >= 20%)"| LogReg["Logistic Regression / Standard Random Forest<br/>Metric: ROC-AUC"]
    ImbalGate -->|"Severe Imbalance (Positive Class < 10%)<br/>e.g. Sepsis, Fraud, Readmissions"| RareRF["Cost-Sensitive Random Forest / LightGBM<br/>Metrics: PR-AUC, Brier Score, Youden J"]

    %% Operational Extensions
    MLStart -->|"Protected Demographic Bias Audit"| Fairness["Algorithmic Fairness & Disparate Impact<br/>Rule: Four-Fifths (80%) Threshold"]
    MLStart -->|"Sequential Chronological Demand"| TimeSeries["Lagged Ensemble Time Series Forecaster<br/>Validation: Rolling-Origin Cross-Validation"]

    %% Unsupervised Discovery
    MLStart -->|"Unsupervised: No Target Labels"| UnsupGoal{"Discovery Objective"}
    UnsupGoal -->|"Feature Compression / Collinearity"| PCA["Principal Component Analysis (PCA)<br/>Scree Plot / Explained Variance"]
    UnsupGoal -->|"Sub-Phenotyping / Patient Segmentation"| Cluster["K-Means / DBSCAN Clustering"]

    %% Cluster Stability Gate
    Cluster --> ClusterGate{"Unsupervised Cluster Validation Gate"}
    ClusterGate -->|"Partition Cohesion & Separation"| Silh["Silhouette Score: s in [-1, +1]<br/>Threshold: s > 0.50 strong structure"]
    ClusterGate -->|"Cluster Scatter-to-Separation Ratio"| DBI["Davies-Bouldin Index (DBI >= 0)<br/>Lower is Better: DBI < 1.0 compact"]
    ClusterGate -->|"Variance Ratio Criterion"| CHI["Calinski-Harabasz Index (CHI)<br/>Between-cluster vs Within-cluster variance"]

    classDef supervised fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef rare fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#991b1b;
    classDef cluster fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef eval fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class Ridge,RF_Reg,LogReg,TimeSeries supervised;
    class RareRF,Fairness rare;
    class PCA,Cluster cluster;
    class Silh,DBI,CHI eval;
    class MLStart,RegGoal,ImbalGate,UnsupGoal,ClusterGate decision;
```"""
    st.markdown(mermaid_code)

    # ML Validation Guidance Box
    with st.expander("🔬 Unsupervised Cluster Validation & Stability Heuristics", expanded=False):
        ml_guide_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #9333ea; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #6b21a8; margin-bottom: 0.35rem;">
                🧩 How to Rigorously Validate Unsupervised Clustering
            </div>
            Because unsupervised discovery lacks ground-truth labels, classical $p$-values and classification accuracy do not exist. To prevent clustering random noise into false patient subtypes, use these 3 mathematical criteria:<br><br>
            • <strong>1. Silhouette Score ($s \in [-1, +1]$):</strong> Measures how similar an observation is to its own cluster compared to other clusters. $s > 0.50$ indicates strong, genuine clusters; $s < 0.25$ indicates artificial clustering on continuous uniform noise.<br>
            • <strong>2. Davies-Bouldin Index ($\text{DBI} \ge 0$):</strong> Evaluates the ratio of within-cluster dispersion to between-cluster separation. Lower values indicate better clustering; $\text{DBI} < 1.0$ is generally considered a solid, compact partition.<br>
            • <strong>3. Calinski-Harabasz Index ($\text{CHI} \ge 0$):</strong> The Variance Ratio Criterion, calculating the ratio of between-cluster variance to within-cluster variance multiplied by degrees of freedom. Higher values indicate denser, better-separated clusters.<br>
            • <strong>Sample Scale Heuristic:</strong> Always require at least $N \ge 20\text{--}50$ samples per expected cluster ($N \ge 20k$) and $N \ge 10 \times p$ observations per continuous feature.
        </div>
        """
        render_html(ml_guide_html)

    # Interactive ML Selector
    st.markdown("#### ⚡ Quick Machine Learning Architecture Selector")
    m1, m2 = st.columns(2)
    with m1:
        ml_target_type = st.selectbox(
            "What is your modeling problem?",
            options=[
                "Predict continuous clinical or operational target (Regression)",
                "Predict rare binary event with severe imbalance (< 10%)",
                "Predict standard binary event (Balanced ~50%)",
                "Bedside integer score / Clinical risk stratification (TRIPOD)",
                "Audit demographic fairness and disparate impact",
                "Forecast chronological sequential time series demand",
                "Discover latent patient clusters / sub-phenotypes (Unsupervised)"
            ],
            key="ml_tree_target_select"
        )
    with m2:
        ml_priority = st.selectbox(
            "What is your primary architectural priority?",
            options=[
                "Maximum predictive accuracy and non-linear interactions",
                "Interpretability and odds-ratio coefficients",
                "Precision-Recall calibration for rare events (PR-AUC / Brier)",
                "Cluster stability and mathematical validation (Silhouette/DBI/CHI)",
                "Regulatory audit compliance (Four-Fifths Rule)"
            ],
            key="ml_tree_prio_select"
        )

    # Mapping
    ml_routing = {
        "Predict continuous clinical or operational target (Regression)": ("elastic_net_regression", "Elastic Net Regularized Regression"),
        "Predict rare binary event with severe imbalance (< 10%)": ("cost_sensitive_rf", "Cost-Sensitive Random Forest (PR-AUC)"),
        "Predict standard binary event (Balanced ~50%)": ("logistic_regression", "Binary Logistic Regression (Calibrated Probabilities)"),
        "Bedside integer score / Clinical risk stratification (TRIPOD)": ("risk_stratification", "Clinical Risk Stratification (Bedside Sullivan Points)"),
        "Audit demographic fairness and disparate impact": ("fairness_audit", "Algorithmic Fairness & 80% Disparate Impact Audit"),
        "Forecast chronological sequential time series demand": ("time_series_forecaster", "Lagged Ensemble Time Series Forecaster"),
        "Discover latent patient clusters / sub-phenotypes (Unsupervised)": ("kmeans_clustering", "K-Means Clustering with 3-Metric Validation")
    }
    mkey, mname = ml_routing.get(ml_target_type, ("kmeans_clustering", "K-Means Clustering"))

    st.success(f"🎯 **Recommended Architecture:** `{mname}`")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        dest_label = "Page 7 👉" if mkey == "time_series_forecaster" else ("Page 2 👉" if mkey == "risk_stratification" else "Page 3 👉")
        if st.button(f"Launch {mname} on {dest_label}", key=f"btn_jump_ml_{mkey}", use_container_width=True):
            if mkey == "time_series_forecaster":
                st.switch_page("pages/7_⏱️_Time_Series_and_Forecasting.py")
            elif mkey == "risk_stratification":
                st.session_state["epi_nav_mode"] = "direct"
                st.session_state["epi_direct_method_key"] = "risk_stratification"
                st.switch_page("pages/2_🏥_Epidemiology_Methods.py")
            else:
                st.session_state["ml_direct_model_key"] = mkey
                st.switch_page("pages/3_🤖_Machine_Learning.py")


# =====================================================================
# 5. Biostatistical Sample Size & Power Planning Decision Tree
# =====================================================================

def render_sample_size_decision_tree() -> None:
    """Renders the Sample Size and Statistical Power planning decision tree covering

    10 analytical frameworks, inflation cushions, and cluster randomized designs.
    """
    st.markdown("### 📐 Biostatistical Sample Size & Power Planning Tree (10 Frameworks)")
    st.markdown("""
    Under ICH-E9 and research ethics governance, conducting an underpowered trial is unethical because it exposes participants
    to experimental burdens without sufficient probability of detecting a real effect. This decision tree routes you to the exact
    closed-form sample size formula based on your primary trial endpoint and study design.
    """)

    mermaid_code = """```mermaid
flowchart TD
    SSStart{"Primary Trial Endpoint Scale"}

    %% Continuous Path
    SSStart -->|"Continuous Outcome (Means)"| MeanDesign{"Group Design"}
    MeanDesign -->|"Two Independent Arms (Parallel RCT)"| SS_TwoMeans["Two Independent Means (t-test)<br/>Formula: 2 * (Z_alpha + Z_beta)^2 / d^2"]
    MeanDesign -->|"Paired Before-and-After (Same Subjects)"| SS_Paired["Paired Means Difference<br/>Formula: (Z_alpha + Z_beta)^2 / d_z^2"]
    MeanDesign -->|"Multi-Arm Trial (3+ Groups)"| SS_ANOVA["One-Way ANOVA (F-test)<br/>Cohen's f Effect Size"]

    %% Binary Path
    SSStart -->|"Binary Outcome (Proportions & Risk)"| PropDesign{"Study Architecture"}
    PropDesign -->|"Two Independent Proportions (RCT)"| SS_Prop["Two Independent Proportions<br/>Fleiss Continuity Correction"]
    PropDesign -->|"Prospective Cohort (Risk Ratio)"| SS_RR["Cohort Relative Risk Sizing<br/>Kelsey / Fleiss Equations"]
    PropDesign -->|"Case-Control Study (Odds Ratio)"| SS_OR["Case-Control Odds Ratio Sizing<br/>1:k Control Matching Ratio"]

    %% Survival & Clustering
    SSStart -->|"Time-to-Event Survival Outcome"| SS_LogRank["Schoenfeld Log-Rank Sizing<br/>Total Required Events E"]
    SSStart -->|"Unsupervised Phenotyping Discovery"| SS_Clust["Cluster Stability Heuristics<br/>N >= 20k samples, N >= 10p per feature"]

    %% Protocol Inflation Cushions
    SS_TwoMeans & SS_Paired & SS_Prop & SS_RR & SS_OR --> CushGate{"Trial Protocol Inflation Factors"}
    CushGate -->|"Participant Attrition / Dropouts"| SS_Loss["Attrition Cushion Adjustment<br/>N_adj = N / (1 - Loss_Rate)"]
    CushGate -->|"Cluster Randomization (GP practices, wards)"| SS_Cluster["Cluster RCT Variance Inflation<br/>VIF = 1 + (m - 1) * ICC"]

    classDef continuous fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef binary fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef survival fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef cushion fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class SS_TwoMeans,SS_Paired,SS_ANOVA continuous;
    class SS_Prop,SS_RR,SS_OR binary;
    class SS_LogRank,SS_Clust survival;
    class SS_Loss,SS_Cluster cushion;
    class SSStart,MeanDesign,PropDesign,CushGate decision;
```"""
    st.markdown(mermaid_code)

    # Power Rules Box
    with st.expander("📋 Protocol Sizing Governance & Regulatory Checklist", expanded=False):
        power_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #ea580c; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #9a3412; margin-bottom: 0.35rem;">
                📐 Key Biostatistical Sizing Rules for Grant Protocols
            </div>
            • <strong>Two-Tailed Significance ($\alpha = 0.05$):</strong> Sets the maximum risk of declaring a false discovery (Type I error) to $5\%$, giving critical value $Z_{1-\alpha/2} = 1.960$.<br>
            • <strong>Statistical Power ($1 - \beta \ge 80\%$ or $90\%$):</strong> Pre-specifies an $80\%$ ($Z_{1-\beta} = 0.842$) or $90\%$ ($Z_{1-\beta} = 1.282$) probability of correctly detecting a true difference of magnitude $\delta$.<br>
            • <strong>Minimal Clinically Important Difference (MCID):</strong> Never calculate sample size on whatever difference happens to be convenient. You must base $\delta$ on the smallest clinical difference that would alter medical practice or patient outcomes.<br>
            • <strong>Cluster Randomization Penalty ($\text{VIF}$):</strong> When clinics, schools, or GP surgeries are randomized rather than individuals, outcomes within the same cluster correlate (Intraclass Correlation $\rho > 0$). The required sample size must be multiplied by $\text{VIF} = 1 + (m - 1)\rho$, where $m$ is cluster size.
        </div>
        """
        render_html(power_rules_html)

    # Interactive Sizing Selector
    st.markdown("#### ⚡ Quick Sample Size Framework Selector")
    p1, p2 = st.columns(2)
    with p1:
        ss_endpoint = st.selectbox(
            "What is your primary endpoint / outcome type?",
            options=[
                "Continuous Means (e.g. Blood pressure reduction, HbA1c change)",
                "Binary Proportions (e.g. Cure rate %, Remission %)",
                "Relative Risk (Prospective Cohort incidence)",
                "Odds Ratio (Case-Control with 1:k matching)",
                "Survival Time-to-Event (Hazard Ratio & Censored Deaths)",
                "Unsupervised Machine Learning Cluster Stability"
            ],
            key="ss_tree_endpoint_select"
        )
    with p2:
        ss_design = st.selectbox(
            "What is your trial / study structure?",
            options=[
                "Two Independent Parallel Groups (1:1 Allocation)",
                "Paired Pre-Post Design (Within-Subject)",
                "Unbalanced Group Allocation (e.g. 1:2 or 1:3 ratio)",
                "Cluster-Randomized Trial (Community Clinics / Schools)",
                "Longitudinal Cohort with Projected Attrition"
            ],
            key="ss_tree_design_select"
        )

    # Sizing routing
    ss_routing = {
        "Continuous Means (e.g. Blood pressure reduction, HbA1c change)": ("two_means", "Continuous: Two Independent Means (t-test)"),
        "Binary Proportions (e.g. Cure rate %, Remission %)": ("two_proportions", "Binary: Two Independent Proportions (Z-test)"),
        "Relative Risk (Prospective Cohort incidence)": ("cohort_relative_risk", "Epidemiology: Cohort Relative Risk Power"),
        "Odds Ratio (Case-Control with 1:k matching)": ("case_control_odds_ratio", "Epidemiology: Case-Control Odds Ratio Power"),
        "Survival Time-to-Event (Hazard Ratio & Censored Deaths)": ("survival_logrank", "Survival: Schoenfeld Log-Rank Events"),
        "Unsupervised Machine Learning Cluster Stability": ("kmeans_clustering", "Machine Learning: Unsupervised Cluster Stability")
    }
    skey, sname = ss_routing.get(ss_endpoint, ("two_means", "Two Independent Means"))

    st.success(f"🎯 **Recommended Sizing Framework:** `{sname}`")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if skey == "kmeans_clustering":
            if st.button(f"Inspect Cluster Stability Metrics on Page 3 👉", key="btn_jump_ss_ml", use_container_width=True):
                st.session_state["ml_direct_model_key"] = "kmeans_clustering"
                st.switch_page("pages/3_🤖_Machine_Learning.py")
        else:
            if st.button(f"Launch {sname} on Page 4 👉", key=f"btn_jump_ss_{skey}", use_container_width=True):
                st.session_state["sample_size_direct_framework_key"] = skey
                st.switch_page("pages/4_📐_Sample_Size_and_Power.py")


# =====================================================================
# 6. Qualitative Analysis & Natural Language Processing (NLP) Decision Tree
# =====================================================================

def render_nlp_decision_tree() -> None:
    """Renders the comprehensive Qualitative Analysis and Natural Language Processing (NLP)

    decision tree covering all 7 methods across 5 progressive evolutionary levels.
    """
    st.markdown("### 🗣️ Qualitative Analysis & Natural Language Processing Decision Tree (7 Methods)")
    st.markdown(r"""
    This decision tree guides you from your **Qualitative Text Analysis Objective** 
    (Sentiment, Term Weighting, Supervised Triage, Unsupervised Themes, Entity Tagging, or Factual Q&A)
    through **Annotation Availability** (Labeled vs. Cold-Start / Zero-Shot) 
    and **Computational / Governance Constraints** (CPU Rule-Based vs. Statistical ML vs. Deep Transformers & Foundation LLMs) 
    to the mathematically and architecturally justified method.
    """)

    mermaid_code = """```mermaid
flowchart TD
    NLPStart(["Qualitative Research Question or Text Dataset"]) --> TaskType{"What is the Primary Qualitative / Text Goal?"}

    %% 1. Sentiment & Negation
    TaskType -->|"Sentiment & Polarity Scoring"| NegScope{"Does the Text Contain Clinical Negation?"}
    NegScope -->|"Yes: 'denies pain', 'no fever'"| M1["1. Rule-Based VADER + Clinical NegEx Scope Inversion<br/>(Prefix/Suffix Scope Bounding, [-1, +1] Valence)"]
    NegScope -->|"No: Standard Review Polarity"| M1_std["1. Standard Lexicon Sentiment Analysis<br/>(Compound Score with Word Weights)"]

    %% 2. Keyword & Importance
    TaskType -->|"Keyword Weighting & Lexical Similarity"| M2["2. TF-IDF Vectorizer & Cosine Similarity<br/>(Term Frequency-Inverse Document Frequency, Dot Product)"]

    %% 3. Supervised Classification
    TaskType -->|"Document / Incident Triage & Categorization"| LabelCheck{"Are Annotated Training Labels Available?"}
    LabelCheck -->|"Yes: Labeled Data Available (N >= 500)"| M3["3. Supervised Multinomial Naive Bayes & Linear SVM<br/>(Laplace Smoothing, Log-Odds Hyperplane)"]
    LabelCheck -->|"No: Zero-Shot Cold-Start"| M7_zs["7. Foundation LLM Zero-Shot Triage<br/>(NLI Hypothesis Entailment e.g. BART-MNLI)"]

    %% 4. Unsupervised Themes
    TaskType -->|"Unsupervised Thematic & Concept Discovery"| DiscoveryType{"Discovery Paradigm"}
    DiscoveryType -->|"Probabilistic Topic Distributions"| M4["4. Latent Dirichlet Allocation (LDA)<br/>(Dirichlet Priors alpha/beta, NPMI Coherence C_v)"]
    DiscoveryType -->|"Continuous Geometric Space"| M5["5. Dense Semantic Vector Embeddings<br/>(Sentence-Transformers all-MiniLM-L6-v2, 384D)"]

    %% 5. Information Extraction & Q&A
    TaskType -->|"Entity Extraction & Factual Question Answering"| GenType{"Extraction vs Generative Synthesis"}
    GenType -->|"Clinical Entities (Conditions, Drugs, Tests)"| M6["6. Clinical Named Entity Recognition (NER)<br/>(IOB Span Tagging, Exact Offsets, Micro F1)"]
    GenType -->|"Contextual Synthesis & Policy Q&A"| M7_rag["7. Retrieval-Augmented Generation (RAG)<br/>(Context Injection, LLM Prompt, Faithfulness Guardrail)"]

    classDef level1 fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef level2 fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef level3 fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef level4 fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef level5 fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#9f1239;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class M1,M1_std level1;
    class M2,M3 level2;
    class M4 level3;
    class M5 level4;
    class M6,M7_zs,M7_rag level5;
    class NLPStart,TaskType,NegScope,LabelCheck,DiscoveryType,GenType decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 Qualitative NLP Triage Principles: Rule-Based vs. Statistical vs. Deep LLM RAG", expanded=False):
        nlp_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #0284c7; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #0369a1; margin-bottom: 0.35rem;">
                🗣️ Architectural Decision Rules for Qualitative Text Analytics
            </div>
            • <strong>Clinical Negation Scope Inversion:</strong> Standard sentiment lexicons fail in medical text because "no chest pain" or "denies shortness of breath" contains positive clinical symptoms that naive models score as high morbidity. Always precede sentiment scoring with NegEx regex scope detection to invert valence within a 6-token window.<br>
            • <strong>Exact Lexical Matching vs. Dense Semantic Embeddings:</strong> TF-IDF is superior when exact terminology, part numbers, or drug codes matter. Dense embeddings (Sentence-Transformers) are required when synonymous phrasing must be matched without shared words (e.g., "sluggish UI" matching "frozen application").<br>
            • <strong>Supervised Statistical Classifiers vs. Zero-Shot LLMs:</strong> When $\ge 500$ verified historical labels exist, Multinomial Naive Bayes and Linear SVM execute in microseconds with 100% deterministic reproducibility on basic CPUs. When encountering a cold-start classification task with zero training labels, utilize Foundation LLMs via NLI Zero-Shot inference.<br>
            • <strong>Automated Faithfulness Audits for RAG:</strong> Never deploy standalone generative LLMs on clinical protocols or enterprise SOPs without retrieval grounding. Always compute Faithfulness ($\text{Verified Claims} / \text{Total Claims} \ge 85\%$) to detect hallucinations before surfacing outputs.
        </div>
        """
        render_html(nlp_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick Qualitative NLP Method Selector")
    n1, n2 = st.columns(2)
    with n1:
        nlp_objective = st.selectbox(
            "What is your primary qualitative text analysis goal?",
            options=[
                "Extract clinical sentiment & detect negation (e.g. 'no chest pain')",
                "Measure keyword importance & document similarity across text records",
                "Classify tickets/reports with labeled training examples (Supervised ML)",
                "Discover latent topics / themes across unlabelled text (Topic Modeling)",
                "Semantic search & concept matching without exact word overlap",
                "Extract clinical entities (diseases, drugs, procedures) & IOB spans",
                "Cold-start zero-shot classification or contextual Q&A with grounded evidence (RAG)"
            ],
            key="nlp_tree_obj_select"
        )
    with n2:
        nlp_compute = st.selectbox(
            "Computational resources & deployment environment:",
            options=[
                "Lightweight / CPU / Zero-training rule-based",
                "Standard statistical NLP (Scikit-Learn / Gensim)",
                "Deep Transformer / Foundation Model (Sentence-Transformers / Hugging Face)"
            ],
            key="nlp_tree_compute_select"
        )

    # NLP Routing
    nlp_routing = {
        "Extract clinical sentiment & detect negation (e.g. 'no chest pain')": (
            "vader_negex_sentiment",
            "Rule-Based Sentiment Analysis & Clinical NegEx Scope",
            "Level 1: Rule-Based & Negation Resolution"
        ),
        "Measure keyword importance & document similarity across text records": (
            "tfidf_vectorizer_similarity",
            "TF-IDF Vectorization & Pairwise Cosine Similarity",
            "Level 2: Statistical Vectorization"
        ),
        "Classify tickets/reports with labeled training examples (Supervised ML)": (
            "naive_bayes_svm_classifier",
            "Supervised Text Classification (Naive Bayes & Linear SVM)",
            "Level 2: Supervised Text Classification"
        ),
        "Discover latent topics / themes across unlabelled text (Topic Modeling)": (
            "lda_topic_modeling",
            "Latent Dirichlet Allocation (LDA) Topic Modeling",
            "Level 3: Thematic Discovery & Topic Modeling"
        ),
        "Semantic search & concept matching without exact word overlap": (
            "dense_embeddings_similarity",
            "Dense Semantic Embeddings & Vector Search",
            "Level 4: Dense Continuous Embeddings"
        ),
        "Extract clinical entities (diseases, drugs, procedures) & IOB spans": (
            "clinical_ner_extraction",
            "Clinical Named Entity Recognition (NER) & Span Tagging",
            "Level 5: State-of-the-Art Generative AI & NER"
        ),
        "Cold-start zero-shot classification or contextual Q&A with grounded evidence (RAG)": (
            "llm_zeroshot_rag",
            "LLM Zero-Shot Classification & Retrieval-Augmented Generation (RAG)",
            "Level 5: State-of-the-Art Generative AI & RAG"
        )
    }

    nkey, nname, ntier = nlp_routing.get(nlp_objective, ("vader_negex_sentiment", "Sentiment & NegEx", "Level 1"))

    st.success(f"🎯 **Recommended Method:** `{nname}` — *{ntier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {nname} on Page 6 👉", key=f"btn_jump_nlp_{nkey}", use_container_width=True):
            st.session_state["active_nlp_method"] = nkey
            st.switch_page("pages/6_🗣️_Natural_Language_Processing.py")



# =====================================================================
# 8. Causal Inference & Policy Evaluation Decision Tree
# =====================================================================

def render_causal_inference_decision_tree() -> None:
    """Renders the comprehensive Causal Inference and Quasi-Experimental Policy Evaluation
    decision tree covering all 5 architectures.
    """
    st.markdown("### 🔍 Causal Inference & Quasi-Experimental Policy Evaluation Decision Tree (5 Architectures)")
    st.markdown(r"""
    This decision tree guides you from your **Observational Study Architecture & Policy Assignment Rule** 
    (Numerical Cutoffs, Temporal Guideline Shocks, Clinical Confounding by Indication, or Omitted Variables) 
    through **Identifying Assumptions & Falsification Audits** 
    (Parallel Pre-Trends, Durbin-Watson Autocorrelation, Covariate Balance SMDs, McCrary Density Continuity, and Instrument Relevance $F > 10$) 
    to the mathematically and econometrically justified causal estimator.
    """)

    mermaid_code = """```mermaid
flowchart TD
    CausalStart(["Observational Policy / Clinical Treatment Question"]) --> AssignMech{"What is the Treatment / Policy Assignment Rule?"}

    %% 1. Regression Discontinuity
    AssignMech -->|"Strict Numerical Triage Cutoff (e.g. Risk >= 10%)"| CompCheck{"Is Compliance at the Threshold Perfect or Partial?"}
    CompCheck -->|"Strict: 100% Deterministic Prescribing"| M1["1. Sharp Regression Discontinuity Design (Sharp RDD)<br/>• Local Linear OLS within MSE-optimal bandwidth<br/>• Jump at Cutoff = Unconfounded Intent-to-Treat"]
    CompCheck -->|"Partial / Fuzzy: Probability Jumps"| M1_fuzzy["1. Fuzzy Regression Discontinuity (Fuzzy RDD)<br/>• Instrumented via Threshold Indicator (2SLS)<br/>• Local Average Treatment Effect (LATE) for Compliers"]

    %% 2. Temporal Policy Shock
    AssignMech -->|"Temporal Policy Shock or National Guideline"| CtrlCheck{"Are Unexposed Comparison Groups Observed Over Time?"}
    CtrlCheck -->|"No: Single Longitudinal Population Series"| M2["2. Interrupted Time Series (ITS)<br/>• Segmented OLS (Level Shift + Slope Trajectory Change)<br/>• Durbin-Watson Autocorrelation & Newey-West HAC SEs"]
    CtrlCheck -->|"Yes: Treated & Control Groups Tracked Across Time"| PreTrend{"Do Pre-Policy Trends Run Parallel?"}
    PreTrend -->|"Yes: Parallel Pre-Trends Confirmed (p > 0.05)"| M3["3. Difference-in-Differences (DiD)<br/>• Two-Way Fixed Effects (TWFE) Panel Regression<br/>• 2x2 Subtraction Matrix & Event Studies"]
    PreTrend -->|"No: Differential Divergence"| M3_syn["3. Synthetic Control / Covariate-Adjusted DiD<br/>• Convex Optimization of Weighted Donor Units"]

    %% 3. Observational Cohort Selection
    AssignMech -->|"Clinician / Patient Discretion (Observational Practice)"| ConfCheck{"Are All Key Confounders Measured at Baseline?"}
    ConfCheck -->|"Yes: Conditional Exchangeability on Observables"| OverlapCheck{"Is There Substantial Propensity Score Overlap?"}
    OverlapCheck -->|"Yes: Common Support Satisfied"| M4["4. Propensity Score Matching (PSM) & IPTW<br/>• Logistic Propensity Scores & SMD Balance Audits<br/>• 1:1 Nearest-Neighbor Caliper Matching<br/>• Inverse Probability of Treatment Weighting (IPTW)"]
    OverlapCheck -->|"No: Heavy Off-Support Imbalance"| M4_trim["4. Trimmed IPTW / Overlap Weighting<br/>• Crump Truncation (0.10 <= e_i <= 0.90)"]

    %% 4. Unmeasured Confounding / IV
    ConfCheck -->|"No: Unmeasured Confounders (Omitted Variable Bias)"| ExogCheck{"Is a Valid Exogenous Instrument (Z) Available?"}
    ExogCheck -->|"Yes: Z affects X but has no direct path to Y"| M5["5. Instrumental Variables (IV) & Mendelian Randomization<br/>• First-Stage F-Statistic > 10 (Relevance)<br/>• Two-Stage Least Squares (2SLS) & Wald Ratio Estimator"]
    ExogCheck -->|"No Valid Instrument Available"| Block["Causal Identification Blocked<br/>• Perform Sensitivity Analysis (Rosenbaum Bounds / E-Value)<br/>• Report as Associational Rather than Causal"]

    classDef rdd fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef its fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef did fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef psm fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef iv fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#9f1239;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;
    classDef blocked fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b;

    class M1,M1_fuzzy rdd;
    class M2 its;
    class M3,M3_syn did;
    class M4,M4_trim psm;
    class M5 iv;
    class Block blocked;
    class CausalStart,AssignMech,CompCheck,CtrlCheck,PreTrend,ConfCheck,OverlapCheck,ExogCheck decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 Quasi-Experimental Triage Principles: Counterfactuals, Common Support & Exogeneity", expanded=False):
        causal_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #059669; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #065f46; margin-bottom: 0.35rem;">
                🔍 Econometric & Epidemiological Rules for Quasi-Experimental Identification
            </div>
            • <strong>Regression Discontinuity (Local Randomization):</strong> When clinical guidelines enforce a strict numerical triage cutoff (e.g. statin prescribing at $\text{QRISK} \ge 10.0\%$), patients just above and below the cutoff differ only by chance. RDD estimates the Local Average Treatment Effect (LATE) for patients on the clinical margin of prescribing.<br>
            • <strong>Interrupted Time Series (Secular Drift vs Shock):</strong> When a national policy shock impacts an entire healthcare population simultaneously with no unexposed comparison group, segmented regression isolates whether the policy caused an immediate intercept level shift ($\beta_2$), a slope trajectory redirection ($\beta_3$), or both, while controlling for autocorrelation via Newey-West HAC.<br>
            • <strong>Difference-in-Differences (Parallel Pre-Trends):</strong> When unexposed comparison hospitals or geographical regions exist alongside treated units, DiD removes both time-invariant baseline differences and common secular drift. The parallel pre-trends test ($p > 0.05$ on pre-reform leads) must hold.<br>
            • <strong>Propensity Score Matching (Unconfoundedness on Observables):</strong> In observational cohorts where clinicians choose treatments, matching or IPTW balances observed baseline confounders ($\text{SMD} < 0.10$), mimicking a randomized trial on observable characteristics. It assumes no hidden unmeasured confounders.<br>
            • <strong>Instrumental Variables & Mendelian Randomization (Omitted Variable Bias):</strong> When unmeasured confounders exist (e.g. lifestyle, frailty), an exogenous instrument $Z$ that influences treatment $X$ without directly affecting outcome $Y$ isolates the unconfounded causal effect via 2SLS. The first-stage $F$-statistic must exceed 10 (Stock-Yogo rule).
        </div>
        """
        render_html(causal_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick Causal Inference Method Selector")
    c1, c2 = st.columns(2)
    with c1:
        causal_scenario = st.selectbox(
            "What is your observational evaluation scenario?",
            options=[
                "Hospital 7-day staffing reform with unexposed comparison trusts tracked over time",
                "National prescribing guideline evaluated on a monthly longitudinal sequence before and after launch",
                "Robotic vs laparoscopic surgical cohort with measured baseline confounders (age, comorbidities)",
                "Statin therapy initiated strictly at numerical QRISK threshold score (>= 10.0%)",
                "Biomarker exposure (CRP) subject to unmeasured lifestyle confounding, with genetic variants available"
            ],
            key="causal_tree_scenario_select"
        )
    with c2:
        causal_setting = st.selectbox(
            "Primary identification requirement & data structure:",
            options=[
                "Panel data across treated & control units (Test parallel pre-trends)",
                "Single population longitudinal time series (Check serial autocorrelation)",
                "Observational cohort with rich baseline covariates (Check common support)",
                "Sharp or fuzzy numerical triage threshold (Check McCrary density continuity)",
                "Exogenous genetic allele or geographic distance instrument (Check F > 10)"
            ],
            key="causal_tree_setting_select"
        )

    # Causal Routing
    causal_routing = {
        "Hospital 7-day staffing reform with unexposed comparison trusts tracked over time": (
            "difference_in_differences",
            "Difference-in-Differences (DiD) & Event Studies",
            "Two-Way Fixed Effects Panel Analysis"
        ),
        "National prescribing guideline evaluated on a monthly longitudinal sequence before and after launch": (
            "interrupted_time_series",
            "Interrupted Time Series (ITS) & Segmented Regression",
            "Single-Population Longitudinal Surveillance"
        ),
        "Robotic vs laparoscopic surgical cohort with measured baseline confounders (age, comorbidities)": (
            "propensity_score_matching",
            "Propensity Score Matching (PSM) & IPTW",
            "Observational Confounder Balancing"
        ),
        "Statin therapy initiated strictly at numerical QRISK threshold score (>= 10.0%)": (
            "regression_discontinuity",
            "Regression Discontinuity Design (Sharp & Fuzzy RDD)",
            "Quasi-Randomized Cutoff Exploitation"
        ),
        "Biomarker exposure (CRP) subject to unmeasured lifestyle confounding, with genetic variants available": (
            "instrumental_variables",
            "Instrumental Variables (IV) & Mendelian Randomization",
            "Exogenous Variation / 2SLS"
        )
    }

    ckey, cname, ctier = causal_routing.get(
        causal_scenario,
        ("difference_in_differences", "Difference-in-Differences (DiD)", "Quasi-Experimental Panel")
    )

    st.success(f"🎯 **Recommended Method:** `{cname}` — *{ctier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {cname} on Page 8 👉", key=f"btn_jump_causal_{ckey}", use_container_width=True):
            st.session_state["causal_direct_method_key"] = ckey
            st.session_state["causal_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.switch_page("pages/8_🔍_Causal_Inference_and_Policy_Evaluation.py")


# =====================================================================
# 8. Statistical Process Control (SPC) & Quality Improvement (QI) Tree
# =====================================================================

def render_spc_qi_decision_tree() -> None:
    """Renders the comprehensive Statistical Process Control and Quality Improvement
    decision tree covering all 5 architectures.
    """
    st.markdown("### 📈 Statistical Process Control (SPC) & Quality Improvement (QI) Decision Tree (5 Architectures)")
    st.markdown(r"""
    This decision tree guides clinical audit teams, healthcare analysts, and QI directors from **Measure Type & Data Scale** 
    (Continuous Measures, Event Counts, Binomial Proportions, or Executive Target Benchmarks) 
    through **Overdispersion Audits & Special Cause Detection Rules** 
    (Laney $\sigma_z$ inflation factor, Shift, Trend, Runs, and Shewhart $3\sigma$ control limits) 
    to the mathematically and operationally justified SPC chart.
    """)

    mermaid_code = """```mermaid
flowchart TD
    SPCStart(["Clinical Quality or Process Metric"]) --> DataType{"What is the Nature of the Quality Measure?"}

    %% 1. Continuous Measures
    DataType -->|"Continuous Variable (Wait Times, Length of Stay, Dosing)"| XmR["2. Shewhart XmR / I-MR Chart<br/>• Individual measures & Moving Ranges (k=2)<br/>• Upper/Lower Limits: X-bar ± 2.66 * MR-bar<br/>• Detects step shifts and moving range spikes"]

    %% 2. Proportions / Rates
    DataType -->|"Defect / Adherence Proportion (Events / Opportunities)"| PropScale{"Are Subgroup Sizes Variable or Very Large (n > 2,000)?"}
    PropScale -->|"Yes / High Denominator: Test Laney sigma_z"| CheckLaneyP{"Is sigma_z > 1.10 (Between-Subgroup Overdispersion)?"}
    CheckLaneyP -->|"Yes: Overdispersion Detected"| LaneyP["3. Laney p'-Chart<br/>• Corrects artificially narrow binomial limits<br/>• SE_Laney = SE_p * sigma_z<br/>• Prevents false alarm special causes"]
    CheckLaneyP -->|"No: Pure Binomial Sampling Noise"| StdP["3. Standard p-Chart<br/>• Classical 3-sigma binomial control limits<br/>• p-bar ± 3 * sqrt(p*(1-p)/n_i)"]
    PropScale -->|"No / Small Constant Subgroups"| StdP

    %% 3. Count Rates per Unit
    DataType -->|"Poisson Count Rate (e.g. Infections per 1,000 Bed-Days)"| RateScale{"Are Denominators Large or Susceptible to Clustering?"}
    RateScale -->|"Yes: Test Laney sigma_z on z-scores"| CheckLaneyU{"Is sigma_z > 1.10 (Overdispersed Poisson)?"}
    CheckLaneyU -->|"Yes: Laney Inflation Required"| LaneyU["4. Laney u'-Chart<br/>• Poisson rate limits inflated by sigma_z<br/>• Avoids tampering on high bed-day wards"]
    CheckLaneyU -->|"No: Homogeneous Poisson"| StdU["4. Standard u-Chart<br/>• Classical Poisson rate 3-sigma limits<br/>• u-bar ± 3 * sqrt(u-bar / n_i)"]
    RateScale -->|"No: Small Equal Exposure Units"| StdU

    %% 4. Initial QI Project / Small Sample
    DataType -->|"Early-Stage QI Project / Small Sequential Count (N < 20)"| RunChart["1. Healthcare Run Chart<br/>• Baseline median of first 10-12 points<br/>• 4 Non-Random Rules: Shift (>=6), Trend (>=5), Runs, Astronomical Point<br/>• Requires no complex distribution assumptions"]

    %% 5. Board Oversight & Targets
    DataType -->|"Executive Board Governance vs Fixed Numerical Standard"| MDC["5. NHS Making Data Count & Capability<br/>• Variation Icon: Special Cause (Improvement/Concern) vs Common Cause<br/>• Assurance Icon: Consistent Pass, Flapping, or Consistent Fail<br/>• Process Capability: Cp and Cpk indices"]

    classDef xmr fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef prop fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef rate fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef rc fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef mdc fill:#ecfdf5,stroke:#047857,stroke-width:2px,color:#065f46;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class XmR xmr;
    class LaneyP,StdP prop;
    class LaneyU,StdU rate;
    class RunChart rc;
    class MDC mdc;
    class SPCStart,DataType,PropScale,CheckLaneyP,RateScale,CheckLaneyU decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 SPC & Quality Improvement Principles: Deming's Variation, Laney Overdispersion & Tampering", expanded=False):
        spc_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #16a34a; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #166534; margin-bottom: 0.35rem;">
                📈 Core Principles for Healthcare Quality Improvement & SPC Selection
            </div>
            • <strong>Common Cause vs Special Cause Variation:</strong> Common cause variation is the normal, predictable noise inherent in any healthcare system. Special cause variation represents an authentic historical shift or external shock. Reacting to common cause noise as if it were a crisis is <em>tampering</em>, which increases system variance.<br>
            • <strong>Healthcare Run Charts (Early Project Prototyping):</strong> When launching a new PDSA cycle with fewer than 20 points, a run chart using a 10-12 point baseline median and 4 mathematical probability rules (Shift $\ge 6$, Trend $\ge 5$, Runs count, Astronomical Point) provides immediate statistical rigour without parametric assumptions.<br>
            • <strong>Shewhart XmR / I-MR Charts (Continuous Individual Metrics):</strong> For continuous indicators where data points arrive one by one (e.g. ED wait time, length of stay), limits are derived from the Average Moving Range ($ar{R}$): $	ext{UCL/LCL} = ar{X} \pm 2.66 	imes ar{R}$.<br>
            • <strong>Laney p' and u' Overdispersion Correction ($\sigma_z$):</strong> In large hospital trusts ($n_i > 2,000$), standard binomial or Poisson control limits become artificially narrow, triggering frequent false alarms. The Laney adjustment calculates the moving range of standardized $z$-scores to scale limits by $\sigma_z$, restoring authentic $3\sigma$ coverage.<br>
            • <strong>NHS Making Data Count (MDC):</strong> Pairs process variation with target capability, reporting twin icons: Variation (Common Cause / Special Cause Improvement / Special Cause Concern) and Assurance (Pass / Flapping / Fail) alongside $C_{pk}$ capability indices.
        </div>
        """
        render_html(spc_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick SPC & Quality Improvement Method Selector")
    c1, c2 = st.columns(2)
    with c1:
        spc_scenario = st.selectbox(
            "What is your quality improvement or audit scenario?",
            options=[
                "Monthly elective surgical site infections during initial chlorhexidine QI bundle rollout (N=24 months)",
                "Emergency department door-to-needle thrombolysis time for individual stroke admissions (minutes)",
                "Hospital-wide 4-hour A&E access breach proportions with very large patient volumes (n > 5,000/month)",
                "Ward catheter-associated urinary tract infection (CAUTI) rates per 1,000 catheter-days with variable occupancy",
                "Elective day-case surgery rate evaluated against 85% national target for trust board assurance"
            ],
            key="spc_tree_scenario_select"
        )
    with c2:
        spc_setting = st.selectbox(
            "Primary analytical requirement & mathematical model:",
            options=[
                "Baseline median with Shift (>=6 pts), Trend (>=5 pts) and Runs count verification",
                "Continuous individual measures with moving range (k=2) and 3-sigma limits",
                "Binomial proportion with Laney sigma_z overdispersion expansion for large subgroups",
                "Poisson infection count per 1,000 exposure units with Laney overdispersion adjustment",
                "Executive dual-icon system (Variation + Target Assurance) with Cp and Cpk capability"
            ],
            key="spc_tree_setting_select"
        )

    spc_routing = {
        "Monthly elective surgical site infections during initial chlorhexidine QI bundle rollout (N=24 months)": (
            "run_chart",
            "Healthcare Run Chart",
            "Non-Parametric Median & 4 Detection Rules"
        ),
        "Emergency department door-to-needle thrombolysis time for individual stroke admissions (minutes)": (
            "xmr_chart",
            "Shewhart XmR / I-MR Chart",
            "Individual Continuous Measures & Moving Range Limits"
        ),
        "Hospital-wide 4-hour A&E access breach proportions with very large patient volumes (n > 5,000/month)": (
            "p_chart",
            "Laney p' & Standard p-Chart",
            "Binomial Proportions with Overdispersion sigma_z Correction"
        ),
        "Ward catheter-associated urinary tract infection (CAUTI) rates per 1,000 catheter-days with variable occupancy": (
            "u_chart",
            "Laney u' & Standard u-Chart",
            "Poisson Infection Rates per 1,000 Bed-Days"
        ),
        "Elective day-case surgery rate evaluated against 85% national target for trust board assurance": (
            "making_data_count",
            "NHS Making Data Count & Capability Analysis",
            "System Icons (Variation/Assurance) & Process Capability (Cp, Cpk)"
        )
    }

    skey, sname, stier = spc_routing.get(
        spc_scenario,
        ("run_chart", "Healthcare Run Chart", "Quality Improvement Surveillance")
    )

    st.success(f"🎯 **Recommended Method:** `{sname}` — *{stier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {sname} on Page 9 👉", key=f"btn_jump_spc_{skey}", use_container_width=True):
            st.session_state["spc_direct_method_key"] = skey
            st.session_state["spc_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.switch_page("pages/9_📈_Statistical_Process_Control_and_Quality_Improvement.py")


# =====================================================================
# 9. Health Economics & Health Technology Assessment (HTA / NICE) Tree
# =====================================================================

def render_hta_decision_tree() -> None:
    """Renders the comprehensive Health Economics & Health Technology Assessment (HTA / NICE)
    decision tree covering all 5 architectures.
    """
    st.markdown("### 💷 Health Economics & Health Technology Assessment (HTA / NICE) Decision Tree (5 Architectures)")
    st.markdown(r"""
    This decision tree guides health economists, market access teams, and NHS commissioners from 
    **Appraisal Scope & Decision Objective** (Cost-Effectiveness vs Payer Affordability) 
    through **Disease Progression Dynamics & Parameter Uncertainty Audits** 
    (Trial-Based ICERs, Markov State-Transition Matrices, Multi-Arm NMB Rankings, and Monte Carlo PSA) 
    to the mathematically and regulatorily justified health economic model.
    """)

    mermaid_code = """```mermaid
flowchart TD
    HTAStart(["Health Economic & Policy Question"]) --> HTAGoal{"What is the Primary Evaluation Objective?"}

    %% 1. Efficiency vs Affordability
    HTAGoal -->|"Long-Term Economic Efficiency (Value for Money / QALY)"| CompDesign{"How Many Competing Strategies Are Being Compared?"}
    CompDesign -->|"Two Strategies (Novel Intervention vs Current Standard)"| DiseaseNat{"Is the Condition an Acute Event or Chronic Progressive State?"}
    DiseaseNat -->|"Acute / Fixed Horizon (Trial-Based Assessment)"| M1["1. ICER & Cost-Effectiveness Plane<br/>• Incremental Cost ΔC and Health Gain ΔQALY<br/>• 4-Quadrant Placement (Dominant, Dominated, Trade-off)<br/>• Evaluated against NICE £20,000–£30,000 / QALY threshold"]
    DiseaseNat -->|"Chronic Progressive Disease (Recurrent Events, Remissions)"| M3["3. Markov Cohort State-Transition Model<br/>• Multi-state matrix transitions (Well, Progression, Dead)<br/>• Half-cycle correction (trapezoidal rule)<br/>• 3.5% NICE annual discounting on costs and QALYs"]

    CompDesign -->|"Three or More Competing Active Strategies"| M2["2. Net Monetary Benefit (NMB) & NHB<br/>• Linear value transformation: NMB = λ * ΔE - ΔC<br/>• Eliminates ratio instability and negative denominator traps<br/>• Direct ranking: Select strategy that maximizes NMB"]

    %% 2. Parameter Uncertainty
    HTAGoal -->|"Quantifying Decision Uncertainty & Probability of Error"| M4["4. Probabilistic Sensitivity Analysis (PSA)<br/>• Monte Carlo sampling across parameter distributions<br/>• Gamma for skewed costs, Beta for health utilities (0-1)<br/>• Cost-Effectiveness Acceptability Curves (CEAC) & EVPI"]

    %% 3. Budget Impact / Affordability
    HTAGoal -->|"Short-Term Financial Affordability for Payer (1 to 5 Years)"| M5["5. Budget Impact Analysis (BIA)<br/>• Target eligible patient epidemiology and diagnosis cascade<br/>• Dynamic market uptake penetration trajectories<br/>• Gross pharmacy expenditure minus hospital bed-day offsets<br/>• Evaluated against the NHS £20m Budget Impact Test"]

    classDef ce fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef nmb fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef markov fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef psa fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef bia fill:#fff1f2,stroke:#e11d48,stroke-width:2px,color:#9f1239;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class M1 ce;
    class M2 nmb;
    class M3 markov;
    class M4 psa;
    class M5 bia;
    class HTAStart,HTAGoal,CompDesign,DiseaseNat decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 Health Economics & HTA Principles: QALYs, NICE Thresholds & Opportunity Costs", expanded=False):
        hta_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #ca8a04; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #854d0e; margin-bottom: 0.35rem;">
                💷 Core Principles for Health Technology Assessment & Economic Modeling
            </div>
            • <strong>Opportunity Cost in Fixed-Budget Healthcare:</strong> Public healthcare providers (NHS) cannot create additional funding out of thin air. Reimbursing a costly drug displaces existing NHS services. The NICE cost-effectiveness threshold (£20,000–£30,000 / QALY) reflects the marginal productivity of the NHS.<br>
            • <strong>Cost-Effectiveness vs Budget Impact:</strong> Cost-Effectiveness Analysis evaluates long-term efficiency across a patient's entire lifetime (cost per QALY gained with 3.5% discounting). Budget Impact Analysis evaluates immediate cash-flow affordability for local commissioners over a 1-to-5 year operational horizon.<br>
            • <strong>The Net Monetary Benefit (NMB) Solution:</strong> When ranking 3 or more competing clinical regimens, pairwise ICERs become prone to ratio instability and extended dominance. Transforming outcomes into NMB ($\lambda \cdot \Delta E - \Delta C$) provides a linear scale where the highest NMB unambiguously represents the optimal clinical strategy.<br>
            • <strong>Markov Cohort Modeling & Half-Cycle Correction:</strong> Chronic disease modeling requires tracking cohorts across discrete health states (e.g. Progression-Free, Progressed Disease, Death). Half-cycle corrections (averaging start and end of cycle state distributions) prevent over- or under-estimating outcomes.<br>
            • <strong>Probabilistic Sensitivity Analysis (PSA):</strong> Deterministic point estimates do not capture real-world uncertainty. PSA assigns statistical distributions to all parameters (Gamma for costs, Beta for health utilities) to produce Cost-Effectiveness Acceptability Curves (CEAC) for committee decision-making.
        </div>
        """
        render_html(hta_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick Health Economics & HTA Method Selector")
    c1, c2 = st.columns(2)
    with c1:
        hta_scenario = st.selectbox(
            "What is your health technology assessment scenario?",
            options=[
                "Biologic DMARD vs standard care in severe rheumatoid arthritis (Incremental cost per QALY)",
                "Heart failure multi-arm comparison ranking 3 active treatments (ACEi vs ARNI vs SGLT2i)",
                "10-year oncology survival trace with progression-free and metastatic health states (3.5% discounting)",
                "Minimally invasive heart valve implant with parameter uncertainty across 1,000 Monte Carlo iterations",
                "NHS England 5-year financial impact for 250,000 eligible chronic kidney disease patients"
            ],
            key="hta_tree_scenario_select"
        )
    with c2:
        hta_setting = st.selectbox(
            "Primary analytical requirement & mathematical framework:",
            options=[
                "Pairwise ICER calculation (£/QALY) on 4-quadrant Cost-Effectiveness Plane",
                "Linear Net Monetary Benefit (NMB) ranking across multiple competing strategies",
                "Multi-state Markov matrix simulation with half-cycle correction & NICE discounting",
                "Monte Carlo PSA parameter sampling with Cost-Effectiveness Acceptability Curves (CEAC)",
                "5-year undiscounted budget impact forecasting with hospitalization bed-day offsets"
            ],
            key="hta_tree_setting_select"
        )

    hta_routing = {
        "Biologic DMARD vs standard care in severe rheumatoid arthritis (Incremental cost per QALY)": (
            "icer_plane",
            "Cost-Effectiveness & ICER Plane",
            "Pairwise Value for Money & Threshold Appraisal"
        ),
        "Heart failure multi-arm comparison ranking 3 active treatments (ACEi vs ARNI vs SGLT2i)": (
            "nmb_nhb",
            "Net Monetary Benefit (NMB) & NHB",
            "Linear Value Metric & Multi-Strategy Ranking"
        ),
        "10-year oncology survival trace with progression-free and metastatic health states (3.5% discounting)": (
            "markov_model",
            "Markov Decision Analytic Cohort Model",
            "Longitudinal Chronic Disease State Transitions"
        ),
        "Minimally invasive heart valve implant with parameter uncertainty across 1,000 Monte Carlo iterations": (
            "psa_ceac",
            "Probabilistic Sensitivity Analysis (PSA) & CEAC",
            "Decision Risk & Uncertainty Modeling"
        ),
        "NHS England 5-year financial impact for 250,000 eligible chronic kidney disease patients": (
            "budget_impact",
            "Budget Impact Analysis (BIA)",
            "5-Year Payer Financial Feasibility & Affordability"
        )
    }

    hkey, hname, htier = hta_routing.get(
        hta_scenario,
        ("icer_plane", "Cost-Effectiveness & ICER Plane", "Health Economics Evaluation")
    )

    st.success(f"🎯 **Recommended Method:** `{hname}` — *{htier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {hname} on Page 10 👉", key=f"btn_jump_hta_{hkey}", use_container_width=True):
            st.session_state["hta_direct_method_key"] = hkey
            st.session_state["hta_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.switch_page("pages/10_💷_Health_Economics_and_HTA.py")



# =====================================================================
# 10. Missing Data Mechanisms, Imputation & Sensitivity Audits Tree
# =====================================================================

def render_missing_data_decision_tree() -> None:
    """Renders the comprehensive Missing Data Mechanisms, Imputation & Sensitivity Audits
    decision tree covering all 5 architectures.
    """
    st.markdown("### 🧩 Missing Data Mechanisms, Imputation & Sensitivity Audits Decision Tree (5 Architectures)")
    st.markdown(r"""
    This decision tree guides clinical trialists, biostatisticians, and epidemiologists from 
    **Missingness Mechanism Diagnosis** (MCAR vs MAR vs MNAR) through **Imputation Strategy Selection** 
    (Complete Case retention decay, Single Imputation variance collapse, Multiple Imputation by Chained Equations with Rubin's Pooling, and Algorithmic $k$-NN / MissForest) 
    to **Regulatory Sensitivity Audits** (ICH E9 R1 Addendum Tipping Point Analysis).
    """)

    mermaid_code = """```mermaid
flowchart TD
    MissStart(["Incomplete Dataset or Clinical Dropout Pattern"]) --> MechAudit{"What is the Diagnosed or Plausible Missingness Mechanism?"}

    %% 1. MCAR Path
    MechAudit -->|"Missing Completely at Random (MCAR)"| LittleCheck{"Is Missing Fraction < 5% on Uncorrelated Covariates?"}
    LittleCheck -->|"Yes: Negligible Loss (< 5%)"| M2["2. Complete Case Analysis & Single Imputation Hazards<br/>• Listwise deletion unbiased under strict MCAR<br/>• Danger: (1-p)^K multi-variable retention collapse<br/>• Mean/LOCF imputation strictly censured due to variance deflation"]
    LittleCheck -->|"No: Substantial Missingness or Formal Test Required"| M1["1. Rubin's Taxonomy & Diagnostic Audits<br/>• Little's Multivariate MCAR Test (d^2 ~ Chi-Square)<br/>• Missingness indicators R_i & logistic dropout modeling<br/>• Missing data pattern matrix & retention audit"]

    %% 2. MAR Path
    MechAudit -->|"Missing at Random (MAR: Conditional on Observed Covariates)"| ModelType{"What is the Analytical Objective & Data Structure?"}
    ModelType -->|"Parametric Inference / Regression with Rubin's Pooling"| M3["3. MICE & Rubin's Rules Pooling<br/>• Fully Conditional Specification (FCS / PMM)<br/>• Generate m completed datasets (m >= 20)<br/>• Rubin's Rules: Q_bar, U_bar, B, Total Variance T, FMI, RIV, df"]
    ModelType -->|"Non-Parametric / High-Dimensional Non-Linear Interactions"| M4["4. Algorithmic k-NN & MissForest Imputation<br/>• Gower distance k-nearest neighbors donor matching<br/>• Iterative non-parametric Random Forest (MissForest)<br/>• Preserves complex multi-way non-linear interactions"]

    %% 3. MNAR Path
    MechAudit -->|"Missing Not at Random (MNAR: Informative Dropout)"| M5["5. MNAR Pattern-Mixture & Tipping Point Analysis<br/>• Mandated by ICH E9 (R1) Addendum on Estimands<br/>• Pattern-mixture delta-adjustment penalty on dropouts<br/>• Systematic tipping point search delta* where significance flips (p >= 0.05)"]

    classDef diag fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef cca fill:#fef2f2,stroke:#ef4444,stroke-width:2px,color:#991b1b;
    classDef mice fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef ml fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef tipping fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class M1 diag;
    class M2 cca;
    class M3 mice;
    class M4 ml;
    class M5 tipping;
    class MissStart,MechAudit,LittleCheck,ModelType decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 Missing Data Principles: Rubin's Taxonomy, Variance Collapse & ICH E9 Compliance", expanded=False):
        missing_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #7c3aed; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #5b21b6; margin-bottom: 0.35rem;">
                🧩 Core Principles for Missing Data Mechanisms & Principled Imputation
            </div>
            • <strong>Rubin's Taxonomy (MCAR vs. MAR vs. MNAR):</strong> Under MCAR, missingness is completely independent of both observed and unobserved data. Under MAR, missingness depends on observed covariates (e.g. older patients skip tests, but conditioned on age, missingness is random). Under MNAR, missingness depends on unobserved values (e.g. patients with severe depressive relapse refuse questionnaires).<br>
            • <strong>The Lethal Hazard of Single Mean Imputation:</strong> Replacing missing values with the sample mean falsely deflates sample variance by $(1-\pi)$, artificially shrivels standard errors by $\sqrt{1-\pi}$, and inflates Type I errors (manufacturing false-positive trial results). Last Observation Carried Forward (LOCF) is banned by regulators because it unrealistically assumes zero trajectory drift.<br>
            • <strong>Why Multiple Imputation (MICE) is the Scientific Standard:</strong> MICE generates $m$ complete datasets using chained conditional regression models. Rubin's Rules explicitly calculate between-imputation variance $B$ to add back the honest uncertainty caused by missing values into the total variance $T = \bar{U} + (1 + 1/m)B$.<br>
            • <strong>Algorithmic Imputation (k-NN & MissForest):</strong> When relationships are non-linear or feature high-order interactions, parametric regressions in MICE can be misspecified. $k$-NN with Gower distance and MissForest (iterative random forest) capture non-parametric topology without distribution assumptions.<br>
            • <strong>ICH E9 (R1) Mandate & Tipping Point Audits:</strong> Because MAR is fundamentally untestable from observed data alone, international regulators mandate tipping point sensitivity analysis. By applying a systematic clinical penalty $\delta$ exclusively to dropouts, we identify the exact threshold $\delta^*$ where trial significance flips ($p \ge 0.05$).
        </div>
        """
        render_html(missing_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick Missing Data & Imputation Method Selector")
    c1, c2 = st.columns(2)
    with c1:
        missing_scenario = st.selectbox(
            "What is your clinical or observational missing data scenario?",
            options=[
                "ICU sepsis registry with 15% missing arterial lactate; testing whether missingness is independent of severity (Little's test)",
                "Cardiovascular cohort evaluating power loss from listwise deletion vs variance shrinkage from mean imputation & LOCF",
                "Multicenter diabetes RCT evaluating HbA1c reduction across covariates with 20% MAR missingness (MICE & Rubin's rules)",
                "Complex EHR registry with non-linear biomarker interactions and mixed categorical/continuous labs (k-NN / MissForest)",
                "Phase III depression RCT evaluating drug vs placebo where dropouts may have experienced worse symptom relapse (ICH E9 tipping point)"
            ],
            key="missing_tree_scenario_select"
        )
    with c2:
        missing_setting = st.selectbox(
            "Primary analytical requirement & mathematical framework:",
            options=[
                "Multivariate Little's d^2 MCAR test, pattern matrix, and logistic missingness indicator modeling",
                "Quantifying listwise retention collapse (1-p)^K and variance shrinkage factor (1-pi)",
                "Chained equations (MICE / PMM) generating m datasets and pooling via Rubin's Rules (Q_bar, U_bar, B, T, FMI)",
                "Non-parametric Gower distance k-NN donor matching or iterative Random Forest (MissForest)",
                "Pattern-mixture delta-adjustment sensitivity audit locating regulatory tipping point delta* (p >= 0.05)"
            ],
            key="missing_tree_setting_select"
        )

    missing_routing = {
        "ICU sepsis registry with 15% missing arterial lactate; testing whether missingness is independent of severity (Little's test)": (
            "rubin_taxonomy",
            "Rubin's Taxonomy & Diagnostic Audits",
            "Little's MCAR Test & Missingness Pattern Diagnostics"
        ),
        "Cardiovascular cohort evaluating power loss from listwise deletion vs variance shrinkage from mean imputation & LOCF": (
            "complete_case_vs_single",
            "Complete Case Analysis vs. Single Imputation Hazards",
            "Listwise Retention Collapse & Variance Shrinkage Audits"
        ),
        "Multicenter diabetes RCT evaluating HbA1c reduction across covariates with 20% MAR missingness (MICE & Rubin's rules)": (
            "mice_rubin_rules",
            "MICE & Rubin's Rules Pooling",
            "Principled Multiple Imputation & Variance Decomposition"
        ),
        "Complex EHR registry with non-linear biomarker interactions and mixed categorical/continuous labs (k-NN / MissForest)": (
            "knn_missforest",
            "Algorithmic & Distance-Based Imputation",
            "k-NN Donor Matching & MissForest Ensemble"
        ),
        "Phase III depression RCT evaluating drug vs placebo where dropouts may have experienced worse symptom relapse (ICH E9 tipping point)": (
            "mnar_tipping_point",
            "MNAR Tipping Point Sensitivity Analysis",
            "ICH E9 (R1) Pattern-Mixture Delta-Adjustment Audit"
        )
    }

    mkey, mname, mtier = missing_routing.get(
        missing_scenario,
        ("rubin_taxonomy", "Rubin's Taxonomy & Diagnostic Audits", "Diagnostic & Theoretical Foundations")
    )

    st.success(f"🎯 **Recommended Method:** `{mname}` — *{mtier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {mname} on Page 11 👉", key=f"btn_jump_missing_{mkey}", use_container_width=True):
            st.session_state["missing_direct_method_key"] = mkey
            st.session_state["missing_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.switch_page("pages/11_🧩_Missing_Data_and_Imputation.py")



# =====================================================================
# 11. Longitudinal & Hierarchical Mixed-Effects Models Tree
# =====================================================================

def render_mixed_models_decision_tree() -> None:
    """Renders the comprehensive Longitudinal & Hierarchical Mixed-Effects Models
    decision tree covering all 5 architectures.
    """
    st.markdown("### 🪜 Longitudinal & Hierarchical Mixed-Effects Models Decision Tree (5 Architectures)")
    st.markdown(r"""
    This decision tree guides clinical trialists, biostatisticians, and healthcare data scientists from 
    **Data Structure & Outcome Scale** (Continuous vs Binary/Count) through **Clustering Hierarchy** 
    (Repeated Patient Measures vs 3-Level Organizational Nesting) to **Covariance Modeling & Goodness-of-Fit** 
    (Laird & Ware LMM, GLMM non-collapsibility, RM-ANOVA sphericity recovery, and Nakagawa $R^2$).
    """)

    mermaid_code = """```mermaid
flowchart TD
    MixedStart(["Longitudinal Repeated Measures or Multi-Tier Clustered Cohort"]) --> DataOutcome{"What is the Mathematical Form of the Outcome Variable?"}

    %% 1. Continuous Outcome
    DataOutcome -->|"Continuous Metric (e.g. ADAS-Cog, Blood Pressure)"| TierCheck{"What is the Clustered Organizational Structure?"}
    TierCheck -->|"2 Levels: Repeated Patient Follow-up Visits (Level 1 in Level 2)"| SlopeCheck{"Do Patients Differ Only at Baseline or Also in Recovery Speed?"}
    SlopeCheck -->|"Baseline Level Only"| M1a["1. LMM Random Intercept Model<br/>• Common population slope beta_time<br/>• Personalized starting baseline u_0i<br/>• Intraclass Correlation Coefficient (ICC)"]
    SlopeCheck -->|"Both Baseline & Rate of Change"| M1b["1. LMM Random Intercept + Slope Model<br/>• Laird & Ware formulation (Y_i = X_i*beta + Z_i*u_i + e_i)<br/>• Personalized trajectory slope u_1i * t<br/>• Empirical Bayes shrinkage estimates (BLUPs)"]

    TierCheck -->|"3+ Levels: Patients in Clinicians in Hospital Trusts"| M4["4. Hierarchical 3-Level Clustered Models<br/>• Additive variance decomposition (hosp + clin + error)<br/>• Variance Partition Coefficients (VPC_hosp, VPC_clin)<br/>• Contextual within-between neighborhood effects"]

    %% 2. Discrete / Non-Normal
    DataOutcome -->|"Binary Relapse (0/1) or Poisson Infection Counts"| M2["2. Generalized Linear Mixed Models (GLMM)<br/>• Logistic mixed regression (Logit link) & Poisson mixed (Log link)<br/>• Subject-Specific OR vs Marginal Population-Averaged OR<br/>• Non-collapsibility attenuation gap audit"]

    %% 3. Classical Comparison
    DataOutcome -->|"Evaluating Legacy Repeated Measures ANOVA vs LMM"| M3["3. RM-ANOVA vs. LMM Sphericity Tradeoff<br/>• Mauchly's sphericity test & Greenhouse-Geisser epsilon penalty<br/>• Listwise deletion penalty (devastating power loss)<br/>• LMM retains 100% of enrolled patients under MAR likelihood"]

    %% 4. Model Diagnostics & R2
    DataOutcome -->|"Residual Covariance Selection & Effect Size Reporting"| M5["5. Covariance Selection & Nakagawa R²<br/>• Maximum Likelihood (ML) vs REML unbiased estimation<br/>• Compound Symmetry vs Autoregressive AR(1) error decay<br/>• Nakagawa Marginal R²_m & Conditional R²_c metrics"]

    classDef lmm fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#312e81;
    classDef glmm fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef rmanova fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef h3 fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef diag fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class M1a,M1b lmm;
    class M2 glmm;
    class M3 rmanova;
    class M4 h3;
    class M5 diag;
    class MixedStart,DataOutcome,TierCheck,SlopeCheck decision;
```"""
    st.markdown(mermaid_code)

    # Triage Principles Container
    with st.expander("💡 Longitudinal & Mixed Model Principles: Sphericity, Non-Collapsibility & VPC", expanded=False):
        mixed_rules_html = r"""
        <div style="background-color: #f8fafc; border-left: 4px solid #4f46e5; padding: 0.85rem 1.1rem; border-radius: 4px; font-size: 0.88rem; line-height: 1.5; color: #1e293b;">
            <div style="font-weight: 700; color: #3730a3; margin-bottom: 0.35rem;">
                🪜 Core Principles for Longitudinal & Hierarchical Mixed Modeling
            </div>
            • <strong>Fixed Effects vs. Random Effects:</strong> Fixed effects represent the overarching population-average trend (e.g. drug effect across the whole trial). Random effects model individual subject personalities, allowing every patient to have their own unique baseline starting point ($u_{0i}$) and rate of recovery ($u_{1i}$).<br>
            • <strong>Why Mixed Models Crush Repeated Measures ANOVA:</strong> Classical RM-ANOVA uses listwise deletion: a patient missing 1 checkup has ALL visits thrown away. Linear Mixed Models (LMM) use full-information likelihood to retain 100% of participants, properly handling missing visits at random (MAR) and continuous, irregular follow-up times.<br>
            • <strong>The Non-Collapsibility Gap in GLMM:</strong> In logistic mixed models, the subject-specific conditional odds ratio ($	ext{OR}_{	ext{SS}}$) is always systematically larger in magnitude than the population-averaged marginal odds ratio ($	ext{OR}_{	ext{marginal}}$) due to non-collapsibility of the logit link.<br>
            • <strong>Multilevel Variance Partition Coefficients (VPC):</strong> In 3-level healthcare data (patients in surgeons in hospital trusts), VPC decomposes total variation into percentages attributable to hospital trust policies, surgeon skill, and patient biology.<br>
            • <strong>Nakagawa & Schielzeth (2013) R² Metrics:</strong> Standard $R^2$ is undefined in mixed models. Report Marginal $R^2_m$ (variance explained purely by fixed treatments) alongside Conditional $R^2_c$ (variance explained by both fixed treatments and patient random baselines).
        </div>
        """
        render_html(mixed_rules_html)

    # Interactive Method Selector
    st.markdown("#### ⚡ Quick Longitudinal & Multilevel Method Selector")
    c1, c2 = st.columns(2)
    with c1:
        mixed_scenario = st.selectbox(
            "What is your longitudinal or hierarchical research scenario?",
            options=[
                "Alzheimer's trial with 4 annual visits; patients start at different severities and decline at different speeds",
                "ICU sepsis monitoring evaluating repeated binary organ failure relapse (0/1) across hospital wards",
                "Physiotherapy chronic pain trial with 20% missed appointment visits and sphericity breakdown (p < 0.001)",
                "NHS surgical audit evaluating recovery times across patients nested within 120 surgeons across 25 Hospital Trusts",
                "Serial troponin biomarker draws over 72 hours; selecting between Compound Symmetry and AR(1) error decay"
            ],
            key="mixed_tree_scenario_select"
        )
    with c2:
        mixed_setting = st.selectbox(
            "Primary analytical requirement & mathematical framework:",
            options=[
                "Laird & Ware LMM with random intercepts and personalized random slopes (ADAS-Cog trajectory)",
                "Generalized Linear Mixed Model (GLMM) mixed logistic regression with subject-specific vs marginal OR",
                "Evaluating Mauchly's sphericity, Greenhouse-Geisser epsilon, and LMM 100% MAR patient retention",
                "3-level hierarchical variance decomposition (VPC_hosp, VPC_clin, VPC_error) and contextual effects",
                "Covariance model selection (VC, CS, AR1, UN) via AIC/BIC and Nakagawa marginal/conditional R^2"
            ],
            key="mixed_tree_setting_select"
        )

    mixed_routing = {
        "Alzheimer's trial with 4 annual visits; patients start at different severities and decline at different speeds": (
            "lmm_random_slopes",
            "Linear Mixed-Effects Models (LMM)",
            "Random Intercepts & Personalized Trajectory Slopes"
        ),
        "ICU sepsis monitoring evaluating repeated binary organ failure relapse (0/1) across hospital wards": (
            "glmm_binary_count",
            "Generalized Linear Mixed Models (GLMM)",
            "Repeated Binary Relapse & Count Outcomes"
        ),
        "Physiotherapy chronic pain trial with 20% missed appointment visits and sphericity breakdown (p < 0.001)": (
            "rm_anova_vs_lmm",
            "RM-ANOVA vs. Mixed Models Tradeoff",
            "Sphericity Violations & Missing Data Likelihood Recovery"
        ),
        "NHS surgical audit evaluating recovery times across patients nested within 120 surgeons across 25 Hospital Trusts": (
            "hierarchical_3level",
            "Hierarchical 3-Level Clustered Models",
            "Multilevel Organizational Variance Partitioning (VPC)"
        ),
        "Serial troponin biomarker draws over 72 hours; selecting between Compound Symmetry and AR(1) error decay": (
            "covariance_model_selection",
            "Covariance Structure Selection & Diagnostics",
            "AR(1) Error Decay, ML vs. REML, and Nakagawa R^2 Metrics"
        )
    }

    mkey, mname, mtier = mixed_routing.get(
        mixed_scenario,
        ("lmm_random_slopes", "Linear Mixed-Effects Models (LMM)", "Continuous Repeated Measures")
    )

    st.success(f"🎯 **Recommended Method:** `{mname}` — *{mtier}*")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {mname} on Page 12 👉", key=f"btn_jump_mixed_{mkey}", use_container_width=True):
            st.session_state["mixed_direct_method_key"] = mkey
            st.session_state["mixed_nav_mode_radio"] = "🔍 Direct Method Explorer"
            st.switch_page("pages/12_🪜_Longitudinal_and_Mixed_Models.py")
