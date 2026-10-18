"""Flowchart and Decision Tree UI Components.

Provides comprehensive visual decision trees, multi-pillar triage diagrams,
interactive path simulators, and educational triage rules across:
1. Master Cross-Pillar Triage (Statistics vs Epidemiology vs Machine Learning)
2. Classical Statistical Hypothesis Testing (18 Tests, Parametric & Non-Parametric)
3. Observational Epidemiology & Public Health (17 Methods, Risk, Rates & Screening)
4. Machine Learning & Predictive Modeling (16 Architectures, Rare Events & Clustering)
5. Biostatistical Sample Size & Power Planning (10 Frameworks, Attrition & Clusters)
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

    to the appropriate quantitative analytical pillar.
    """
    st.markdown("### 🌐 Master Cross-Pillar Triage: Choosing Your Analytical Pillar")
    st.markdown("""
    When designing a quantitative analysis, the most fundamental decision is determining **which analytical paradigm** fits your research question, data collection process, and governance requirements:
    * **Pillar 1: Classical Statistics** — Hypothesis testing, $p$-values, effect sizes, and confidence intervals under controlled or randomized conditions.
    * **Pillar 2: Observational Epidemiology** — Quantifying disease risk, incidence rates, diagnostic accuracy, and controlling confounding without experimental randomization.
    * **Pillar 3: Machine Learning & AI** — Predictive scoring, handling high-dimensional complex patterns, rare event detection, and unsupervised phenotypic discovery.
    * **Pillar 4: Biostatistical Sample Size & Power** — Formal trial planning, minimal detectable differences, and ethical sample sizing.
    """)

    # Visual Mermaid Diagram
    mermaid_code = """```mermaid
flowchart TD
    Start(["Research Question or Analytical Problem"]) --> Q1{"What is the Primary Analytical Goal?"}

    Q1 -->|"Hypothesis Testing & Causal Inference"| Q2{"Was Data Collected under Controlled / Randomized Conditions?"}
    Q2 -->|"Yes: Controlled / RCT / Pre-Post"| P1["📊 Pillar 1: Classical Statistical Tests<br/>(t-tests, ANOVA, Mann-Whitney, Chi-Square)"]
    Q2 -->|"No: Observational Cohort / Survey"| P2["🏥 Pillar 2: Observational Epidemiology<br/>(Relative Risk, Odds Ratio, Standardization, Confounding)"]

    Q1 -->|"Predictive Scoring, Automation or Pattern Discovery"| Q3{"Are Ground-Truth Target Labels Available?"}
    Q3 -->|"Yes: Target Y Known"| ML_Sup["🤖 Pillar 3: Supervised ML<br/>(Regularized Regression, Cost-Sensitive Ensembles, PR-AUC)"]
    Q3 -->|"No: Unsupervised Discovery"| ML_Unsup["🧩 Pillar 3: Unsupervised ML<br/>(K-Means Clustering, DBSCAN, PCA, Stability Metrics)"]

    Q1 -->|"Prospective Study Sizing & Protocol Design"| P4["📐 Pillar 4: Sample Size & Power<br/>(Closed-Form Formulas, MCID, Attrition Cushion, VIF)"]

    P1 -.->|"Requires Power Sizing"| P4
    P2 -.->|"Requires Power Sizing"| P4
    ML_Unsup -.->|"Requires Sample Sizing"| P4

    classDef primary fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef epi fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef ml fill:#faf5ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef power fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412;
    classDef decision fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#0f172a;

    class P1 primary;
    class P2 epi;
    class ML_Sup,ML_Unsup ml;
    class P4 power;
    class Q1,Q2,Q3 decision;
```"""
    st.markdown(mermaid_code)

    # Master Triage Comparison Matrix
    with st.expander("📋 Master Triage Matrix: 4-Way Cross-Pillar Comparison", expanded=False):
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
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Primary Objective</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Test predefined null hypothesis ($H_0$), confirm intervention effects</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Quantify disease risk, incidence rates, and control observational confounding</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Maximize out-of-sample prediction accuracy or discover latent patient clusters</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Determine minimum sample size needed to detect effect size $\delta$ at $\ge 80\%$ power</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Data Architecture</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Controlled experiments, randomized trials, small-to-moderate samples ($N < 500$)</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Cohorts, case-control matching, surveys, demographic registries</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">High-dimensional tabular data, telemetry streams, $N \ge 500$, many features</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Pre-trial planning stage before data collection commences</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #ffffff;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Key Output Metrics</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">$t$-stat, $F$-ratio, $p$-value, Cohen's $d$, 95% Confidence Intervals</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Relative Risk (RR), Odds Ratio (OR), DSR, Sensitivity/Specificity, NNT</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">PR-AUC, ROC-AUC, Brier Score, Silhouette, Davies-Bouldin, Calinski-Harabasz</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">Total Required $N$, Per-Group $n$, Power Curves, Attrition Cushion ($N / (1-L)$)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0; background-color: #f8fafc;">
                        <td style="padding: 0.6rem 0.85rem; font-weight: 600; color: #334155;">Primary Governance</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">CONSORT 2010, ICH-E9, APA Guidelines</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">STROBE Guidelines, PRISMA 2020, STARD</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">TRIPOD-AI, NHS AI Ethics, Four-Fifths 80% Rule</td>
                        <td style="padding: 0.6rem 0.85rem; color: #1f2937;">NIHR, MRC, FDA Statistical Guidance, SPIRIT</td>
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
                "Predict rare clinical event (e.g. ICU Sepsis)",
                "Forecast sequential time series demand",
                "Discover unknown sub-phenotypes / patient clusters",
                "Determine required sample size for a grant proposal"
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
                "Pre-Study Design Planning Stage"
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

    if "grant proposal" in triage_goal or "Pre-Study" in triage_data:
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
    elif "Predict rare clinical event" in triage_goal or "Forecast sequential" in triage_goal:
        resolved_pillar = "Machine Learning & AI"
        if "Forecast" in triage_goal:
            resolved_method = "Lagged Ensemble Time Series Forecaster"
        else:
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

    Cohort, Case-Control, Cross-Sectional, Standardization, Screening, and Meta-Analysis.
    """
    st.markdown("### 🏥 Observational Epidemiology & Public Health Decision Tree (17 Methods)")
    st.markdown("""
    Observational studies cannot rely on experimental randomization. This decision tree routes based on 
    **Temporal Direction** (Prospective vs Retrospective vs Cross-Sectional), **Sampling Strategy** (Exposed vs Outcome-First), 
    and **Confounding Adjustment** to determine the valid epidemiological effect measure.
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

    %% Standardization
    EpiStart -->|"Demographic Comparison across Regions"| Std{"Age-Specific Data Availability"}
    Std -->|"Local Age-Specific Death Rates Known"| DSR["Direct Age-Standardized Rate (DSR)<br/>European Standard Population"]
    Std -->|"Only Total Deaths Known in Small District"| SMR["Indirect Standardized Mortality Ratio (SMR)<br/>Expected Deaths Benchmark"]

    %% Evidence Synthesis & Survival
    EpiStart -->|"Clinical Evidence Synthesis / Time-to-Event"| Synth{"Synthesis Type"}
    Synth -->|"Pooling Multiple Trials (PRISMA 2020)"| Meta["Random-Effects Meta-Analysis<br/>DerSimonian-Laird & I-squared"]
    Synth -->|"Time-to-Event Survival Analysis"| KM["Kaplan-Meier Curves & Log-Rank Test<br/>Cox Proportional Hazards"]

    classDef cohort fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef casecontrol fill:#fef2f2,stroke:#dc2626,stroke-width:2px,color:#991b1b;
    classDef screening fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef std fill:#fefce8,stroke:#ca8a04,stroke-width:2px,color:#854d0e;
    classDef decision fill:#f8fafc,stroke:#475569,stroke-width:1.5px,color:#0f172a;

    class RR,IRR,AR cohort;
    class OR,MH,Outbreak casecontrol;
    class Prev,Diag screening;
    class DSR,SMR,Meta,KM std;
    class EpiStart,Cohort,CC,Cross,Std,Synth decision;
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
                "Regional Demographic Rate Standardization",
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
                "Direct Age-Standardized Rate (DSR) with Derbyshire Map",
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
        "Direct Age-Standardized Rate (DSR) with Derbyshire Map": ("standardization", "Direct Standardization (DSR & Map)"),
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
        "Audit demographic fairness and disparate impact": ("fairness_audit", "Algorithmic Fairness & 80% Disparate Impact Audit"),
        "Forecast chronological sequential time series demand": ("time_series_forecaster", "Lagged Ensemble Time Series Forecaster"),
        "Discover latent patient clusters / sub-phenotypes (Unsupervised)": ("kmeans_clustering", "K-Means Clustering with 3-Metric Validation")
    }
    mkey, mname = ml_routing.get(ml_target_type, ("kmeans_clustering", "K-Means Clustering"))

    st.success(f"🎯 **Recommended Architecture:** `{mname}`")
    col_jump, _ = st.columns([1.5, 2.5])
    with col_jump:
        if st.button(f"Launch {mname} on Page 3 👉", key=f"btn_jump_ml_{mkey}", use_container_width=True):
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
