"""Custom Sidebar Navigation Component.

Provides a unified, structured multi-section sidebar navigation separating:
1. 🎯 Strategy, Problem Formulation & Universal Method Selector (Cross-Pillar Router)
2. 📚 Individual Educational Modules (Isolated Step-by-Step Training)
"""

from typing import Optional
import streamlit as st


def render_custom_sidebar_navigation(current_page: str = ""):
    """Renders structured custom sidebar navigation with distinct sections for

    method selection vs. isolated educational training modules.
    """
    # 1. Hide default Streamlit flat page navigation list via CSS
    st.sidebar.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. App Branding Header
    st.sidebar.markdown("""
    <div style="margin-bottom: 0.9rem; padding-bottom: 0.6rem; border-bottom: 1px solid #e2e8f0;">
        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 0.4rem;">
            🧭 <span>Method Finder</span>
        </div>
        <div style="font-size: 0.74rem; color: #64748b; margin-top: 0.15rem; line-height: 1.25;">
            Decision Support & Quantitative Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Section 1: Problem Formulation & Universal Method Selector
    st.sidebar.markdown("""
    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #166534; margin-bottom: 0.15rem;">
            🎯 Strategy & Method Selection
        </div>
        <div style="font-size: 0.72rem; color: #15803d; line-height: 1.3;">
            Formulate research problems, evaluate 3-pillar tradeoffs & find recommended methods.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.page_link(
        "app.py",
        label="Home Overview Hub",
        icon="🏠"
    )
    st.sidebar.page_link(
        "pages/7_🎯_Study_Design_and_Method_Selector.py",
        label="Study Design & Selector",
        icon="🎯"
    )
    st.sidebar.page_link(
        "pages/8_🗺️_Decision_Flowcharts.py",
        label="Decision Trees & Flowcharts",
        icon="🗺️"
    )

    st.sidebar.markdown("---")

    # Clean key for selectbox to avoid conflicts across page reruns
    page_key_slug = current_page.replace(".", "_").replace("/", "_").replace(" ", "_")

    # 4. Section 2: Statistical Fundamentals
    st.sidebar.markdown("""
    <div style="background-color: #fefce8; border: 1px solid #fef08a; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #854d0e; margin-bottom: 0.15rem;">
            📚 Statistical Fundamentals
        </div>
        <div style="font-size: 0.72rem; color: #713f12; line-height: 1.3;">
            Core concepts: scales, central tendency, dispersion, CLT & confidence intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fundamentals_topics = {
        "⚡ Jump to a fundamental topic...": None,
        "🏷️ 1. Measurement Scales & Data Types": "scales_data_types",
        "⚖️ 2. Central Tendency (Mean/Median/Mode)": "central_tendency",
        "📏 3. Dispersion, Spread & IQR (Tukey's Fences)": "dispersion_iqr",
        "🔬 4. SD vs. Standard Error of the Mean (SEM)": "sd_vs_sem",
        "🔔 5. Distribution Shapes, Skewness & CLT": "distributions_clt",
        "🎯 6. Confidence Intervals (Mean & Proportion)": "confidence_intervals"
    }

    selected_fund_jump = st.sidebar.selectbox(
        "Fundamentals Topic Selector:",
        options=list(fundamentals_topics.keys()),
        index=0,
        key=f"nav_fund_select_{page_key_slug}",
        help="Jump directly to any fundamental statistical concept."
    )

    if selected_fund_jump and fundamentals_topics[selected_fund_jump] is not None:
        target_topic = fundamentals_topics[selected_fund_jump]
        st.session_state["fundamentals_active_topic"] = target_topic
        if current_page != "pages/0_📚_Statistical_Fundamentals.py":
            st.switch_page("pages/0_📚_Statistical_Fundamentals.py")

    st.sidebar.page_link(
        "pages/0_📚_Statistical_Fundamentals.py",
        label="0. Statistical Fundamentals",
        icon="📚"
    )

    st.sidebar.markdown("---")

    # 5. Section 3: Individual Educational Training Modules
    st.sidebar.markdown("""
    <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #1e40af; margin-bottom: 0.15rem;">
            📚 Individual Training Modules
        </div>
        <div style="font-size: 0.72rem; color: #1d4ed8; line-height: 1.3;">
            Self-contained modules with worked examples, Excel guides & live simulations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dropdown Quick-Jump Selector for Educational Modules
    learning_pages = {
        "⚡ Jump to a training module...": None,
        "📊 1. Classical Statistical Tests": "pages/1_📊_Statistical_Tests.py",
        "🏥 2. Epidemiological Methods": "pages/2_🏥_Epidemiology_Methods.py",
        "🤖 3. Machine Learning & AI": "pages/3_🤖_Machine_Learning.py",
        "📐 4. Sample Size & Power": "pages/4_📐_Sample_Size_and_Power.py",
        "📋 5. Survey & Questionnaire Analysis": "pages/5_📋_Survey_and_Questionnaire_Analysis.py",
        "🗣️ 6. Natural Language Processing": "pages/6_🗣️_Natural_Language_Processing.py"
    }

    # Clean key for selectbox to avoid conflicts across page reruns
    page_key_slug = current_page.replace(".", "_").replace("/", "_").replace(" ", "_")
    selected_jump = st.sidebar.selectbox(
        "Training Module Selector:",
        options=list(learning_pages.keys()),
        index=0,
        key=f"nav_quick_select_{page_key_slug}",
        help="Select any individual educational module to study its methodology in isolation."
    )

    if selected_jump and learning_pages[selected_jump] is not None:
        target_route = learning_pages[selected_jump]
        if target_route != current_page:
            st.switch_page(target_route)

    # Direct 1-Click Page Links for Educational Modules
    st.sidebar.page_link(
        "pages/1_📊_Statistical_Tests.py",
        label="1. Classical Statistics",
        icon="📊"
    )
    st.sidebar.page_link(
        "pages/2_🏥_Epidemiology_Methods.py",
        label="2. Epidemiology Methods",
        icon="🏥"
    )
    st.sidebar.page_link(
        "pages/3_🤖_Machine_Learning.py",
        label="3. Machine Learning & AI",
        icon="🤖"
    )
    st.sidebar.page_link(
        "pages/4_📐_Sample_Size_and_Power.py",
        label="4. Sample Size & Power",
        icon="📐"
    )
    st.sidebar.page_link(
        "pages/5_📋_Survey_and_Questionnaire_Analysis.py",
        label="5. Survey & Questionnaire Analysis",
        icon="📋"
    )
    st.sidebar.page_link(
        "pages/6_🗣️_Natural_Language_Processing.py",
        label="6. Natural Language Processing",
        icon="🗣️"
    )

    st.sidebar.markdown("---")


def render_roundtrip_navigation_banner():
    """Renders a non-intrusive contextual breadcrumb banner at the top of educational training pages
    when navigated from the Study Design & Method Selector (Page 7).
    """
    consultation = st.session_state.get("selector_consultation") if hasattr(st, "session_state") else None
    if not isinstance(consultation, dict) or not consultation.get("active"):
        return

    problem_name = consultation.get("problem_name", "Research Problem")
    method_name = consultation.get("method_name", "Recommended Method")
    domain_name = consultation.get("domain_name", "")
    sample_scale = consultation.get("sample_scale", "")
    scale_text = f" • <strong>Cohort:</strong> {sample_scale}" if sample_scale else ""
    domain_text = f" • <strong>Domain:</strong> {domain_name}" if domain_name else ""
    source_title = consultation.get("source_title", "Study Design Selector")
    source_icon = consultation.get("source_icon", "🎯")
    return_page = consultation.get("return_page", "pages/7_🎯_Study_Design_and_Method_Selector.py")
    return_label = consultation.get("return_label", "← Return to Selector")
    sub_text = consultation.get("sub_text", "Method pre-selected based on research objective & data architecture")

    col_banner, col_actions = st.columns([3.8, 1.2])

    with col_banner:
        st.markdown(rf"""
        <div style="background: linear-gradient(90deg, #f0fdf4 0%, #ecfdf5 100%); border-left: 4px solid #16a34a; border-radius: 4px 6px 6px 4px; padding: 0.55rem 0.9rem; margin-bottom: 0.85rem; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
            <div style="display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap;">
                <span style="font-size: 1.05rem;">{source_icon}</span>
                <span style="font-size: 0.86rem; font-weight: 700; color: #166534;">
                    Active Consultation from {source_title}:
                </span>
                <span style="font-size: 0.85rem; color: #1e293b;">
                    <strong>{method_name}</strong> for <em>{problem_name}</em>
                </span>
            </div>
            <div style="font-size: 0.74rem; color: #4b5563; margin-top: 0.15rem; margin-left: 1.5rem;">
                {sub_text}{scale_text}{domain_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_actions:
        b1, b2 = st.columns([1.6, 1])
        with b1:
            if st.button(return_label, key="btn_return_to_selector_roundtrip", use_container_width=True, help=f"Return to {source_title}."):
                st.switch_page(return_page)
        with b2:
            if st.button("✕ Clear", key="btn_clear_selector_roundtrip", use_container_width=True, help="Dismiss this consultation context banner."):
                st.session_state["selector_consultation"] = None
                st.rerun()


