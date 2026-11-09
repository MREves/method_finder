"""Streamlit Page: Statistical Process Control (SPC) & Quality Improvement (QI) Healthcare Analytics.

Pillar 9 of Method Finder.
Guides healthcare executives, quality improvement leads, clinical audit teams, and NHS analysts through:
1. Healthcare Run Charts (Baseline Medians, Shift, Trend, Runs, Astronomical Points)
2. Shewhart XmR / I-MR Charts (Individuals & Moving Range, 3-Sigma Limits, Average Moving Range)
3. Laney p' & Standard p-Charts (Binomial Proportions & Overdispersion sigma_z Correction for Large Patient Subgroups)
4. Laney u' & Standard u-Charts (Poisson Infection Rates per 1,000 Bed-Days with Overdispersion Adjustment)
5. NHS Making Data Count & Capability Analysis (System Icons: Improvement/Concern & Target Assurance Cp, Cpk)
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
from src.ui_components.spc_widgets import (
    render_spc_plain_english_guide,
    render_spc_mode_selector,
    render_spc_method_module
)
from src.spc_qi.spc_schema import SPC_METHODS

st.set_page_config(
    page_title="Statistical Process Control & Quality Improvement | MethodFinder",
    page_icon="📈",
    layout="wide"
)

# Render Custom Sidebar Navigation
render_custom_sidebar_navigation("pages/9_📈_Statistical_Process_Control_and_Quality_Improvement.py")

# Top Header & Contextual Breadcrumb
st.title("📈 Statistical Process Control & Quality Improvement (QI)")
render_roundtrip_navigation_banner()

st.info("💡 **Educational Disclaimer:** All hospital wait times, surgical site infection counts, surgical mortality rates, catheter-associated urinary tract infections (CAUTI), and NHS 4-hour A&E access metrics across this module are completely synthetic and fabricated solely for educational and quality improvement training purposes.")

st.markdown("""
Welcome to the **Statistical Process Control (SPC) & Quality Improvement** module. In healthcare operations and clinical audit, 
distinguishing **common cause variation** (inherent system noise) from **special cause variation** (statistically authentic 
signals of system change or deterioration) is essential. Reacting to common cause variation as if it were a genuine crisis represents 
**tampering**—a management failure that amplifies system instability.

This module delivers mathematically rigorous, publication-grade SPC architectures tailored for NHS England and international 
clinical quality improvement frameworks, featuring Shewhart control limits, Laney $p'$ and $u'$ overdispersion corrections for large trusts, 
and the **NHS Making Data Count** icon taxonomy.
""")

# 1. Plain English Zero-Knowledge Guide
render_spc_plain_english_guide()

st.markdown("---")

# 2. Navigation Mode Selector & Execution Controller
nav_mode, selected_method_key = render_spc_mode_selector()

# 3. Comprehensive Method Module Execution
render_spc_method_module(selected_method_key)
