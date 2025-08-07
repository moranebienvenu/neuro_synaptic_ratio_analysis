import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy.stats import ttest_ind, ttest_rel, levene, mannwhitneyu, shapiro, wilcoxon, norm
import io
import zipfile
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import colorsys
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Tweedie, Gaussian, Gamma, Poisson
from statsmodels.genmod.families.links import Log, Identity, InversePower, Sqrt, NegativeBinomial
from nilearn import plotting
import seaborn as sns
from statsmodels.stats.power import TTestIndPower, TTestPower
from itertools import combinations
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from io import BytesIO
from functions import process_zip_to_dataframe

st.markdown(
    """
    <style>
    .main {
        background-color: #e6f1fb;
    }
    .custom-container {
        background-color: #eaf4fc;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
    }

    .custom-title {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 46px;
        font-weight: 600;
        color: #1f4e79;
        margin-bottom: 5px;
        border-bottom: 3px solid transparent;
        display: inline-block;
        animation: underline-slide 3s ease-in-out infinite alternate;
    }

    @keyframes underline-slide {
        0%   { border-color: #1f4e79; }
        100% { border-color: #82c0ff; }
    }

    .custom-subtitle {
        font-size: 20px;
        color: #1f4e79;
        margin-top: 5px;
        font-weight: 400;
    }
    </style>

    <div class="custom-container">
        <div class="custom-title">
            🧠 NeuroClinical Dashboard
        </div>
        <div class="custom-subtitle">
            Explore neurotransmitter ratios and their clinical relevance
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )

# 1. Upload
uploaded_zip = st.file_uploader(
    "📦 **Upload a ZIP file containing all patient CSVs obtained with NeuroTmap.py and a clinical_data file** ",
    type=["zip"]
)

st.caption(
    """
    The ZIP file must include:
    - One or more `output_les_dis_sub-XXX_ses-VX.csv` files
    - One or more `output_pre_post_synaptic_ratio_sub-XXX_ses-VX.csv` files
    - Optional one `clinical_data.csv` or `.xlsx` file  
    This clinical file **must include a `subject` column** matching the filenames, and can include other variables like:
    `sex`, `timepoint`, `repetition_score`, `comprehension_score`, `naming_score`, `composite_score`,  `lesion_volume`
    Lesion volume must be in mm3.
    """
)

if uploaded_zip is not None:
    st.session_state.uploaded_zip = uploaded_zip
    with st.spinner("⏳ Processing..."):
        df_combined = process_zip_to_dataframe(uploaded_zip)
        st.session_state['df_combined'] = df_combined
    if not df_combined.empty:
        st.success("✅ All data combined successfully!")
        if st.checkbox("Show full combined dataset"):
            st.dataframe(df_combined)  
    else:
        st.warning("❌ No data could be combined. Please check the filenames or their contents.")

# 2. Page selection
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Neurotransmitters imbalance Visualisation"):
        st.session_state.page = "visualisation"
        st.rerun()

with col2:
    if st.button("📈 Statistical Analysis"):
        st.session_state.page = "stats"
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
   