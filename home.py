import streamlit as st
from functions import process_zip_to_dataframe

st.markdown(
    """
    <style>
    .home-container {
        text-align: center;
        padding: 50px;
        background-color: #f0f6fc;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    .home-title {
        font-size: 48px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 10px;
    }
    .home-subtitle {
        font-size: 20px;
        color: #4f6d7a;
        margin-bottom: 40px;
    }
    .stButton>button {
        font-size: 18px;
        padding: 0.75em 2em;
        margin: 1em;
        border-radius: 10px;
        background-color: #1f4e79;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2873a3;
    }
    </style>
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
    with st.spinner("⏳ Processing..."):
        st.session_state['uploaded_zip'] = uploaded_zip
        df_combined = process_zip_to_dataframe(uploaded_zip)
        st.session_state['df_combined'] = df_combined
    if not df_combined.empty:
        st.success("✅ All data combined successfully!")
        if st.checkbox("Show full combined dataset"):
            st.dataframe(df_combined)  
    else:
        st.warning("❌ No data could be combined. Please check the filenames or their contents.")

# 2. Page selection
with st.container():
    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.markdown('<div class="home-title">🧠 NeuroClinical Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-subtitle">Explore neurotransmitter ratios and their clinical relevance</div>', unsafe_allow_html=True)

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