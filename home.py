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
#from functions import process_zip_to_dataframe

def process_zip_to_dataframe(uploaded_zip):
    """Lit un ZIP NeuroTmap avec tous les fichiers csv "output_les_dis_'lesion_name'" et " output_pre_post_synaptic_ratio_'lesion_name'
    et retourne un DataFrame combiné sujet par sujet 
    Si présence d'un fichier csv ou excel nommé "clinical_data.csv/xlsx" alors concatener aussi sur le nom des sujets au fichier global"""

    # Dictionnaires de DataFrame par type
    data_les_dis = {}
    data_pre_post = {}
    df_clinical = None
    missing_clinical_subjects = []
    systems_list = []

    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
        for filename in z.namelist():
            # ⛔️ Ignorer tous les fichiers systèmes ou non pertinents
            if (
                "__MACOSX" in filename 
                or filename.endswith(".DS_Store") 
                or "/._" in filename 
                or (not "output_les_dis" in filename and not "output_pre_post_synaptic_ratio" in filename and "clinical_data" not in filename)
            ):
                continue

            if filename.endswith(".csv") or filename.endswith(".xlsx"):
                # Cas du fichier clinique
                if "clinical_data" in filename:
                    with z.open(filename) as f:
                        try:
                            if filename.endswith(".csv"):
                                content = f.read().decode("utf-8")
                                if "," in content and " " in content:
                                    df_clinical = pd.read_csv(io.StringIO(content), sep=" ", decimal=",")
                                else:
                                    df_clinical = pd.read_csv(io.StringIO(content))
                            else:
                                df_clinical = pd.read_excel(f, engine="openpyxl")
                            st.success("✅ Clinical file loaded")
                        except Exception as e:
                            st.warning(f"❌ Error loading clinical file: {e}")
                    continue  


            # Extraction du subject : sub-XXX_ses-VX
            if "output_les_dis" in filename or "output_pre_post_synaptic_ratio" in filename:
                match = re.search(r"sub-[A-Za-z0-9]+_ses-V[0-9]+", filename)
                if not match:
                    st.warning(f"❌ Subject name not recognized in: {filename}")
                    continue

            subject = match.group(0)

            with z.open(filename) as f:
                try:
                    df = pd.read_csv(f, sep=' ', index_col=0)
                except Exception as e:
                    st.warning(f"❌ Read error {filename}: {e}")
                    continue

            # Classement par type
            if "output_les_dis" in filename:
                # On attend que df ait une index colonne et des colonnes systèmes
                        # Par exemple, index = ['loc_inj_sub-A1503_ses-V2', ...]
                row = {"subject": subject}
                # Détection des systèmes (colonnes du fichier)
                if not systems_list and len(df.columns) > 0:
                    systems_list = list(df.columns)
                for idx in df.index:
                    prefix = str(idx).split('_sub-')[0]  # ex: loc_inj 
                    for system in df.columns:
                        colname = f"{prefix}_{system}"
                        row[colname] = df.at[idx, system]
                data_les_dis[subject] = pd.DataFrame([row])  

            elif "output_pre_post_synaptic_ratio" in filename:
                 # Ici, on va parcourir les colonnes, qui sont de la forme "A4B2 presynaptic", "VAChT postsynaptic", etc.
                        # On crée un dictionnaire pour la ligne du sujet
                        row = {"subject": subject}

                        for col in df.columns:
                            # Ex: "A4B2 presynaptic" -> pre_A4B2
                            # Ex: "VAChT postsynaptic" -> post_VAChT
                            m = re.match(r"(.+?)\s+(presynaptic|postsynaptic)", col)
                            if m:
                                system_name = m.group(1).strip()
                                syn_type = m.group(2).strip()
                                prefix = "pre" if syn_type == "presynaptic" else "post"
                                new_colname = f"{prefix}_{system_name}"
                                val = df[col].iloc[0] if not df.empty else None
                                row[new_colname] = val
                        data_pre_post[subject] = pd.DataFrame([row])
    
            

    # Fusion sujet par sujet
    combined_rows = []
    all_subjects = set(data_les_dis.keys()) | set(data_pre_post.keys()) 

    for subject in sorted(all_subjects):
        row = {"subject": subject}

        if subject in data_les_dis:
            try:
                row.update(data_les_dis[subject].iloc[0].to_dict())
            except Exception as e:
                st.warning(f"❌ Data extraction error for les_dis {subject}: {e}")

        # Ajouter les données pre_post
        if subject in data_pre_post:
            try:
                row.update(data_pre_post[subject].iloc[0].to_dict())
            except Exception as e:
                st.warning(f"❌ Data extraction error for pre_post {subject}: {e}")

        # Ajouter les données cliniques si disponibles
        if df_clinical is not None:
            match_row = df_clinical[df_clinical['subject'] == subject]
            if not match_row.empty:
                row.update(match_row.iloc[0].to_dict())
            else:
                missing_clinical_subjects.append(subject)

        combined_rows.append(row)
    # Création du DataFrame final
    final_df = pd.DataFrame(combined_rows)
    
    # Réorganisation des colonnes pour avoir un ordre logique
    if systems_list:
        ordered_columns = ['subject']
        
        # Ajouter les colonnes les_dis dans l'ordre: loc_inj, loc_inj_perc, tract_inj, tract_inj_perc pour chaque système
        for system in systems_list:
            for measure in ['loc_inj', 'loc_inj_perc', 'tract_inj', 'tract_inj_perc']:
                colname = f"{measure}_{system}"
                if colname in final_df.columns:
                    ordered_columns.append(colname)
        
        # Ajouter les colonnes pre/post a la suite des colonnes précédentes
        pre_post_columns = [col for col in final_df.columns if col.startswith(('pre_', 'post_')) and col not in ordered_columns]
        ordered_columns.extend(sorted(pre_post_columns))
        
        # Ajouter les colonnes cliniques restantes
        other_columns = [col for col in final_df.columns if col not in ordered_columns and col != 'subject']
        ordered_columns.extend(other_columns)
        
        final_df = final_df[ordered_columns]
        if missing_clinical_subjects:
                st.info("ℹ️ No clinical data found for the following subjects: " + ", ".join(missing_clinical_subjects))

    return final_df

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
   