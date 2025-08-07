import streamlit as st
from functions import get_family_and_link, heck_model_assumption, safe_glm_crossgroup, perform_group_comparison,clean_groups_for_variable, get_correlation_matrix, extract_subject_id, clean_predictor_name, process_zip_to_dataframe,get_subjects_and_title, detect_group
import pandas as pd
import plotly.express as px
import plotly.io as pio
from scipy.stats import ttest_ind, ttest_rel, levene, mannwhitneyu, shapiro, wilcoxon, norm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Tweedie, Gaussian, Gamma, Poisson
from statsmodels.genmod.families.links import Log, Identity, InversePower, Sqrt
from itertools import combinations
from scipy.stats import shapiro, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import multipletests
from io import BytesIO
import io
import zipfile
import re
import os

st.set_page_config(page_title="Statistical Analysis", layout="wide")

if 'uploaded_zip' in st.session_state:
    uploaded_zip = st.session_state['uploaded_zip']
else:
    uploaded_zip = None

if 'df_combined' in st.session_state:
    df_combined = st.session_state['df_combined']
else:
    df_combined = None

# Puis tu peux utiliser ces variables normalement
if uploaded_zip is not None:
    st.write("ZIP file is loaded")


   
    # st.header("📈 Statistical Analysis")

    analysis_method = st.selectbox(
        "Choose analysis method:",
        options=["GLM", "T-Test", "Correlation"]
    )

    if analysis_method == "GLM":
        dist = st.selectbox("Select distribution:", ["Gaussian", "Gamma", "Poisson", "Tweedie"])
        if dist == "Tweedie":
            var_power = st.number_input("Variance power", min_value=0.0, max_value=3.0, value=1.5, step=0.1)
  
        interaction = st.checkbox("Include interaction", value=False)
        fam_to_links = {
            "Gaussian": ["identity", "log", "inverse"],
            "Gamma": ["identity", "log", "inverse"],
            "Poisson": ["log", "identity"],
            "Tweedie": ["log", "identity", "inverse"],
        }
        link_map = {
            "log": Log(),
            "identity": Identity(),
            "inverse": InversePower(),
            "sqrt": Sqrt()
        }
        valid_links = fam_to_links[dist]
        link = st.selectbox("Select link function:", options=valid_links)
        selected_link = link_map[link]

    elif analysis_method == "T-Test":
        st.markdown("T-Test options will be handled automatically based on normality and variance homogeneity.")

    elif analysis_method == "Correlation":
        st.markdown("Correlation test will be chosen automatically (Pearson if normal, Spearman otherwise).")

    # ==== Section filtre sur les sujets ====
    if analysis_method == "Correlation":
        analysis_type = st.radio(
                    "Analysis type:",
                    ["By session and sex", "Personalized subject list"],
                    horizontal=True
                )
    else : 
        analysis_type = st.radio(
                    "Analysis type:",
                    ["By session and sex", "Combined filter", "Personalized subject list"],
                    horizontal=True
                )
        
    if analysis_type != "Personalized subject list":

        #bug avec "sex_bin" quand include interaction
        if analysis_method=="GLM":
            subjects, title, sex_filter, session = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=[],
                is_overlay=False,
                context="stats"
            )
            subject_groups = {
                    subj: detect_group(subj) for subj in subjects
                }
                
            available_groups = sorted(set(subject_groups.values()))
            selected_groups = st.multiselect(
                "Filter by subject group:",
                options=available_groups,
                default=available_groups,
                key="group_filter_2"
            )
            # Filtrer les sujets en fonction des groupes sélectionnés
            subjects = [subj for subj in subjects if subject_groups[subj] in selected_groups]

        elif analysis_method == "T-Test": 
            st.write("#### Group 1 Selection")
            # Sélection du groupe 1
            group1_subjects, group1_title, _, _ = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=[],
                is_overlay=False,
                context="stats_group1"
            )
            
            # Filtre par groupe pour le groupe 1
            group1_subject_groups = {
                subj: detect_group(subj) for subj in group1_subjects
            }
            available_groups_group1 = sorted(set(group1_subject_groups.values()))
            selected_groups_group1 = st.multiselect(
                "Filter Group 1 by subject group:",
                options=available_groups_group1,
                default=available_groups_group1,
                key="group_filter_group1"
            )
            group1_subjects = [subj for subj in group1_subjects if group1_subject_groups[subj] in selected_groups_group1]
            
            st.write("---")
            st.write("#### Group 2 Selection")
            # Sélection du groupe 2 (en excluant les sujets déjà sélectionnés dans le groupe 1)
            group2_subjects, group2_title, _, _ = get_subjects_and_title(
                df=df_combined,
                analysis_type=analysis_type,
                existing_subjects=group1_subjects,  # Exclure les sujets du groupe 1
                is_overlay=False,
                context="stats_group2"
            )
            
            # Filtre par groupe pour le groupe 2
            group2_subject_groups = {
                subj: detect_group(subj) for subj in group2_subjects
            }
            available_groups_group2 = sorted(set(group2_subject_groups.values()))
            selected_groups_group2 = st.multiselect(
                "Filter Group 2 by subject group:",
                options=available_groups_group2,
                default=available_groups_group2,
                key="group_filter_group2"
            )
            group2_subjects = [subj for subj in group2_subjects if group2_subject_groups[subj] in selected_groups_group2]
            
            # Affichage du nombre de sujets sélectionnés
            st.write(f"Number of subjects in Group 1: {len(group1_subjects)}")
            st.write(f"Number of subjects in Group 2: {len(group2_subjects)}")
            
            # Vérification qu'il y a assez de sujets
            if len(group1_subjects) < 3 or len(group2_subjects) < 3:
                st.warning("Each group must contain at least 3 subjects for statistical comparison")

        elif analysis_method=="Correlation":
            # Colonnes pour la sélection
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Set 1 Configuration")
                group1_subjects, group1_title, _, session1 = get_subjects_and_title(
                    df=df_combined,
                    analysis_type=analysis_type,
                    existing_subjects=[],
                    is_overlay=False,
                    context="corr_group1"
                )

                group1_subject_groups = {
                    subj: detect_group(subj) for subj in group1_subjects
                }
                available_groups_group1 = sorted(set(group1_subject_groups.values()))
                selected_groups_group1 = st.multiselect(
                    "Filter Group 1 by subject group:",
                    options=available_groups_group1,
                    default=available_groups_group1,
                    key="group_filter_group1_corr"
                )
                group1_subjects = [subj for subj in group1_subjects if group1_subject_groups[subj] in selected_groups_group1]

            with col2:
                st.markdown("#### 🔴 Set 2 Configuration")
                group2_subjects, group2_title, _ , session2= get_subjects_and_title(
                    df=df_combined,
                    analysis_type=analysis_type,
                    existing_subjects=[],
                    is_overlay=False,
                    context="corr_group2"
                )

                group2_subject_groups = {
                    subj: detect_group(subj) for subj in group2_subjects
                }
                available_groups_group2 = sorted(set(group2_subject_groups.values()))
                selected_groups_group2 = st.multiselect(
                    "Filter Group 2 by subject group:",
                    options=available_groups_group2,
                    default=available_groups_group2,
                    key="group_filter_group2_corr"
                )
                group2_subjects = [subj for subj in group2_subjects if group2_subject_groups[subj] in selected_groups_group2]

    else:
        # Choix manuel de plusieurs sujets parmi tous (pas de filtre session/sex)
        st.caption(
    """
    The subject selection must include at least 3 subjects by group"""
    )   
        if analysis_method=="GLM":
            all_subjects = sorted(df_combined['subject'].unique())
            selected_subjects = st.multiselect(
                "Select personalized subjects:",
                options=all_subjects,
                key="personalized_subjects_select"
            )
            subjects = selected_subjects

        elif analysis_method=="Correlation":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Set 1 Configuration")
                all_subjects = sorted(df_combined['subject'].unique())
                # Sélection Groupe 1 (sans exclusion)
                group1_subjects = st.multiselect(
                    "Select group 1 subjects",
                    options=all_subjects,
                    key="corr_group1_select"
                )
                session1 = "combined"
            with col2:
                st.markdown("#### 🔴 Set 2 Configuration")  
                all_subjects = sorted(df_combined['subject'].unique())  
                # Sélection Groupe 2 (PEUT inclure les mêmes sujets que groupe 1)
                group2_subjects = st.multiselect(
                    "Select group 2 subjects",
                    options=all_subjects,  # Pas de filtrage des sujets du groupe 1
                    key="corr_group2_select"
                )
                session2 = "combined"
        
    if analysis_type == "By session and sex" and sex_filter == "All":
        # Supprimer les colonnes liées au sexe si elles existent sinon échec de GLM
        df_combined = df_combined.drop(columns=[col for col in ["sex", "Sexe_bin"] if col in df_combined.columns])

   
    if analysis_method == "GLM":
        # Initialisation du session state
        if 'glm_stat' not in st.session_state:
            st.session_state.glm_stat = {
                'ran': False,
                'results': None,
                'variables': {
                    'outcomes': [],
                    'covariates': [],
                    'predictors': []
                },
                'plot_config': {
                    'selected_var': None,
                    'show_points': True,
                    'color_by': "None",
                    'figure': None,
                    'last_run_variable': None
                },
                'analysis_done': False,
                'data_ready': False
            }

        # Filtrer le dataframe avec la liste des sujets sélectionnés
        df_filtered = df_combined[df_combined['subject'].isin(subjects)]
        # Conversion du sexe en binaire 
        sex_col = next((col for col in ['Sexe', 'sex', 'gender', 'Sex', 'sexe'] if col in df_filtered.columns), None)
        if sex_col and 'Sexe_bin' not in df_filtered.columns:
            sex_mapping = {
                'm': 0, 'male': 0, 'homme': 0, 'man': 0, 'men': 0,
                'f': 1, 'female': 1, 'femme': 1, 'woman': 1, 'women': 1
            }
            df_filtered['Sexe_bin'] = df_filtered[sex_col].astype(str).str.strip().str.lower().map(sex_mapping)
            
            if df_filtered['Sexe_bin'].isna().any():
                st.warning("Certaines valeurs de sexe n'ont pas pu être converties")

        st.write(f"Filtered subjects count: {len(subjects)}")
        st.write(" Data filtered before the model:", df_filtered)
        st.write("DataFrame size:", df_filtered.shape)
        
        # Re-définir la liste des systèmes 
        systems = [col.replace('loc_inj_', '') for col in df_combined.columns 
                if col.startswith('loc_inj_') and not col.startswith('loc_inj_perc')]

        # Séparer les colonnes cliniques (variables dépendantes) 
        clinical_cols = [col for col in df_combined.columns if col not in ['subject','Sexe_bin','sex']
                        and not any(sys in col for sys in systems)]

        # Séparer les prédicteurs liés aux systèmes
        system_predictors = [col for col in df_combined.columns
                            if any(sys in col for sys in systems)]

       
        # Variables : Pre/Post-synaptic ratios
        pre_post_vars = [var for var in [
            "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
            "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
            "pre_5HT4", "pre_5HT6",
            "post_VAChT", "post_DAT", "post_5HTT"
        ] if var in system_predictors]

        # Variables : neurotransmitter systems (local & tract)
        nt_systems_vars_loc = [var for var in [
            "loc_inj_GABAa", "loc_inj_mGluR5", 
            "loc_inj_MU", "loc_inj_H3",
            "loc_inj_CB1",  "loc_inj_A4B2", 
            "loc_inj_M1",  "loc_inj_VAChT", 
            "loc_inj_D1", "loc_inj_D2", 
            "loc_inj_DAT",  "loc_inj_Nor", 
            "loc_inj_5HT1a",  "loc_inj_5HT1b", 
            "loc_inj_5HT2a",  "loc_inj_5HT4", 
            "loc_inj_5HT6", "loc_inj_5HTT"
        ] if var in system_predictors]

        nt_systems_vars_tract= [var for var in [
            "tract_inj_GABAa", "tract_inj_mGluR5",
            "tract_inj_MU", "tract_inj_H3",
            "tract_inj_CB1", "tract_inj_A4B2",
            "tract_inj_M1", "tract_inj_VAChT",
            "tract_inj_D1", "tract_inj_D2",
            "tract_inj_DAT",  "tract_inj_Nor",
            "tract_inj_5HT1a",  "tract_inj_5HT1b",
            "tract_inj_5HT2a",  "tract_inj_5HT4",
            "tract_inj_5HT6",  "tract_inj_5HTT"
        ] if var in system_predictors]


        # Partie 2 : Interface utilisateur pour la GLM
        st.subheader("🧠 GLM Variable Selection")

        selected_outcomes = st.multiselect(
            "Select clinical outcomes to predict (dependent variables):",
            options=clinical_cols,
            key="outcomes_selection",
            help="Choose one or more clinical/behavioral outcomes from your dataset."
        )
        # Filtrer les covariables disponibles (exclure celles déjà sélectionnées comme outcomes)
        available_covariates = [col for col in clinical_cols if col not in selected_outcomes and col!="sex"]

        if analysis_type == "By session and sex" :
            available_covariates=available_covariates
        else: 
            if 'Sexe_bin' not in available_covariates and 'Sexe_bin' in df_filtered.columns:
                available_covariates.append('Sexe_bin')

        selected_covariates = st.multiselect(
        "Select covariates for GLM:",
        options=available_covariates,
        default=[],
        key="covariate_selection"
        )

        system_options = {
            "Synaptic ratio": pre_post_vars,
            "Neurotransmitter (Loc)": nt_systems_vars_loc,
            "Neurotransmitter (Tract)": nt_systems_vars_tract
        }

        selected_system = st.radio(
            "Choose the predictor system:",
            list(system_options.keys()),
            key="system_selector"
        )

        # Récupération des variables associées
        selected_predictor = system_options[selected_system]

        interaction_vars = ['Sexe_bin', 'Group', None]
        if interaction: 
            selected_interaction = st.selectbox(
            "Select variable for interaction with predictors (or None):",
            options=interaction_vars,
            index=0,
            key="interaction_choice"
        )
        
        if dist == "Tweedie":
            var_power = var_power
            family = Tweedie(var_power=var_power, link=selected_link)
        elif dist == "Gamma":
            family = Gamma(link=selected_link)
        elif dist == "Poisson":
            family = Poisson(link=selected_link)
        else:
            family = Gaussian(link=selected_link)

        for outcome in selected_outcomes:
            outcome_values = df_filtered[outcome].dropna()
            if (outcome_values <= 0).any() and dist in ["Gamma", "Poisson"]:
                st.warning(f"⚠️ La variable {outcome} contient des valeurs négatives ou nulles - inapproprié pour {dist}")
            if (outcome_values % 1 != 0).any() and dist == "Poisson":
                st.warning(f"⚠️ La variable {outcome} contient des valeurs non entières - inapproprié pour Poisson")
        st.markdown("---")
        run_glm = st.button("🚀 Run GLM on selected outcomes")
        if run_glm:
            st.session_state.glm_stat['run_triggered'] = True
            st.session_state.glm_stat.update({
                    'variables': {
                        'outcomes': selected_outcomes,
                        'covariates': selected_covariates,
                        'predictors': selected_predictor
                    },
                    'ran': True,
                    'analysis_done': True,
                    'data_ready': True
                })
            st.session_state.df_filtered_glm = df_filtered.copy()

        if st.session_state.glm_stat.get('analysis_done') and st.session_state.glm_stat.get('data_ready'):
            df_filtered = st.session_state.get('df_filtered_glm', df_combined.copy())
            # Vérification préalable
            if len(subjects) < 3:
                st.error("⚠️ Vous devez sélectionner au moins 3 sujets pour l'analyse")
                st.stop()
                
            if not selected_outcomes:
                st.error("⚠️ Veuillez sélectionner au moins une variable dépendante")
                st.stop()
            if not selected_predictor:
                st.error("⚠️ Veuillez sélectionner au moins une variable indépendante")
                st.stop()
            previous_selected_var = None
            previous_show_points = None
            previous_color_by = None

            # Afficher un aperçu des données
            st.write("Preview of selected data:")
            st.write(df_filtered[selected_outcomes + selected_covariates + selected_predictor].describe())
        
            
            # Liste des variables numériques
            glm_config = st.session_state.glm_stat['plot_config']

            # Liste des variables numériques disponibles
            all_numeric_vars = [
                col for col in selected_outcomes + selected_covariates + selected_predictor
                if pd.api.types.is_numeric_dtype(df_filtered[col])
            ]

            if not all_numeric_vars:
                st.warning("⚠️ No numeric variable available for display.")
                st.stop()

            st.subheader("📊 Distribution of selected variables")

            selected_var = st.selectbox(
                "Choose a variable to visualize:",
                options=all_numeric_vars,
                index=all_numeric_vars.index(previous_selected_var) if previous_selected_var in all_numeric_vars else 0,
                key="glm_selected_var_box"
            )

            col1, col2 = st.columns(2)
            with col1:
                show_points = st.checkbox(
                    "Show individual points",
                    value=previous_show_points,
                    key="glm_show_points"
                )

            with col2:
                color_options = ["None"] + [col for col in ['Sexe_bin', 'Group'] if col in df_filtered.columns]               
                color_by = st.selectbox(
                    "Color by:",
                    options=color_options,
                    index=color_options.index(previous_color_by) if previous_color_by in color_options else 0,
                    key="glm_color_by"
                )
            
            previous_selected_var = glm_config.get('selected_var')
            previous_show_points = glm_config.get('show_points')
            previous_color_by = glm_config.get('color_by')

      
            if (
                selected_var != previous_selected_var
                or show_points != previous_show_points
                or color_by != previous_color_by
                or glm_config['figure'] is None
            ):

                if color_by != "None" and color_by in df_filtered.columns:
                    fig = px.box(
                        df_filtered,
                        y=selected_var,
                        x=color_by if len(df_filtered[color_by].unique()) > 1 else None,
                        color=color_by,
                        title=f"Distribution of {selected_var} by {color_by}",
                        points="all" if show_points else None,
                        color_discrete_map={0: '#3498db', 1: '#e74c3c'} if color_by == 'Sexe_bin' else None
                    )
                else:
                    fig = px.box(
                        df_filtered,
                        y=selected_var,
                        title=f"Distribution of  {selected_var}",
                        points="all" if show_points else None,
                        color_discrete_sequence=['#3498db']
                    )

                fig.update_layout(hovermode="x unified")
                glm_config['figure'] = fig
                glm_config['last_run_variable'] = selected_var
                glm_config['selected_var'] = selected_var
                glm_config['show_points'] = show_points
                glm_config['color_by'] = color_by

            # Afficher la figure
            if glm_config['figure']:
                st.plotly_chart(glm_config['figure'], use_container_width=True)
                st.session_state.glm_stat['run_triggered'] = False
            
   
            systems_mapping = {"Selected System": selected_predictor}
            with st.spinner('Running GLM models...'):
                glm_results = safe_glm_crossgroup(
                    df_predictors=df_filtered,
                    df_outcomes=df_filtered,
                    outcomes=selected_outcomes,
                    systems=systems_mapping,
                    covariate=selected_covariates,
                    visit_name=title if 'title' in locals() else "GLM Run",
                    interaction_var=selected_interaction if interaction else None,
                    family=family, 
                )
            if glm_results.empty:
                st.error("""
                ❌ No GLM results obtained. Possible causes:
                   Missing data (NaN) in the selected variables
                   Too few observations after cleaning
                   Model convergence issues
                   Inappropriate family/link function for your data
                """)
        
            else:
                st.subheader("📊 GLM Results")
                st.dataframe(glm_results)
                # Visualisation interactive des résultats
                st.subheader("📈 Visualization of results")
                
                # Couleurs pour les systèmes
                base_colors = {
                    'A4B2': '#76b7b2',
                    'M1': '#59a14f',
                    'VAChT': '#edc948',
                    'D1': '#b07aa1',
                    'D2': '#ff9da7',
                    'DAT': '#9c755f',
                    'Nor': '#79706e',
                    '5HT1a': '#86bcb6',
                    '5HT1b': '#d95f02',
                    '5HT2a': '#e7298a',
                    '5HT4': '#66a61e',
                    '5HT6': '#e6ab02',
                    '5HTT': '#a6761d',
                }
           
                prefixes_pre = ['pre_', 'loc_inj_', 'tract_inj_']
                prefixes_post = ['post_', 'loc_inj_', 'tract_inj_']

                keys_pre = ['A4B2', 'M1', 'D1', 'D2', 'Nor', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
                keys_post = ['VAChT', 'DAT', '5HTT']
                neuro_colors = {}

                for key, color in base_colors.items():
                    neuro_colors[key] = color

                for key in keys_pre:
                    for prefix in prefixes_pre:
                        neuro_colors[prefix + key] = base_colors[key]

                for key in keys_post:
                    for prefix in prefixes_post:
                        neuro_colors[prefix + key] = base_colors[key]
                
                for outcome in glm_results['Outcome'].unique():
                    outcome_data = glm_results[glm_results['Outcome'] == outcome].copy()
                    outcome_data['Clean_Predictor'] = outcome_data['Predictor'].apply(clean_predictor_name)
                    outcome_data = outcome_data[outcome_data['Clean_Predictor'].isin(neuro_colors.keys())]

                    if outcome_data.empty:
                        st.write(f"No relevant predictors to display for outcome {outcome}.")
                        continue

                    fig = go.Figure()

                    for _, row in outcome_data.iterrows():
                        predictor = row['Clean_Predictor']
                        color = neuro_colors.get(predictor, '#1f77b4')

                        fig.add_trace(go.Bar(
                            x=[predictor],
                            y=[row['Coefficient']],
                            name=predictor,
                            marker_color=color,
                            text=[f"p={row['P-value']:.3f}"],
                            textposition='auto',
                            hoverinfo='text',
                            hovertext=(
                                f"Predictor: {predictor}<br>"
                                f"Coefficient: {row['Coefficient']:.3f}<br>"
                                f"p-value: {row['P-value']:.3f}"
                            )
                        ))

                        if row['Significant']:
                            fig.add_annotation(
                                x=predictor,
                                y=row['Coefficient'],
                                text="*",
                                showarrow=False,
                                font=dict(size=20, color='black'),
                                yshift=10
                            )

                    fig.update_layout(
                        title=f"GLM Results for {outcome}",
                        xaxis_title="Predictors",
                        yaxis_title="Coefficient",
                        barmode='group',
                        showlegend=False,
                        hovermode='closest',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # ✅ Bouton de téléchargement Excel
                #st.subheader("📥 GLM Results Export")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    glm_results.to_excel(writer, sheet_name='GLM_Results', index=False)

                st.download_button(
                    label="📥 Download results in Excel format",
                    data=output.getvalue(),
                    file_name=f"glm_results_{selected_system.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


   
    #quand je choisis deux groupes de sujets avec les memes base - pas mme version pour test independant ca marche quand emme ///rajouter une condition pour base-subject
    #sinon l'utilisateur est censé le faire de lui meme ce choix -- modification secondaire
    elif analysis_method == "T-Test":
        st.subheader("T-Test Configuration")
        if 'ttest' not in st.session_state:
            st.session_state.ttest = {
                'ran': False,
                'results_df': None,
                'cleaned_data': None,
                'df1_clean': None,
                'df2_clean': None,
                'variables': None,
                'paired': False,
                'analysis_done': False,
                'data_ready_ttest': False,
                'active_ttest_tab': "Statistic Results",
                'plot_config': {
                    'plot_type': "Violin plot",
                    'selected_var': None,
                    'figure': None,
                    'last_run_variable': None,
                    'last_run_plot_type': None
                    }
            }
           
        # Ajouter le choix du type de test
        paired_test = st.checkbox(
            "Paired data (same subjects in both groups)", 
            value=False,
            key="paired_test"
        )
        
        if analysis_type == "Personalized subject list":
            all_subjects = sorted(df_combined['subject'].unique())
            
            # Groupe 1 
            group1_subjects = st.multiselect(
                "Select group 1 subjects",
                options=all_subjects, 
                key="group1_select"
            )
            #Groupe 2 -- verifier que choix de session possible et juste base du sujet qui revient --no a modifier
            if paired_test:
                # Mode apparié - mêmes sujets
                if group1_subjects:
                    # Extraire les identifiants de base (sans le suffixe avec la version)
                    base_ids = [subj.split('-V')[0] for subj in group1_subjects]
                    available_for_pairing = []
                    for base in base_ids:
                        available_for_pairing.extend([s for s in all_subjects if s.startswith(base)])
                    group2_subjects = st.multiselect(
                        "Select matching subjects for group 2",
                        options=list(set(available_for_pairing)),  # Éviter les doublons
                        default=group1_subjects,
                        key="group2_select_paired"
                    )
                else:
                    group2_subjects = []
            else:
                # Mode indépendant - sujets différents
                available_group2 = [s for s in all_subjects if s not in group1_subjects]
                group2_subjects = st.multiselect(
                    "Select group 2 subjects",
                    options=available_group2,
                    key="group2_select_independent"
                )

        elif analysis_type == "By session and sex":
            if paired_test:
                # Extraire les identifiants de base (sans le suffixe -V1/-V2)
                base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
                base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
                
                # Trouver les bases communes aux deux groupes
                common_base_ids = base_ids_group1 & base_ids_group2
                
                if not common_base_ids:
                    st.error("No matching subject pairs found between the two groups")
                    st.stop()
                
                # Filtrer les sujets pour ne garder que ceux dont la base est commune
                group1_subjects = [subj for subj in group1_subjects 
                                if subj.split('-V')[0] in common_base_ids]
                group2_subjects = [subj for subj in group2_subjects 
                                if subj.split('-V')[0] in common_base_ids]
                
                # Vérifier qu'on a au moins 3 paires
                if len(common_base_ids) < 3:
                    st.warning(f"Only {len(common_base_ids)} paired subjects found (minimum 3 required)")
                
                # Afficher le résultat du matching
                st.success(f"Found {len(common_base_ids)} valid subject pairs for paired analysis")
            elif not paired_test:
                base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
                base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
                overlapping_bases = base_ids_group1 & base_ids_group2

                if overlapping_bases:
                    st.error(f"Independent test requires different subjects – overlapping base IDs found: {', '.join(sorted(overlapping_bases))}")
                    st.stop()
        else:  # Pour By session/sex ou Combined filter
            if paired_test:
                st.warning("Paired test requires explicit subject pairing - use 'Personalized subject list' or 'By session and sex' mode")
                st.stop()
                
        #Vérifications finales
        if not group1_subjects or not group2_subjects:
            st.warning("Please select subjects for both groups")
            st.stop()
            
        if paired_test:
            if len(group1_subjects) != len(group2_subjects):
                st.error("Paired test requires same number of subjects in both groups")
                st.stop()
                
            # Vérifier que ce sont bien les mêmes sujets (versions différentes possibles)
            base_ids_group1 = {subj.split('-V')[0] for subj in group1_subjects}
            base_ids_group2 = {subj.split('-V')[0] for subj in group2_subjects}
            
            if base_ids_group1 != base_ids_group2:
                st.error("Paired test requires matching subject IDs (only version suffix should differ)")
                st.stop()        

        df_group1 = df_combined[df_combined['subject'].isin(group1_subjects)]
        df_group2 = df_combined[df_combined['subject'].isin(group2_subjects)]

        numeric_cols = df_group1.select_dtypes(include=[np.number]).columns.tolist()
        variables_to_compare = st.multiselect(
            "Select variables to compare:",
            options=numeric_cols,
            key="ttest_vars"
        )

        cleaned_data = {}
        if st.button("Run Statistical Comparison"):
            results = []
            
            for var in variables_to_compare:
                if var in df_group1.columns and var in df_group2.columns:
                    # Nettoyage des groupes pour cette variable
                    df1_clean, df2_clean, n_pairs = clean_groups_for_variable(df_group1, df_group2, var, paired_test)
                    cleaned_data[var] = (df1_clean, df2_clean)
                    # Vérification du nombre de sujets après nettoyage et suppression
                    if paired_test:
                        if n_pairs is None or n_pairs < 3:
                            st.warning(f"{var}: Only {n_pairs if n_pairs is not None else 0} valid subject pairs found (minimum 3 required). Skipping.")
                            continue
                        st.markdown(f"**{var}** – Found {n_pairs} valid subject pairs for paired analysis.")
                        all_removed_subjects = [] 
                    else:
                        removed_subjects_g1 = df_group1[df_group1[var].isna()]['subject'].tolist()
                        removed_subjects_g2 = df_group2[df_group2[var].isna()]['subject'].tolist()
                        all_removed_subjects = removed_subjects_g1 + removed_subjects_g2
                        if len(df1_clean) < 3 or len(df2_clean) < 3:
                            st.warning(f"{var}: Not enough valid subjects for independent analysis (≥3 per group). Skipping.") # dire cest quel subject qui a été retiré du à un none
                            if all_removed_subjects:
                                st.markdown(f"<span style='color:grey'>Removed due to missing values: {', '.join(all_removed_subjects)}</span>", unsafe_allow_html=True)
                                st.markdown(f"**{var}** – Number of valid subjects: Group 1 = {len(df1_clean)}, Group 2 = {len(df2_clean)}")
                            continue

                    if all_removed_subjects:
                        st.markdown(f"<span style='color:grey'>Removed due to missing values: {', '.join(all_removed_subjects)}</span>", unsafe_allow_html=True)


                    test_results = perform_group_comparison(
                        df1_clean[var],
                        df2_clean[var],
                        paired=paired_test  
                    )
                    if test_results:
                        test_results['variable'] = var  
                        results.append(test_results)

            
            if not results:
                st.error("No valid comparisons could be performed")
                st.stop()
                
            results_df = pd.DataFrame(results).reset_index(drop=True)
            # Supprime les colonnes dupliquées
            results_df = results_df.loc[:, ~results_df.columns.duplicated()]
            
            # Réorganisation des colonnes 
            cols_order = [
                'variable', 'test_type', 'p_value', 'significant',
                'mean_group1', 'mean_group2', 'effect_size', 'power',
                'n_group1', 'n_group2', 'statistic',
                'shapiro_p1', 'shapiro_p2'
            ]
            # Ajoutez 'levene_p' seulement si présent (sujets indépendants)
            if 'levene_p' in results_df.columns:
                cols_order.append('levene_p')
            cols_order = [c for c in cols_order if c in results_df.columns]

            # Stockage des résultats
            st.session_state.ttest.update({
                'results_df': results_df,
                'cleaned_data': cleaned_data,
                'df1_clean': df1_clean,
                'df2_clean': df2_clean,
                'variables': variables_to_compare,
                'paired': paired_test, 
                'analysis_done' : True,
                'data_ready_ttest': True,
                'ran': True,
            })


    
        # ---------  Affichage ------------
        if st.session_state.ttest.get('ran', False) and st.session_state.ttest['data_ready_ttest']:
            config = st.session_state.ttest['plot_config']
            tab_options = ["Statistic Results", "Visualization"]
            selected_tab = st.radio(
                "Navigation",
                tab_options,
                horizontal=True,
                index=tab_options.index(st.session_state.ttest['active_ttest_tab']),
                key='tab_selector_ttest',
                label_visibility="hidden"
            )
            st.session_state.ttest['active_ttest_tab'] = selected_tab

            #---------- Onglet 1 : Resultats T-Test ----------
            if selected_tab == "Statistic Results":
                st.subheader("T-Test Results")

                format_dict = {
                    'p_value': '{:.4f}',
                    'effect_size': '{:.3f}',
                    'power': '{:.3f}',
                    'mean_group1': '{:.3f}',
                    'mean_group2': '{:.3f}',
                    'shapiro_p1': '{:.4f}',
                    'shapiro_p2': '{:.4f}'
                }
                if 'levene_p' in st.session_state.ttest['results_df'].columns:
                    format_dict['levene_p'] = '{:.4f}'

                if st.session_state.ttest['results_df'].columns.is_unique and st.session_state.ttest['results_df'].index.is_unique:
                    st.dataframe(st.session_state.ttest['results_df'].style.format(format_dict).applymap(
                        lambda x: 'background-color: yellow' if isinstance(x, float) and x < 0.05 else '',
                        subset=[col for col in ['p_value', 'shapiro_p1', 'shapiro_p2', 'levene_p'] if col in st.session_state.ttest['results_df'].columns]
                    ))
                else:
                    st.warning("Colonnes ou index non uniques : affichage sans style.")
                    st.dataframe(st.session_state.ttest['results_df'])

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.ttest['results_df'].to_excel(writer, index=False, sheet_name='T-Test Results')

                st.download_button(
                    label="Download results as Excel",
                    data=output.getvalue(),
                    file_name="statistical_results.xlsx",
                    mime="applicat"
                    "ion/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            #---------- Onglet 2 : Visualisation ----------
            elif selected_tab == "Visualization":
        
                st.subheader("Group Comparison Plots")

                plot_type = st.radio(
                    "Choose plot type:",
                    ["Violin plot", "Box plot"],
                    index=["Violin plot", "Box plot"].index(config['plot_type']),
                    key="plot_type_radio"
                )

                selected_var = st.selectbox(
                    "Select variable to display:",
                    options=st.session_state.ttest['variables'],
                    index=st.session_state.ttest['variables'].index(config['selected_var']) if config['selected_var'] in st.session_state.ttest['variables'] else 0,
                    key="selected_var_box"
                )

                config['plot_type'] = plot_type
                config['selected_var'] = selected_var

                if (selected_var != config.get('last_run_variable')) or (plot_type != config.get('last_run_plot_type')):

                    if selected_var in st.session_state.ttest['cleaned_data']:
                        df1_clean, df2_clean = st.session_state.ttest['cleaned_data'][selected_var]

                        if not df1_clean.empty and not df2_clean.empty:
                            df_plot = pd.concat([
                                df1_clean[[selected_var]].assign(Group="Group 1"),
                                df2_clean[[selected_var]].assign(Group="Group 2")
                            ])

                            if plot_type == "Violin plot":
                                fig = go.Figure()
                                colors = ['#1f77b4', '#ff7f0e']
                                for i, group in enumerate(df_plot["Group"].unique()):
                                    fig.add_trace(go.Violin(
                                        y=df_plot[df_plot["Group"] == group][selected_var],
                                        name=group,
                                        box_visible=True,
                                        meanline_visible=True,
                                        points="all",
                                        line_color='black',
                                        fillcolor=colors[i]
                                    ))
                            else:
                                fig = px.box(
                                    df_plot,
                                    x="Group",
                                    y=selected_var,
                                    points="all",
                                    color="Group",
                                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                                )

                            fig.update_layout(
                                title=f"{selected_var} - Group Comparison",
                                height=500,
                                margin=dict(l=20, r=20, t=60, b=20),
                                showlegend=True
                            )

                            config['figure'] = fig
                            config['last_run_variable'] = selected_var
                            config['last_run_plot_type'] = plot_type
                        else:
                            st.warning("Not enough data to generate plot.")
                    else:
                        st.warning(f"No cleaned data found for variable: {selected_var}")

                if config['figure']:
                    st.plotly_chart(config['figure'], use_container_width=True)

           

    elif analysis_method =="Correlation":
        st.subheader("🔗 Correlation Analysis")

        if 'corr' not in st.session_state:
            st.session_state.corr = {
                'ran': False,
                'show_all': False,
                'p_thresh': 0.05,
                'active_tab': "Visualization",
                'data_ready': False,
                'data': {
                    'corr_matrix': None,
                    'pval_matrix': None,
                    'cross_corr': None, 
                    'cross_pvals': None,
                    'session1': "",
                    'session2': ""
                    }
            }

        # === Configuration des variables ===
        st.markdown("### Correlation Variables Configuration")
        df_filtered = df_combined[df_combined['subject'].isin(group1_subjects + group2_subjects)]

        # Types de variables disponibles
        system_options = {
            "Synaptic ratio": [
                "pre_A4B2", "pre_M1", "pre_D1", "pre_D2",
                "pre_5HT1a", "pre_5HT1b", "pre_5HT2a",
                "pre_5HT4", "pre_5HT6",
                "post_VAChT", "post_DAT", "post_5HTT"
            ],
            "Neurotransmitter (Loc)": [f"loc_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"loc_inj_{sys}" in df_filtered.columns],

            "Neurotransmitter (Tract)": [f"tract_inj_{sys}" for sys in [
                "GABAa", "mGluR5", "MU", "H3", "CB1", "A4B2", "M1", "VAChT",
                "D1", "D2", "DAT", "Nor", "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
            ] if f"tract_inj_{sys}" in df_filtered.columns],

            "Clinical Outcomes": [col for col in df_filtered.columns 
                        if col not in ['subject', 'Sexe_bin', 'sex', 'lesion_volume']
                        and not col.startswith(('loc_inj_', 'tract_inj_', 'pre_', 'post_'))]
        } 

        with col1:
            system_type1 = st.radio(
                "Variable type for Set 1:",
                options=list(system_options.keys()),
                key="system_type1"
            )
            vars1 = system_options[system_type1]
            
        with col2:
            system_type2 = st.radio(
                "Variable type for Set 2:",
                options=list(system_options.keys()),
                key="system_type2"
            )
            vars2 = system_options[system_type2]
      
        # === Exécution ===
        if st.button("🚀 Run Correlation Analysis", key="run_corr_button"):
            if not group1_subjects or not group2_subjects:
                st.error("Please select subjects for both groups")
                st.stop()

            if not vars1 or not vars2:
                st.error("Please select variables for both sets")
                st.stop()

            try:
                df_filtered["subject_base"] = df_filtered["subject"].apply(extract_subject_id)

                if session1 != "combined":
                    df1 = df_filtered[
                        (df_filtered["subject"].isin(group1_subjects)) &
                        (df_filtered["subject"].str.contains(f"_ses-{session1}"))
                    ][["subject_base"] + vars1].drop_duplicates()
                else:
                    df1 = df_filtered[
                        df_filtered["subject"].isin(group1_subjects)
                    ][["subject_base"] + vars1].drop_duplicates()

                if session2 != "combined":
                    df2 = df_filtered[
                        (df_filtered["subject"].isin(group2_subjects)) &
                        (df_filtered["subject"].str.contains(f"_ses-{session2}"))
                    ][["subject_base"] + vars2].drop_duplicates()
                else:
                    df2 = df_filtered[
                        df_filtered["subject"].isin(group2_subjects)
                    ][["subject_base"] + vars2].drop_duplicates()

                if session1 == "combined"or session2 == "combined":
                    suffix1 = "_set1"
                    suffix2 = "_set2"
                else:
                    suffix1 = f"_{session1}_1" if session1 == session2 else f"_{session1}"
                    suffix2 = f"_{session2}_2" if session1 == session2 else f"_{session2}"

                df1_renamed = df1.rename(columns={col: col + suffix1 for col in vars1})
                df2_renamed = df2.rename(columns={col: col + suffix2 for col in vars2})
                
                # Vérification des sujets communs
                common_ids = sorted(set(df1_renamed["subject_base"]) & set(df2_renamed["subject_base"]))
                if not len(common_ids) > 3:
                    st.error(f"Only {len(common_ids)} common subjects found (minimum 3 required)")
                    st.stop()
                
                # Fusion finale pour avoir un dataframe pour la matrice de correlation
                df_corr = df1_renamed[df1_renamed["subject_base"].isin(common_ids)].merge(
                    df2_renamed[df2_renamed["subject_base"].isin(common_ids)],
                    on="subject_base"
                )

                st.write(f"Final dataset shape: {df_corr.shape}")
                if df_corr.shape[0] < 3:
                    st.error("Not enough valid subjects after merging datasets")
                    st.stop()

                # Calcul des corrélations
                with st.spinner('Calculating correlations...'):
                    corr_matrix, pval_matrix = get_correlation_matrix(df_corr)

                    set1_cols = [col for col in corr_matrix.columns if col.endswith(suffix1)]
                    set2_cols = [col for col in corr_matrix.columns if col.endswith(suffix2)]
                    set1_cols_p=[col for col in pval_matrix.columns if col.endswith(suffix1)]
                    set2_cols_p = [col for col in pval_matrix.columns if col.endswith(suffix2)]
                    cross_corr = corr_matrix.loc[set1_cols, set2_cols]
                    cross_pvals = pval_matrix.loc[set1_cols_p, set2_cols_p]

            
                # préparation pour affichage
                st.success("Analysis completed!")
        
                st.session_state.corr.update({
                    'ran': True,
                    'data_ready': True,
                    'data': {
                        'corr_matrix': corr_matrix,
                        'pval_matrix': pval_matrix,
                        'cross_corr': cross_corr,
                        'cross_pvals': cross_pvals,
                        'session1': session1,
                        'session2': session2
                    }
                })
            except Exception as e:
                st.error(f"Error during correlation calculation: {str(e)}")
                st.session_state.corr['ran'] = False

        # ---------  Affichage ------------
        if st.session_state.corr.get('ran', False) and st.session_state.corr['data_ready']:
            data = st.session_state.corr['data']
            tab_names = ["Full Matrix", "Cross Matrix", "Visualization"]
            selected_tab = st.radio(
                "Navigation",
                tab_names,
                horizontal=True,
                index=tab_names.index(st.session_state.corr['active_tab']),
                key='tab_selector',
                label_visibility="hidden"
            )
            st.session_state.corr['active_tab'] = selected_tab

            # Full Matrix Tab
            if selected_tab == "Full Matrix":
                st.write("### Full Correlation Matrix")
                st.dataframe(st.session_state.corr['data']['corr_matrix'].style.format("{:.2f}"))
                st.write("### P-Value Matrix")
                st.dataframe(st.session_state.corr['data']['pval_matrix'].style.format("{:.4f}"))

            # Cross Matrix Tab
            elif selected_tab == "Cross Matrix":
                st.write(f"### {st.session_state.corr['data']['session1']} vs {st.session_state.corr['data']['session2']} Correlations")
                st.dataframe(st.session_state.corr['data']['cross_corr'].style.format("{:.2f}"))
                st.write("### Significant Correlations")
                sig_corrs = st.session_state.corr['data']['cross_corr'].where(
                    st.session_state.corr['data']['cross_pvals'] < st.session_state.corr['p_thresh']
                )
                st.dataframe(sig_corrs.style.format("{:.2f}"))

            #  Visualization Tab avec heatmap Plotly
            elif selected_tab == "Visualization":
                col1, col2 = st.columns(2)
                with col1:
                    show_all = st.checkbox(
                        "Show all correlations (incl. non-significant)",
                        value=st.session_state.corr.get('show_all', False),
                        key='corr_show_all'
                    )
                    st.session_state.corr['show_all'] = show_all
                with col2:
                    # p_thresh = st.slider(
                    #     "p-value threshold",
                    #     0.001, 0.1,
                    #     value=st.session_state.corr.get('p_thresh', 0.05),
                    #     step=0.001,
                    #     key='corr_p_thresh'
                    # )
                    p_values = [round(x, 3) for x in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]]

                    p_thresh = st.selectbox(
                        "p-value threshold",
                        options=p_values,
                        index=p_values.index(st.session_state.corr.get('p_thresh', 0.05)),
                        key='corr_p_thresh'
                    )
                    st.session_state.corr['p_thresh'] = p_thresh

                # Préparer la matrice à afficher : masquer si nécessaire les corrélations non-significatives selon le seuil choisit par l'utilisateur
                cross_corr_plot = data['cross_corr'].copy()

                if not show_all:
                    mask = data['cross_pvals'] >= p_thresh
                    cross_corr_plot = cross_corr_plot.mask(mask)

                # Plotly heatmap -- si equal fonctionne pas bien, possibilité d'ajusté manuellement la taille des cellules de la heatmap
                fig = px.imshow(
                    cross_corr_plot,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    aspect="equal",
                    labels=dict(color="Correlation")
                )
                fig.update_traces(
                    hoverongaps=False
                )
                fig.update_layout(
                    title=f"Correlations {data['session1']} vs {data['session2']}",
                    margin=dict(l=40, r=40, t=40, b=40),
                    coloraxis_colorbar=dict(title="Correlation"),
                    xaxis=dict(tickangle=45, showgrid=False, zeroline=False,tickfont=dict(color='black')),
                    yaxis=dict(showgrid=False, zeroline=False,tickfont=dict(color='black') ),
              
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Exporter en format PDF (format csv déjà disponible sur la figure interactive pour chaque table séparément)
                if st.session_state.corr['data_ready']:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state.corr['data']['corr_matrix'].to_excel(writer, sheet_name='Full_Correlation')
                        st.session_state.corr['data']['pval_matrix'].to_excel(writer, sheet_name='Full_Pvalues')
                        st.session_state.corr['data']['cross_corr'].to_excel(writer, sheet_name='Cross_Correlation')
                        st.session_state.corr['data']['cross_pvals'].to_excel(writer, sheet_name='Cross_Pvalues')
                    
                    st.download_button(
                        label="📥 Download the results",
                        data=output.getvalue(),
                        file_name=f"correlation_{st.session_state.corr['data']['session1']}_vs_{st.session_state.corr['data']['session2']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
st.markdown("---")
if st.button("🔙 Retour à l'accueil"):
    st.session_state.page = "home"
    st.rerun()