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
    
def create_interactive_plots(df, subjects, title_suffix="", is_group=False, is_overlay=False):

    # Filtrer les données et calculer les moyennes que si groupe sinon mettre les données individuelles pour chaque sujet base/overlay
    plot_data = df[df['subject'].isin(subjects)]
    if is_group or len(subjects) > 1:
        # Cas groupe : on calcule la moyenne
        data_to_plot = plot_data.mean(numeric_only=True)
    else:
        # Cas sujet unique : on prend les données brutes
        data_to_plot = plot_data.iloc[0]  
    # Détection dynamique des systèmes
    systems = [col.replace('loc_inj_', '') for col in df.columns 
               if col.startswith('loc_inj_') and not col.startswith('loc_inj_perc')]
    
    # 1. Préparation des données pour les graphiques 1 et 2
    loc_inj_perc = [data_to_plot[f'loc_inj_perc_{sys}'] for sys in systems]
    tract_inj_perc = [data_to_plot[f'tract_inj_perc_{sys}'] for sys in systems]
    
    # 2. Calcul des ratios pre/post comme NeuroTmap.py
    pre_systems = ['A4B2', 'M1', 'D1', 'D2', '5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6']
    post_systems = ['VAChT', 'DAT', '5HTT']
    
    def safe_get(data, system, prefix):
        col = f'{prefix}_{system}'
        return data[col] if col in data else 0.0
    
    radii3a, radii3b = [], []
    for i, sys in enumerate(pre_systems):
        recep = max(safe_get(data_to_plot, sys, 'loc_inj_perc'), safe_get(data_to_plot, sys, 'tract_inj_perc'))
        trans_sys = 'VAChT' if sys in ['A4B2', 'M1'] else 'DAT' if sys in ['D1', 'D2'] else '5HTT'
        trans = max(safe_get(data_to_plot, trans_sys, 'loc_inj_perc'), safe_get(data_to_plot, trans_sys, 'tract_inj_perc'))
        
        radii3a.append(trans / 0.1 if recep == 0 else trans / recep)
        radii3b.append(recep / 0.1 if trans == 0 else recep / trans)
    
    radii3b_avg = [
        (radii3b[0] + radii3b[1]) / 2,
        (radii3b[2] + radii3b[3]) / 2,
        sum(radii3b[4:9]) / 5
    ]
    
    radii3 = np.append(radii3a, radii3b_avg)
    radii3_log = np.where(radii3 == 0, -1, np.log(radii3))
    
    # Couleurs identiques à NeuroTmap
    # Si overlay, définir une couleur unique -- peut être ajouté hachure transparente pour le rendu
    if is_overlay:
        if title_suffix not in st.session_state.overlay_color_map:
            hue = len(st.session_state.overlay_color_map) * 60 % 360
            color = f"hsla({hue}, 80%, 50%, 0.5)"
            st.session_state.overlay_color_map[title_suffix] = color
        overlay_color = st.session_state.overlay_color_map[title_suffix]

        # Définir une liste de couleurs transparentes pour le overlay
        colors1 = [overlay_color] * len(systems)
        colors3 = [overlay_color if val > 1 else overlay_color.replace("0.5", "0.2") for val in radii3]
    else:
        colors1 = ["#B7B3D7", "#928CC1", "#6E66AD", "#B7DEDA", "#92CEC8", "#6BBDB5", 
                "#EBA8B1", "#FCFCED", "#FBFAE2", "#F8F8D6", "#F8F6CB", "#F6F4BE", "#F5F2B3"]
        colors3 = ['#42BDB5' if val > 1 else '#F5F2B3' for val in radii3]
    
    # Création des subplots
    # fig = sp.make_subplots(
    #     rows=3, cols=1,
    #     # specs=[[{'type': 'polar'}, {'type': 'polar'}],
    #     #      [{'type': 'polar'}, None]],
    #     # column_widths=[0.5, 0.5],
    #     specs=[[{'type': 'polar'}],
    #        [{'type': 'polar'}],
    #        [{'type': 'polar'}]],
    #     vertical_spacing=0.15 ,
    #     subplot_titles=(
    #         f'Receptor/transporter lesion \n',
    #         f'Pre/post synaptic ratios (log scale)\n',
    #         f'Receptor/transporter disconnection\n'
    #     )
    # )
   
    # Configuration commune
    config = {
        'title_x': 0.2,  # Centre les titres ajustable
        'title_font_size': 14,
        'polar': {
            'angularaxis': {
                'direction': 'clockwise',
                'rotation': 90,
            },
            'bargap': 0.1  
        }
    }

    base_val = 0 if is_overlay else None
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()

    # Graphique 1: Lésions
    fig1.add_trace(go.Barpolar(
        r=loc_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    fig1.update_layout(
        title_text=f'<b>Receptor/transporter lesion', #: {title_suffix}</b>',
        polar_radialaxis_ticksuffix='%',
        height=600,
        showlegend=True,
        margin=dict(l=50, r=0, t=30, b=10),
        font=dict(size=12),
        **config
    )

    # Graphique 2: Déconnexions
    fig2.add_trace(go.Barpolar(
        r=tract_inj_perc,
        theta=systems,
        marker_color=colors1[:len(systems)],
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra></extra>',
        width=np.pi/4,
        base=base_val 
    ))
    fig2.update_layout(
        title_text=f'<b>Receptor/transporter disconnection', #: {title_suffix}</b>',
        polar_radialaxis_ticksuffix='%',
        height=600,
        showlegend=True,
        margin=dict(l=50, r=0, t=30, b=10),
        font=dict(size=12),
        **config
    )
 
    # Graphique 3: Ratios
    fig3.add_trace(go.Barpolar(
        r=radii3_log,
        theta=[f"pre {sys}" for sys in pre_systems] + [f"post {sys}" for sys in post_systems],
        marker_color=colors3,
        name=title_suffix,
        hovertemplate='<b>%{theta}</b><br>%{r:.2f}<extra></extra>',
        width=np.pi/4,
        base=base_val
    ))
    fig3.update_layout(
        title_text=f'<b>Pre/post synaptic ratios (log scale)',#: {title_suffix}</b>',
        polar_radialaxis_range=[-1, 1],
        height=500,  
        showlegend=True,
        #margin=dict(l=30, r=10, t=10, b=10),
        font=dict(size=12),
        **config
    )


    return fig1, fig2, fig3, colors1, colors3

def get_subjects_and_title(df, analysis_type, existing_subjects=None, is_overlay=False, context=""):
    """
    Gère la sélection des sujets et génère un titre descriptif avec des clés uniques
    
    Args:
        df: DataFrame contenant les données
        analysis_type: Type d'analyse ("Single subject", "By session and sex", etc.)
        existing_subjects: Liste des sujets à exclure (pour overlay)
        is_overlay: Booléen indiquant si c'est pour un overlay
        context: Chaîne supplémentaire pour rendre les clés uniques
        
    Returns:
        Tuple (liste des sujets, titre, sex, session)
    """
    if existing_subjects is None:
        existing_subjects = []
    
    # Création de préfixes/suffixes uniques pour les clés
    overlay_prefix = "overlay_" if is_overlay else ""
    key_suffix = f"_{context}" if context else ""
    base_key = f"{overlay_prefix}{analysis_type}{key_suffix}".replace(" ", "_").lower()
    
    subjects = []
    title_prefix = "Overlay " if is_overlay else "Base"
    
    # 1. Cas sujet unique
    if analysis_type == "Single subject":
        available_subjects = [s for s in df['subject'].unique() if s not in existing_subjects]
        selected = st.selectbox(
            f"Select {'overlay ' if is_overlay else ''}subject:",
            options=sorted(available_subjects),
            key=f"{base_key}_subject_select"
        )
        subjects = [selected]
        #return subjects, f"{title_prefix}Subject: {selected}"
        return subjects, f"{title_prefix}: {selected}", None, None

    # 2. Cas par session
    elif analysis_type == "By session and sex":
        session = st.selectbox(
            "Select session:",
            options=["V1", "V2", "V3"],
            key=f"{base_key}_session_select"
        )
        
        sex_filter = st.radio(
            f"Sex filter for {session}:",
            ["All", "Men only", "Women only"],
            horizontal=True,
            key=f"{base_key}_sex_filter_{session}"
        )
        
        # Filtrage initial par session
        session_subjects = df[df['subject'].str.contains(f"_ses-{session}")]['subject'].tolist()
        
        # Filtrage supplémentaire par sexe
        if sex_filter != "All":
            gender = "M" if sex_filter == "Men only" else "F"
            session_subjects = df[
                (df['subject'].isin(session_subjects)) & 
                (df['sex'] == gender)
            ]['subject'].tolist()
        
        # Exclusion des sujets existants
        subjects = [s for s in session_subjects if s not in existing_subjects]
        if is_overlay:
            subjects = [s for s in subjects if s not in st.session_state.get("overlay_subjects", [])]

        # Construction du titre
        title = f"{title_prefix}: Session {session}"
        if sex_filter != "All":
            title += f" ({sex_filter})"
            
        return subjects, title, sex_filter, session


    # 3. Cas filtre combiné -- a modifier si d'autres idées de filtre
    else:
        selected_sessions = st.multiselect(
            "Select sessions:",
            options=["V1", "V2", "V3"],
            default=["V1", "V2", "V3"],
            key=f"{base_key}_multisession"
        )
        
        selected_genders = st.multiselect(
            "Select genders:",
            options=["Men (M)", "Women (F)"],
            default=["Men (M)", "Women (F)"],
            key=f"{base_key}_multigender"
        )
        
        # Conversion des genres en codes
        gender_codes = ["M" if g == "Men (M)" else "F" for g in selected_genders]
        
        # Filtrage combiné
        subjects = []
        for session in selected_sessions:
            for gender in gender_codes:
                group = df[
                    (df['subject'].str.contains(f"_ses-{session}")) &
                    (df['sex'] == gender)
                ]
                subjects.extend([s for s in group['subject'].unique() if s not in existing_subjects])
        if is_overlay:
            subjects = [s for s in subjects if s not in st.session_state.get("overlay_subjects", [])]

    
        # Construction du titre
        title = f"{title_prefix}:  "
        title += f"{', '.join(selected_sessions)} sessions"
        title += f", {', '.join(selected_genders)}"
        
        return subjects, title, None, None

def detect_group(subject_id):
    if "_sub-NA" in subject_id or "-NA" in subject_id:
        return "NA"
    elif "_sub-A" in subject_id or "-A" in subject_id:
        return "A"
    #pour les sujets controles pas possible dans NeuroTmap mais peut etre permettre comparaison des scores cliniques uniquement
    elif "_sub-C" in subject_id or "-C" in subject_id: 
        return "C"
    else:
        return "Unknown"

def clear_overlay(df=None, subjects=None, plot_title=None):
                st.session_state.overlay_plots = None
                st.session_state.overlay_subjects = []
                st.session_state.overlay_title = ""
                st.session_state.overlay_ready = False

                # Si on veut regénérer les graphes de base
                if df is not None and subjects and plot_title:
                    with st.spinner("Resetting to base profile..."):
                        fig1, fig2, fig3, _, _ = create_interactive_plots(df, subjects, plot_title)
                        st.session_state.base_plots = (fig1, fig2, fig3)

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def extract_pseudo_r2_cs_from_summary(model):
    summary_text = model.summary().as_text()
    match = re.search(r"Pseudo R-squ\. \(CS\):\s+([\d.]+)", summary_text)
    if match:
        return float(match.group(1))
    return None

def get_family_and_link(dist_name, link_name, var_power=None):
    # Map link functions
    link_map = {
        "log": Log(),
        "identity": Identity(),
        "inverse": InversePower(),
        "sqrt": Sqrt()
    }
    link_func = link_map.get(link_name, Log())  # Log par défaut

    # Map families
    if dist_name == "Gaussian":
        family = Gaussian(link=link_func)
    elif dist_name == "Gamma":
        family = Gamma(link=link_func)
    elif dist_name == "Poisson":
        family = Poisson(link=link_func)
    elif dist_name == "Tweedie":
        power = var_power if var_power is not None else 1.5
        family = Tweedie(var_power=power, link=link_func)
    else:
        family = Gaussian(link=link_func)

    return family

def check_model_assumptions(df, outcome, predictors, family):
    # Vérifier les valeurs manquantes
    missing = df[[outcome] + predictors].isna().sum()
    if missing.any():
        st.warning(f"Données manquantes :\n{missing[missing > 0]}")
    
    # Vérifier les valeurs pour la famille choisie
    if isinstance(family, (Poisson, Gamma)) and (df[outcome] <= 0).any():
        st.error(f"La variable {outcome} contient des valeurs ≤0 - incompatible avec {family.__class__.__name__}")
        return False
    
    # Vérifier la variance pour Poisson
    if isinstance(family, Poisson):
        if df[outcome].var() > df[outcome].mean() * 1.5:
            st.warning("Surdispersion détectée - envisagez Tweedie") #ou NegativeBinomial
    
    return True

def safe_glm_crossgroup(
    df_predictors,
    df_outcomes,
    outcomes,
    systems,
    covariate=[],
    visit_name="",
    family=None,
    interaction_var=None  # ex: 'Sexe_bin', 'Group', or None
):
    results = []

    for outcome in outcomes:
        outcome_var = f"Q('{outcome}')" if any(c.isdigit() for c in outcome) else outcome

        for system, predictors in systems.items():
            for predictor in predictors:
                formula_terms = []
                term = f"Q('{predictor}')" if any(c.isdigit() for c in predictor) else predictor
                formula_terms.append(term)

                if interaction_var and interaction_var in df_predictors.columns:
                    formula_terms.append(f"{term}:{interaction_var}")

                for cov in covariate:
                    if cov != interaction_var:
                        cov_term = f"Q('{cov}')" if any(c.isdigit() for c in cov) else cov
                        formula_terms.append(cov_term)

                formula = f"{outcome_var} ~ {' + '.join(formula_terms)}"
                #st.write("Formule GLM :", formula)

                try:
                    df_predictors_temp = df_predictors.reset_index() if 'subject' not in df_predictors.columns else df_predictors.copy()
                    df_outcomes_temp = df_outcomes.reset_index() if 'subject' not in df_outcomes.columns else df_outcomes.copy()

                    needed_cols = list(set(['subject', predictor] + covariate))
                    if interaction_var and interaction_var not in needed_cols:
                        needed_cols.append(interaction_var)

                    if outcome not in df_outcomes_temp.columns:
                        st.warning(f"{outcome} non trouvé dans les outcomes.")
                        continue

                    df_merged = df_outcomes_temp[['subject', outcome]].merge(
                        df_predictors_temp[needed_cols],
                        on='subject',
                        how='inner'
                    )

                    #st.write(f"Taille après merge: {len(df_merged)} observations")
                    #st.write("Valeurs manquantes par colonne:", df_merged.isna().sum())
                    if df_merged.empty:
                        st.warning(f"Aucune donnée après merge pour {outcome} ~ {predictor}")
                        continue

                    drop_cols = list(set([outcome, predictor] + covariate + ([interaction_var] if interaction_var else [])))
                    df_clean = df_merged.dropna(subset=drop_cols)
                    #st.write(f"Taille après suppression des NaN: {len(df_clean)} observations")

                    if df_clean.empty or len(df_clean) < 3:
                        st.warning(f"Données insuffisantes pour {outcome} ~ {predictor}")
                        continue

                    non_numeric = df_clean[drop_cols].select_dtypes(exclude=['number']).columns.tolist()
                    if non_numeric:
                        st.warning(f"Variables non numériques pour {outcome} ~ {predictor}: {non_numeric}")
                        continue

                    if not check_model_assumptions(df_clean, outcome, [predictor] + covariate, family):
                        st.warning(f"Hypothèses non respectées pour {outcome} ~ {predictor}")
                        continue

                    try:
                        model = smf.glm(formula, data=df_clean, family=family).fit()
                    except Exception as e:
                        error_msg = f"Erreur lors du fit du modèle pour {outcome} ~ {predictor}: {str(e)}"
                        if "endog" in str(e) and "log" in str(family.link):
                            error_msg += "\n⚠️ Essayez une autre famille/link (ex: valeurs négatives avec link='log')"
                        st.error(error_msg)
                        continue

                    n_obs = int(model.nobs)
                    df_resid = int(model.df_resid)
                    df_model = int(model.df_model)
                    log_likelihood = model.llf
                    deviance = model.deviance
                    pearson_chi2 = model.pearson_chi2
                    pseudo_r2 = extract_pseudo_r2_cs_from_summary(model)
                    scale = model.scale

                    for param in model.params.index:
                        coef = model.params[param]
                        pval = model.pvalues[param]
                        is_interaction = ':' in param
                        base_pred = param.split(':')[0].replace("Q('", "").replace("')", "")

                        results.append({
                            'Visit': visit_name,
                            'Outcome': outcome,
                            'System': system,
                            'Predictor': param,
                            'Base_Predictor': base_pred,
                            'Coefficient': coef,
                            'Effect_Type': 'Interaction' if is_interaction else 'Main',
                            'P-value': pval,
                            'Significant': pval < 0.05,
                            'N_obs': n_obs,
                            'Df_resid': df_resid,
                            'Df_model': df_model,
                            'Log-likelihood': log_likelihood,
                            'Deviance': deviance,
                            'Pearson_chi2': pearson_chi2,
                            'Pseudo_R2_CS': pseudo_r2,
                            'Scale': scale,
                        })

                except Exception as e:
                    print(f"Erreur avec {outcome} ~ {predictor}: {e}")
                    continue
    return pd.DataFrame(results)

def perform_group_comparison(group1_data, group2_data, paired=False):
    """
    Effectue une comparaison statistique entre deux groupes avec vérifications préalables
    
    Args:
        group1_data (pd.Series): Données du groupe 1
        group2_data (pd.Series): Données du groupe 2
        paired (bool): Si True, utilise un test apparié quand V1 ; V2 ; V3 comparaison
        
    Returns:
        dict: Dictionnaire contenant tous les résultats statistiques
    """
    # Nettoyage des données
    vals1 = group1_data.dropna()
    vals2 = group2_data.dropna()
    
    # Vérification des effectifs
    if len(vals1) < 3 or len(vals2) < 3:
        return None
    
    if paired and len(vals1) != len(vals2):
        raise ValueError("For paired tests, group sizes must be equal")

    results = {
        'n_group1': len(vals1),
        'n_group2': len(vals2),
        'mean_group1': vals1.mean(),
        'mean_group2': vals2.mean(),
        'std_group1': vals1.std(),
        'std_group2': vals2.std()
    }
    
    # 1. Test de normalité (Shapiro-Wilk)
    shapiro1 = shapiro(vals1)
    shapiro2 = shapiro(vals2)
    results.update({
        'shapiro_p1': shapiro1.pvalue,
        'shapiro_p2': shapiro2.pvalue,
        'normal_dist': (shapiro1.pvalue > 0.05) and (shapiro2.pvalue > 0.05)
    })
    
    # 2. Test d'homogénéité des variances (Levene) - seulement si non apparié
    if not paired:
        levene_test = levene(vals1, vals2)
        results.update({
            'levene_p': levene_test.pvalue,
            'equal_var': levene_test.pvalue > 0.05
        })
    
    # Choix du test statistique
    if paired:
        # Tests pour données appariées
        if results['normal_dist']:
            test_result = ttest_rel(vals1, vals2)
            results.update({
                'test_type': 'Paired t-test',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            })
        else:
            test_result = wilcoxon(vals1, vals2)
            results.update({
                'test_type': 'Wilcoxon signed-rank',
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': test_result.statistic / np.sqrt(len(vals1))
            })
    else:
        # Tests pour groupes indépendants
        if results['normal_dist']:
            if results.get('equal_var', True):
                test_result = ttest_ind(vals1, vals2, equal_var=True)
                test_type = "Student-t (var égales)"
            else:
                test_result = ttest_ind(vals1, vals2, equal_var=False)
                test_type = "Welch-t (var inégales)"
            
            effect_size = (vals1.mean() - vals2.mean()) / np.sqrt((vals1.std()**2 + vals2.std()**2)/2)
            
            try:
                analysis = TTestIndPower()
                power = analysis.power(
                    effect_size=effect_size, 
                    nobs1=len(vals1), 
                    alpha=0.05,
                    ratio=len(vals2)/len(vals1), 
                    alternative='two-sided'
                )
            except:
                power = np.nan
                
            results.update({
                'test_type': test_type,
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'effect_size': effect_size,
                'power': power
            })
        else:
            test_result = mannwhitneyu(vals1, vals2, alternative='two-sided')
            n1, n2 = len(vals1), len(vals2)
            U = test_result.statistic
            Z = (U - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)  # Conversion U → Z
            p_value_from_z = 2 * (1 - norm.cdf(abs(Z)))
            effect_size = Z / np.sqrt(n1 + n2)
            
            results.update({
                'test_type': "Mann-Whitney U",
                'statistic': test_result.statistic,
                'p_value': test_result.pvalue,
                'statistic_z': Z,
                'p_value_from_z': p_value_from_z,
                'effect_size': effect_size,
                #'power': np.nan
            })
    
    results['significant'] = results['p_value'] < 0.05
    return results

def clean_groups_for_variable(df1, df2, var, paired):
    """Supprime les sujets ayant des valeurs manquantes pour une variable sélectionné par l'utilsateur.
       Si paired, conserve uniquement les paires valides."""
    df1_valid = df1[df1[var].notna()]
    df2_valid = df2[df2[var].notna()]

    if paired:
        # On garde les sujets présents et valides dans les deux groupes
        base_ids_1 = df1_valid['subject'].apply(lambda x: x.split('-V')[0])
        base_ids_2 = df2_valid['subject'].apply(lambda x: x.split('-V')[0])
        common_bases = set(base_ids_1).intersection(set(base_ids_2))

        df1_clean = df1_valid[df1_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]
        df2_clean = df2_valid[df2_valid['subject'].apply(lambda x: x.split('-V')[0]).isin(common_bases)]

        return df1_clean, df2_clean, len(common_bases)
    else:
        return df1_valid, df2_valid, None

def get_correlation_matrix(df, include_sex_bin=True):
    """
    Calculate correlation matrix with automatic test selection (Pearson/Spearman)
    and FDR correction for multiple comparisons.
    
    Parameters:
    - df: DataFrame containing the variables to correlate
    - include_sex_bin: Whether to include 'Sexe_bin' in the matrix (False for single-sex analysis)
    
    Returns:
    - corr_matrix: DataFrame of correlation coefficients
    - pval_matrix: DataFrame of FDR-corrected p-values
    """
    # Select and clean numeric data
    df_num = df.select_dtypes(include=['float64', 'int64','bool']).dropna(axis=1, thresh=int(0.5 * len(df)))
    
    # Convert bool to int if needed
    df_num = df_num.apply(lambda x: x.astype(int) if x.dtype == bool else x)

    # Exclude Sexe_bin if requested
    if not include_sex_bin and 'Sexe_bin' in df_num.columns:
        df_num = df_num.drop(columns=['Sexe_bin'])
        
    cols = df_num.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    pvals_list = []
    index_pairs = []

    for col1, col2 in combinations(cols, 2):
        x, y = df_num[col1].dropna(), df_num[col2].dropna()
        common_index = x.index.intersection(y.index)
        x, y = x.loc[common_index], y.loc[common_index]

        if len(x) < 3:
            continue  # Skip pairs with less than 3 observations

        # Normality test with error handling
        try:
            norm_x = shapiro(x)[1] > 0.05
            norm_y = shapiro(y)[1] > 0.05
        except:
            norm_x = norm_y = False

        # Choose appropriate correlation test
        if norm_x and norm_y:
            corr, pval = pearsonr(x, y)
        else:
            corr, pval = spearmanr(x, y)

        # Store results
        corr_matrix.loc[col1, col2] = corr
        corr_matrix.loc[col2, col1] = corr
        pval_matrix.loc[col1, col2] = pval
        pval_matrix.loc[col2, col1] = pval
        
        # Prepare for FDR correction
        pvals_list.append(pval)
        index_pairs.append((col1, col2))

    # Fill diagonal
    np.fill_diagonal(corr_matrix.values, 1.0)
    for col in cols:
        pval_matrix.loc[col, col] = 0.0  # p-value for diagonal is 0

    # Apply FDR correction (Benjamini-Hochberg)
    if pvals_list:
        _, pvals_corrected, _, _ = multipletests(pvals_list, alpha=0.05, method='fdr_bh')
        for (col1, col2), p_corr in zip(index_pairs, pvals_corrected):
            pval_matrix.loc[col1, col2] = p_corr
            pval_matrix.loc[col2, col1] = p_corr
    
    return corr_matrix, pval_matrix

def extract_subject_id(subject_name):
                match = re.match(r"(sub-[A-Za-z0-9]+)_ses-V[0-9]+", subject_name)
                return match.group(1) if match else None

def clean_predictor_name(name):
                    # Enlève Q(' et ') si présent
                    if isinstance(name, str) and name.startswith("Q('") and name.endswith("')"):
                        return name[3:-2]
                    return name
