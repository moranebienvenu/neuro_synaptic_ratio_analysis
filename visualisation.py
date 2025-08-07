import streamlit as st
from functions import process_zip_to_dataframe, create_interactive_plots, get_subjects_and_title, detect_group, clear_overlay, cohens_d, extract_pseudo_r2_cs_from_summary,extract_subject_id,clean_predictor_name

st.set_page_config(page_title="Neurotransmitter Profile Visualisation", layout="wide")
def show():

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


    if uploaded_zip is not None and not df_combined.empty:
        # Initialisation du session state
        if 'base_plots' not in st.session_state:
            st.session_state.base_plots = None
        if 'overlay_plots' not in st.session_state:
            st.session_state.overlay_plots = None
        if 'show_overlay' not in st.session_state:
            st.session_state.show_overlay = False
        if 'overlay_subjects' not in st.session_state:
            st.session_state.overlay_subjects = []
        if "overlay_color_map" not in st.session_state:
            st.session_state.overlay_color_map = {} 

        # st.header("📊 Neurotransmitter Profile Visualization")
        
        # Colonnes pour l'interface
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Sélection du groupe principal
            st.subheader("Main Profile Selection")
            analysis_type = st.radio(
                "Analysis type:",
                ["Single subject", "By session and sex", "Combined filter"],
                horizontal=True
            )
            
            #doit enlever le detect_group si "Single Subject" -- crée un bug
            subjects, plot_title, sex_filter, session = get_subjects_and_title(df_combined, analysis_type,context="main")
            if analysis_type != "Single subject":
                subject_groups = {
                    subj: detect_group(subj) for subj in subjects
                }
                
                available_groups = sorted(set(subject_groups.values()))
                selected_groups = st.multiselect(
                    "Filter by subject group:",
                    options=available_groups,
                    default=available_groups,
                    key="group_filter"
                )
                # Filtrer les sujets en fonction des groupes sélectionnés
                subjects = [subj for subj in subjects if subject_groups[subj] in selected_groups]
            st.write(f"Number of subjects in main profile: {len(subjects)}")
            # Bouton de génération
            if st.button("Generate Main Profile"):
                with st.spinner("Generating main profile..."):
                    is_group = analysis_type != "Single subject"
                    fig1, fig2, fig3, colors1, colors3 = create_interactive_plots(df_combined, subjects, plot_title, is_group=is_group)
                    st.session_state.base_plots = (fig1, fig2, fig3)
                    st.session_state.show_overlay = False  # Reset overlay quand on change le main 
                    st.rerun()

        with col2:
            # 2. Options d'affichage et overlay
            st.subheader("Display Options")
            show_data = st.checkbox("Show selected data", value=False)
            
            prev_show_overlay = st.session_state.get("show_overlay", False)
            show_overlay = st.checkbox(
                "Enable overlay comparison", 
                value=prev_show_overlay,
                key="enable_overlay"
            )

            # Si l'utilisateur a décoché l'overlay, on efface tout
            if prev_show_overlay and not show_overlay:
                clear_overlay(df_combined, subjects, plot_title)
                st.session_state.show_overlay = False
                st.rerun()

            # Met à jour l'état final
            st.session_state.show_overlay = show_overlay
            
            # Si overlay activé, afficher les options de sélection
            if st.session_state.show_overlay:
                st.markdown("---")
                st.subheader("Overlay Profile Selection")
                
                overlay_type = st.radio(
                    "Overlay type:",
                    ["Single subject", "By session and sex", "Combined filter"],
                    horizontal=True,
                    key="overlay_type"
                )

                if "current_overlay_selection" not in st.session_state:
                    st.session_state.current_overlay_selection = []
                # Logique de sélection des sujets overlay
                overlay_subjects, overlay_title, sex_filter, session = get_subjects_and_title(
                    df_combined,
                    overlay_type, 
                    existing_subjects=subjects,
                    is_overlay=True,
                    context="overlay"
                )
                if overlay_type != "Single subject":
                    overlay_subject_groups = {
                        subj: detect_group(subj) for subj in overlay_subjects
                    }
                    
                    available_ov_groups = sorted(set(overlay_subject_groups.values()))
                    selected_ov_groups = st.multiselect(
                        "Filter overlay by subject group:",
                        options=available_ov_groups,
                        default=available_ov_groups,
                        key="overlay_group_filter"
                    )
                    # Filtrer les sujets en fonction des groupes sélectionnés
                    overlay_subjects = [subj for subj in overlay_subjects if overlay_subject_groups[subj] in selected_ov_groups]

                if "overlay_ready" not in st.session_state:
                    st.session_state.overlay_ready = False
                
                # Bouton Apply Overlay
                if st.button("Apply Overlay"):
                    # Met à jour la sélection courante avec la sélection active dans le widget
                    current_overlay = st.session_state.get("overlay_subjects", [])
                    new_subjects = [s for s in overlay_subjects if s not in current_overlay]
                    if new_subjects:
                        for subj in new_subjects:
                            fig1_ov, fig2_ov, fig3_ov, _, _ = create_interactive_plots(
                                df_combined,
                                [subj],  
                                title_suffix=overlay_title,
                                is_group=False,
                                is_overlay=True
                            )
                            if "overlay_plots" not in st.session_state:
                                st.session_state.overlay_plots = []
                            st.session_state.overlay_plots= (fig1_ov, fig2_ov, fig3_ov)
                            st.session_state.overlay_subjects.append(subj)
                            st.session_state.overlay_title = overlay_title
                            st.session_state.overlay_ready = True
                        # Afficher le nombre de sujets APRÈS la mise à jour
                        st.write(f"Number of subjects currently applied in overlay: {len(st.session_state.overlay_subjects)}")
                    else:
                        st.info("Selected subject(s) already in overlay.")

                #bouton pour clear tous les sujets overlay
                
                if st.button("🗑️ Clear Overlay"):
                    clear_overlay(df_combined, subjects, plot_title)
                    st.rerun()

        # ------------------------ AFFICHAGE DES GRAPHIQUES ------------------------
        if "base_plots" in st.session_state and st.session_state.base_plots:
            fig1, fig2, fig3 = st.session_state.base_plots
            # Configurer le layout pour superposition avant d'ajouter les overlays
            for fig in [fig1, fig2, fig3]:
                fig.update_layout(barmode='overlay', bargap=0.2)

            if (
                st.session_state.show_overlay 
                and st.session_state.get("overlay_plots") 
                and st.session_state.get("overlay_ready", False)
            ):
                try:
                    fig1_ov, fig2_ov, fig3_ov = st.session_state.overlay_plots
                    for fig, fig_ov in zip([fig1, fig2, fig3], [fig1_ov, fig2_ov, fig3_ov]):
                        fig.add_traces(fig_ov.data)
                except (ValueError, TypeError) as e:
                    st.warning(f"Erreur lors du chargement des overlays : {e}")
                finally:
                    # Une fois les overlays appliqués, on reset le flag
                    st.session_state.overlay_ready = False     
            # Afficher les graphiques
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.plotly_chart(fig3, use_container_width=True)
                
                # Afficher les données si demandé
                if show_data:
                    st.subheader("Selected Subjects Data")
                    overlay_subjects = st.session_state.get("overlay_subjects", [])
                    all_subjects = list(set(subjects) | set(overlay_subjects))
                    st.dataframe(df_combined[df_combined['subject'].isin(all_subjects)])
    pass
show()

st.markdown("---")
if st.button("🔙 Retour à l'accueil"):
    st.session_state.page = "home"
    st.rerun()