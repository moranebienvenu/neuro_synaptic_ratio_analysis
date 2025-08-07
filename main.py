import streamlit as st

# Config globale
st.set_page_config(page_title="NeuroClinical Dashboard", layout="wide")

# Initialiser la page dans session_state
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Redirection vers la bonne page
if st.session_state.page == "home":
    import home
elif st.session_state.page == "visualisation":
    import visualisation
elif st.session_state.page == "stats":
    import stats 