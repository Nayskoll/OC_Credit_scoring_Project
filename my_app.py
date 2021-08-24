import explorer
import importance_features
import clients_result_list
import one_client_dashboard
import streamlit as st
PAGES = {
    "Client result list": clients_result_list,
    "Explorer": explorer,
    "Importance features": importance_features,
    'One client dashboard': one_client_dashboard
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()