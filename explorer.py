import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def app():
    client_list = pd.read_csv("client_list_raw.csv", index_col="SK_ID_CURR")

    all_features = client_list.columns.tolist()

    st.markdown('# Compare clients')

    client_list_id = st.multiselect(
        'Select client IDs',
        client_list.index)
    'You selected:', client_list_id

    st.write(client_list[client_list.index.isin(client_list_id)])

    st.markdown('# Focus on specific features')

    features_selected = st.multiselect(
        'Filter on specific features',
        all_features, default=["% default"])

    'You selected:', features_selected

    st.write(client_list[features_selected][client_list.index.isin(client_list_id)])

    st.markdown('# Compare with the rest of the clients')

    analyze_feature = st.selectbox(
    'Check the distribution of a feature',
        features_selected)
    st.write('You selected:', analyze_feature)

    # analyze feature distribution
    if analyze_feature:
        if client_list[analyze_feature].dtype == 'O':
            bar_chart_data = client_list.reset_index().groupby([analyze_feature], as_index=False) \
                .SK_ID_CURR.count() \
                .rename(columns={"SK_ID_CURR": "Clients number"}) \
                .set_index(analyze_feature)
            st.bar_chart(bar_chart_data)

        else:
            fig, ax = plt.subplots()
            ax.hist(client_list[analyze_feature], bins=50)
            ax.hist(client_list[analyze_feature], bins=20)
            plt.title(analyze_feature)
            plt.style.use('seaborn-dark-palette')
            st.pyplot(fig)

