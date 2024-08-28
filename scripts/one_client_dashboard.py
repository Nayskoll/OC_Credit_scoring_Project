import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from matplotlib import pyplot as plt


# note: ajouter SKu dans index !
def app():
    st.markdown('# Focus on one client')

    scaled_data = pd.read_csv("scaled_data.csv", index_col="SK_ID_CURR")
    client_list = pd.read_csv("client_list.csv", index_col="SK_ID_CURR")
    features_importance = pd.read_csv("features_importance.csv")

    client_id = st.selectbox(
        'Select a client ID',
        client_list.index)
    st.write('You selected:', client_id)

    default_proba = client_list[client_list.index == client_id]["% default"].values[0].round(2)

    st.write('The default probability of the selected client is {}%'.format(default_proba))

    no_default_users = client_list[client_list.Result == False].index.tolist()
    default_users = client_list[client_list.Result == True].index.tolist()

    if client_id in (default_users):
        st.write('He is considered as a potential default client')
    else:
        st.write('He is not considered as a potential default client')

    scaled_data = scaled_data.reset_index()

    radar_chart_default = scaled_data[scaled_data["SK_ID_CURR"].isin(default_users)] \
        .drop(columns="SK_ID_CURR") \
        .mean()
    default_df = pd.DataFrame(radar_chart_default, columns=["values"]).reset_index()

    radar_chart_no_default = scaled_data[scaled_data["SK_ID_CURR"].isin(no_default_users)] \
        .drop(columns="SK_ID_CURR") \
        .mean()
    no_default_df = pd.DataFrame(radar_chart_no_default, columns=["values"]).reset_index()

    one_client_df = scaled_data[scaled_data["SK_ID_CURR"] == client_id] \
        .drop(columns="SK_ID_CURR") \
        .T \
        .reset_index()
    one_client_df.columns = ["index", "values"]

    st.markdown('### Radar chart based on top 10 features deciles')

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=default_df["values"],
        theta=default_df["index"],
        fill='toself',
        name='Default mean',
        mode='lines',
        line_color='Red'))

    fig.add_trace(go.Scatterpolar(
        r=no_default_df["values"],
        theta=no_default_df["index"],
        fill='toself',
        name='No Default Mean',
        mode='lines',
        line_color='Green'
    ))

    fig.add_trace(go.Scatterpolar(
        r=one_client_df["values"],
        theta=one_client_df["index"],
        fill='toself',
        name='Client Selected',
        mode='lines',
        line_color='Blue'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True
    )

    st.write(fig)

    st.markdown('### Check the coefficient for each features.  If it\'s positive, good client should have low values')

    st.write(features_importance.head(10)[["features", "importance"]])