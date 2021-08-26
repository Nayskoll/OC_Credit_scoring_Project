import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

def app():
    """
    # Features importance
    Here's our first attempt at using data to create a table:
    """

    features_importance = pd.read_csv("features_importance.csv")

    #slider
    slider_values = st.slider("Get the most important features", max_value=100)
    features_importance.iloc[0:slider_values]

    #features importance
    """features"""
    ft_importance_bar = features_importance[["features", "importance"]].iloc[0:min(slider_values, 50)].set_index('features')
    st.bar_chart(ft_importance_bar)

    option = st.selectbox(
        'Choose one feature to see its ranking in the model',
         features_importance['features'])

    'You selected:', option

    option_rank = features_importance[features_importance['features']==option]["rank"].values[0]
    option_total_rank = features_importance["rank"].max()
    option_coeff = features_importance[features_importance['features']==option]["importance"].values[0]

    st.write("{} is ranked {}/{} in terms of features importance in the model. Its importance coefficient is {}.".format(
        option,
        option_rank,
        option_total_rank,
        option_coeff))





