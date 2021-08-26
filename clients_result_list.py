import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



#note: ajouter SKu dans index !
def app():
    """
    # Client Analysis
    Here's our first attempt at using data to create a table:
    """

    features_importance = pd.read_csv("features_importance.csv")
    print(features_importance.head())
    client_list = pd.read_csv("client_list.csv", sep=",")
    print(client_list.head())

    #client_list = pd.read_csv("client_list.csv", index_col="SK_ID_CURR", sep=",")


    #slider
    slider_values = st.slider("Filter on the most important features", max_value=100, value=10)
    features_list = features_importance.iloc[0:slider_values].features.values.tolist()
    features_list.append("% default")


    values = st.slider(
        'Select a range of default probability values',
        5.0, 100.0, (5.0, 100.0))
    st.write('Values:', values[0], values[1])

    'You selected: from ', str(values[0]), ' to ', str(values[0])

    client_list_df = client_list[features_list][(client_list['% default'] >= values[0]) &\
                                        (client_list['% default'] <= values[1])]\
             .sort_values('% default')

    st.write(client_list_df)

    st.text(str('There are {} clients selected, out of {} ({}%)'.format(len(client_list_df),
                                                                        len(client_list),
                                                                        round(len(client_list_df) * 100 /
                                                                              len(client_list), 2))))

    st.text('Distribution of the selected clients\' default probability')

    fig, ax = plt.subplots()
    ax.hist(client_list_df['% default'], bins=50)
    plt.title("Default probability distribution")
    plt.style.use('seaborn-dark-palette')
    st.pyplot(fig)
