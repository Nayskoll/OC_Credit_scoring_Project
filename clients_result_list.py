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
