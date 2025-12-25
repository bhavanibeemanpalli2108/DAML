import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from data_analysis import run_analysis





# load the css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
        
# load_css("style.css")

def da():
    
    st.markdown(
    "<div class='doodle d1'></div><div class='doodle d2'></div>",
    unsafe_allow_html=True
)
    st.markdown(
        "<h2 style='text-align:center;'>Drop your dataset here</h2>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        st.write("File uploaded")

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["df"] = df
        
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.dataframe(df.head())
        st.dataframe(df.describe())
        
        run_analysis(df)

    