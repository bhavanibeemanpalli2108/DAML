import streamlit as st
from app import load_css, da
from data_analysis import run_analysis
from prediction import prediction
# page configuration

st.set_page_config("DAML",layout="centered")


load_css("style.css")



choice = st.sidebar.selectbox(
    "Navigate",
    ["Upload", "Data Analysis", "Prediction"]
)

if choice == "Upload":
    da()

elif choice == "Data Analysis":
    if "df" in st.session_state:
        run_analysis(st.session_state["df"])
    else:
        st.warning("Please upload a dataset first")

elif choice == "Prediction":
    if "df" in st.session_state:
        prediction(st.session_state["df"])
    else:
        st.warning("Please upload a dataset first")
