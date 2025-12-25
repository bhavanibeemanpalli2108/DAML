import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
# from data_analysis import run_analysis


def run_analysis(df):
    
   #Dataset preview
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head())
    st.markdown('</div>',unsafe_allow_html=True)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())


    df.replace(["?", "NA", "N/A", "null", "NULL", "na"], np.nan, inplace=True)

    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])
    missing_percent = (df.isnull().sum() / len(df)) * 100


    for col in df.columns:
        missing_pct = missing_percent[col]

        if missing_pct == 0:
            continue

        # Drop column if too much missing
        if missing_pct > 30:
            df.drop(columns=[col], inplace=True)
            st.write(f"Dropped column: {col} ({missing_pct:.1f}% missing)")

        else:
            # Imputation
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
                st.write(f"Imputed {col} with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                st.write(f"Imputed {col} with mode")




    st.subheader("Outlier Analysis")

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower=lower, upper=upper)


    fig, ax = plt.subplots()
    sns.boxplot(data=df[num_cols])
    st.pyplot(fig)


    st.subheader("Distribution (Numeric)")

    col = st.selectbox("Select numeric column", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True)
    st.pyplot(fig)
    
    
    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0:
        col = st.selectbox("Select categorical column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="YlGn", ax=ax)
    st.pyplot(fig)


    # #prepare the data
    # x,y=df[['total_bill']], df["tip"]
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

    
    st.session_state["clean_df"] = df

    # later: stats, missing values, plots
    return df