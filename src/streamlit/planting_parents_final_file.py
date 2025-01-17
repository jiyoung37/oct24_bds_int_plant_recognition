import streamlit as st
import os
import pandas as pd

# Define the GitHub raw file URL
file_path = os.path.expanduser("~/Documents/DataScientest/plant_dataset.csv")

# raw github path: https://raw.githubusercontent.com/Nielsvd06/streamlit_apps/refs/heads/main/plant_dataset.csv

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# start with the streamlit app

# sidebar and page navigation
st.sidebar.title("Table of contents")
pages = ["Home", "Data overview", "Data analysis & visualization", "Model training", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

# set a dynamic title for each page
st.title(f"Planting Parents: {page}")

if page == pages[0]:
    st.header("Introduction")
    st.write("Welcome to the site of the Planting Parents, where we have trained an AI model to recognise different species of plants and whether they are sick or healthy."
    " We sincerely hope you enjoy our page, that you may find it informative and recognise the plants you want to recognise")
    st.write("### 2nd paragraph")

elif page == pages[1]:
    st.write("### Scope of the data")
    st.write("\n", "Data Frame shape:", "\n", df.shape, "\n")
    st.dataframe(df.describe())
    st.dataframe(df.head(6))

elif page == pages[2]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[3]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[4]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")