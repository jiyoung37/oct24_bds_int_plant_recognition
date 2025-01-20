import streamlit as st
import os
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("./plant_dataset.csv")

# start with the streamlit app

# sidebar and page navigation
st.sidebar.title("Table of contents")
pages = ["Home", "Data overview", "Data analysis & visualization", "Model training", "Model Interpretability", "Conclusion", "Predict your plant"]
page = st.sidebar.radio("Go to", pages)

# set a dynamic title for each page
# st.title(f"Planting Parents: {page}")

if page == pages[0]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./../visualization/Planting_parents_logo.png", use_container_width=True)
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
    st.write("### Model Interpretability")
    st.write("Yanniks Todo's:")
    st.checkbox("Why interpretability matters ")
    st.checkbox("Short explanationon Grad-CAM")
    st.checkbox("Show some examples")
    st.checkbox("Grad-CAMs when going through the layers of a model")

elif page == pages[5]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[6]:
    st.write("### Upload an image to predict the plant type")
    st.write("This subpage should contain the actual app. Here, the user should chose")
    st.checkbox("between different models")
    st.checkbox("wether or not a Grad-CAM of the image should be shown")
    