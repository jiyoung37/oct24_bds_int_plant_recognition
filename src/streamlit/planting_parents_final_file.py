import streamlit as st
# import kagglehub
import os
# import cv2
# import glob
# import shutil
# import re
# import seaborn as sns
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
# from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import ResNet101
# from tensorflow.keras.applications import EfficientNetV2L
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.callbacks import ModelCheckpoint

# Define the path to the file
file_path = os.path.expanduser("~/Documents/DataScientest/plant_dataset.csv")

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)


# start with the streamlit app

# sidebar and page navigation
st.sidebar.title("Table of contents")
pages = ["Home", "Data exploration", "Data Visualization", "Model training", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

# set a dynamic title for each page
st.title(f"Planting Parents: {page}")

if page == pages[0]:
    st.write("### Presentation of the scope and data")
    st.dataframe(df.head(6))
    st.write("\n", "Data Frame shape:", "\n", df.shape, "\n")
    st.dataframe(df.describe())

elif page == pages[1]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[2]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[3]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

elif page == pages[4]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")