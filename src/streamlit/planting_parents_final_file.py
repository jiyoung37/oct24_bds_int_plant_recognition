import streamlit as st
import os
import pandas as pd
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

# Function to load Keras model
def load_keras_model(file_path):
    return tf.keras.models.load_model(file_path)

# Function to load PyTorch model
def load_pytorch_model(file_path):
    model = CustomResNet50(num_classes=38)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# Load class indices from a JSON file
def load_class_indices(file_path="src/streamlit/class_indices.json"):
    with open(file_path, "r") as f:
        return json.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Custom ResNet50 class to define our TL architecture


# Load the CSV file into a DataFrame
df = pd.read_csv("src/streamlit/plant_dataset.csv")

# start with the streamlit app

# sidebar and page navigation
st.sidebar.title("Table of contents")
pages = ["Home", "Data overview", "Data exploration", "Model Training", "Model Interpretability", "Conclusion", "Predict your plant"]
page = st.sidebar.radio("Go to", pages)

# set a dynamic title for each page
# st.title(f"Planting Parents: {page}")

if page == pages[0]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("src/visualization/Planting_parents_logo.png", use_container_width=True)
        st.header("Introduction")

    st.write("Welcome to the site of the Planting Parents, where we have trained an AI model to recognise 14 different species of plants and whether they are sick or healthy by analysing 20 plant diseases."
            " We're a group of young parents who value life and share an interest in growing plant life as well. It has been a great and rewarding challenge to present to you this app with our findings and results." 
            " We sincerely hope you enjoy our page and that you may find it informative and recognise the plants you want to recognise. ")
    st.write("**The planting parents,**")
    st.write("**Lara, Ji-Young, Yannik & Niels**")


####################
# DATA OVERVIEW    #
####################


elif page == pages[1]:
    st.write("In the data overview we'll go a bit deeper in the data we've explored and the data we ultimately used after which we defined our pre-processing steps ready to be analysed for the exploration phase of the project." 
              " Click through the tabs below to learn more: ")
    
    tab1, tab2, tab3 = st.tabs(["The numbers", "plants & diseases", "Pre-processing steps"])
    
    with tab1: # The numbers
        st.write("For our initial research into data, we stumbled upon many datasets that had some sort of plant recognition dataset of many different species from the website Kaggle. ")
        
        st.write("""
            We work with a database called ‘New Plant Diseases Dataset’ which is a public repository available on Kaggle.

It includes multiple subfolders; Train, Valid and Test. These subfolders contain subfolders per plant where the target image files (jpg) are stored, totalling 87,900 images.

    Files in train_path:            70,295
    Files in valid_path:            17,572
    Files in test_path:                 33
    Total files:                    87,900
                 """)
        st.write("\n", "Data Frame shape:", "\n", df.shape, "\n")
        st.write("All images have 256 * 256 as their image size.")
        st.write("In the below table overview you'll find all the names of the plant species and plant diseases: ")
        st.image("src/visualization/Plants_&_Diseases_count.png", use_container_width=True)
    
    with tab2: # Plants & diseases
        st.write("")
        st.image("src/visualization/diseased_plant_samples.png")

        st.write("")
        st.image("src/visualization/healthy_plant_samples.png")

    with tab3: # Pre-processing steps
        st.write(""" As for the preprocessing phase we followed these steps to get to the analysis and visualisation phase of the files:
                 
                1.1 Import kaggle hub and download dataset. 
	            1.2 Define the paths.
	            1.3 Importing necessary libraries for data processing and visualization.
                1.4 Creating a DataFrame with Train and Valid.
                1.5 Check for missing values and duplicates between train and valid subset.

        The data was already well sorted, indexed and labelled. We only had to create an import path and created lists to work of from: 
        # Initialize lists
        image_paths = []
        species_labels = []
        disease_labels = []
        dataset_split = []

        We looked for missing values and duplicates even crosschecked between the train and valid folders and found none. """)


####################
# DATA EXPLORATION #
####################


elif page == pages[2]:
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")


####################
# MODEL TRAINING   #
####################


elif page == pages[3]:
    st.write("### Model Training")
    
    tab1, tab2, tab3 = st.tabs(["Baseline CNN's", "Transfer Learning", "Pre-trained PyTorch"])

    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")

with tab1: # Baseline CNN's (Ji)
with tab2: # Transfer Learning (Niels and Lara)
        st.write("In this section, we tested different pre-trained models")
with tab3: # Pre-trained models with Pytorch (Yannik)


####################
# MODEL INTERPRET  #
####################


elif page == pages[4]:
    st.write("### Model Interpretability")
    st.write("Yanniks Todo's:")
    st.checkbox("Why interpretability matters ")
    st.checkbox("Short explanationon Grad-CAM")
    st.checkbox("Show some examples")
    st.checkbox("Grad-CAMs when going through the layers of a model")


####################
# CONCLUSION       #
####################


elif page == pages[5]:
    st.write("### Conclusion")
    st.write("### 1st paragraph")
    st.write("### 2nd paragraph")


####################
# UPLOAD IMAGE     #
####################


elif page == pages[6]:
    st.write("### Predict your plant")

    # Dropdown menu for selecting a trained model
    model_files = [f for f in os.listdir("src/models/") if f.endswith(".keras") or f.endswith(".pth")]
    selected_model_file = st.selectbox("Select a trained model:", model_files)
    
    # Checkbox for Grad-CAM
    display_grad_cam = st.checkbox("Display Grad-CAM")

    # Drag-and-drop file uploader for image
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Load the selected model
        model_path = os.path.join("src/models", selected_model_file)
        if selected_model_file.endswith(".keras"):
            model = load_keras_model(model_path)
            predictions = model.predict(preprocessed_image)
            predicted_idx = np.argmax(predictions, axis=1)[0]
        elif selected_model_file.endswith(".pth"):
            model = load_pytorch_model(model_path)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            torch_image = transform(image).unsqueeze(0)
            predictions = model(torch_image).detach().numpy()
            predicted_idx = np.argmax(predictions, axis=1)[0]

        # Load class indices
        class_indices = load_class_indices()

        # Display predictions
        st.subheader("Model Prediction")
        # st.write(predictions)
        st.write(class_indices[str(predicted_idx)])
        # st.write(predicted_classes)

    