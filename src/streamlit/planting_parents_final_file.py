import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

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
def load_class_indices(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image, model):
    """
    Preprocess an image to match the input size of the given model.

    Parameters:
        image (PIL.Image): The input image.
        model: The trained model (Keras or PyTorch).

    Returns:
        np.ndarray: The preprocessed image array for Keras.
        torch.Tensor: The preprocessed image tensor for PyTorch.
    """
    if isinstance(model, tf.keras.Model):  # For Keras models
        input_shape = model.input_shape[1:3]  # Extract input size from the model
        resized_image = image.resize(input_shape)
        image_array = np.array(resized_image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    elif isinstance(model, torch.nn.Module):  # For PyTorch models
        input_size = 224  # Default input size for many PyTorch models
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # Resize to input size
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("Unsupported model type. Please provide a Keras or PyTorch model.")

# Custom ResNet50 class to define our TL architecture
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()

        # Initialize the base model without pre-trained weights
        self.base_model = models.resnet50(pretrained=False)
        
        # Modify the final fully connected layer to match the pre-trained model from the .pth file
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 120)
        
        # Initialize the base model and load custom weights
        state_dict = torch.load('src/models/model_src/ResNet50-Plant-model-80.pth', map_location=torch.device('cpu'))
        self.base_model.load_state_dict(state_dict)
        
        # Modify the classifier
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the original FC layer
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)  # Return raw logits
        return x


# Grad-CAM function for Keras
def generate_grad_cam_keras(model, image, layer_name=None):
    # Automatically detect the input size of the model
    input_shape = model.input_shape[1:3]  # Get height and width of the input layer

    # Resize the input image to match the model's input size
    resized_image = image.resize(input_shape)
    img_array = img_to_array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # If layer_name is not provided, find the last convolutional layer
    if not layer_name:
        layer_name = next(
            (layer.name for layer in model.layers[::-1] if isinstance(layer, tf.keras.layers.Conv2D)),
            None
        )
        if not layer_name:
            raise ValueError("No convolutional layer found in the model.")

    # Create a model that maps inputs to activations of the target layer and model output
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Compute the gradient of the loss with respect to the feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute the Grad-CAM heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap /= max_value
    heatmap = heatmap.numpy()

    # Resize the heatmap to the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)

    # Overlay the heatmap on the image
    heatmap = np.array(heatmap)
    colormap = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Apply colormap
    overlay = (colormap * 255).astype(np.uint8)
    overlay_image = Image.blend(image.convert("RGBA"), Image.fromarray(overlay).convert("RGBA"), alpha=0.5)

    return overlay_image

# Grad-CAM function for PyTorch
def generate_grad_cam_pytorch(model, image, target_layer_name=None):
    """
    Generate Grad-CAM visualization for a PyTorch model.
    
    Parameters:
        model: PyTorch model
        image: PIL Image (input image)
        layer_name: str (name of the target convolutional layer), if not provided the function iterates through the
                    model's layer in reverse and finds the last Conv2d layer

    Returns:
        PIL Image with the Grad-CAM overlay
    """
    from torch.nn import Conv2d

    # Automatically detect the last convolutional layer if target_layer_name is not provided
    if target_layer_name is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, Conv2d):
                target_layer_name = name
                break
        if target_layer_name is None:
            raise ValueError("No convolutional layer found in the model.")

    # Transform and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the input size of the model
        transforms.ToTensor(),
    ])
    torch_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    # Register hooks on the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)

    # Forward pass
    model.eval()
    output = model(torch_image)
    class_index = output.argmax(dim=1).item()

    # Backward pass for the target class
    model.zero_grad()
    loss = output[0, class_index]
    loss.backward()

    # Compute Grad-CAM
    gradients = gradients[0].detach().cpu().numpy()
    activations = activations[0].detach().cpu().numpy()
    weights = np.mean(gradients, axis=(2, 3))  # Global average pooling of gradients
    grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights[0]):
        grad_cam += w * activations[0, i, :, :]

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam / grad_cam.max() if grad_cam.max() != 0 else grad_cam

    # Resize heatmap to match original image size
    grad_cam = np.uint8(255 * grad_cam)
    heatmap = Image.fromarray(grad_cam).resize(image.size, resample=Image.BILINEAR)

    # Overlay heatmap on the image
    heatmap = np.array(heatmap)
    colormap = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Apply colormap
    overlay = (colormap * 255).astype(np.uint8)
    overlay_image = Image.blend(image.convert("RGBA"), Image.fromarray(overlay).convert("RGBA"), alpha=0.5)

    return overlay_image

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


    with tab1: # Baseline CNN's (Ji)
        st.write("")


    with tab2: # Transfer Learning
        st.write("**Pre-trained models**")
        st.markdown("· MobileNetV2")
        st.markdown("· VGG16")
        st.markdown("· ResNet 101")
        st.markdown("· ResNet 50")
        st.markdown("· EfficientNetV2")
        st.write("\n")

        


        # All layers frozen (Niels)
        st.write("### **All layers frozen**")



        
        # LAYER UNFREEZING (Lara)
        st.write("### **Unfreezing of layers**")

        # Parameters
        st.write("**Parameters**")
        params = {
        'Dataset size': ['70K'],
        'Augmentation': ['No'],
        'Learning rate': ['0.001'],
        'Image size': ['256x256'],
        'Batch size': [32],
        'Number of Epochs': [50],
        'Test images': [33]
        }

        # Create DataFrame
        parameters = pd.DataFrame(params)
        
        st.markdown(parameters.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")

        # MobileNetV2
        st.markdown("<h2 style='text-align: center; color: black;'>MobileNetV2 </h2>", unsafe_allow_html=True)
        st.markdown("**1) All layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_unfrozen.png")
        st.write("\n")

        
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        st.markdown("**2) Only last block of layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_1Block_unfrozen.png")
        st.write("\n")

                
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_1Block_unfrozen_CM.png")
        st.write("\n")
        st.markdown("""<span style='color: red;'>*No significant change between both trainings</span>""", unsafe_allow_html=True)
        st.write("\n")

        # VGG16
        st.markdown("<h2 style='text-align: center; color: black;'>VGG16 </h2>", unsafe_allow_html=True)
    
        st.markdown("**1) All layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen.png")
        st.write("\n")

        
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        st.markdown("**2) Only last block of layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_1Block_unfrozen.png")
        st.write("\n")

                
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_1Block_unfrozen_CM.png")
        st.write("\n")
        st.markdown("""<span style='color: red;'>*Clear improvement when only the last block of layers is unfrozen</span>""", unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")

        # Summary of metrics
        st.markdown("**Metrics summary**")
        data = {
        'Model': [
        'MobileNetV2 unfrozen', 'MobileNetV2 partly frozen', 
        'VGG16 unfrozen', 'VGG16 partly frozen'
        ],
        'Accuracy': [0.99, 0.99, 0.03, 0.99],
        'Loss': [0.02, 0.02, 3.64, 0.04],
        'Val-accuracy': [0.99, 0.99, 0.03, 0.96],
        'Val-loss': [0.05, 0.05, 3.64, 0.15],
        'Test Prediction Accuracy': [1.00, 0.93, 0.09, 0.88]
        }

        # Create DataFrame with the updated values
        df = pd.DataFrame(data)
        df.set_index ('Model', inplace=True)       
        st.dataframe(df)
        st.write("\n")
        st.write("\n")




        # Modification of other parameters
        st.write("### **Parameter variation for VGG16**")

        st.markdown("**1) Change of learning rate to 0.0001**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_lr10e-4.png")
        st.write("\n")

        st.markdown("**2) Change of input image size to 224x224**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_224.png")
        st.write("\n")

        # Summary of metrics
        st.markdown("**Metrics summary**")
        data2 = {
        'Model': [
            'VGG16 unfrozen', 'VGG16 partly frozen', 'VGG16 unfrozen lr 10E-4', 'VGG16 unfrozen size 224x224'
        ],
        'Accuracy': [0.03, 0.99, 0.99, 1.00],
        'Loss': [3.64, 0.04, 0.05, 0.00],
        'Val-accuracy': [0.03, 0.96, 0.96, 1.00],
        'Val-loss': [3.64, 0.15, 0.14, 0.01],
        'Test Prediction Accuracy': [0.09, 0.88, 0.97, 0.97]
        }
        # Create DataFrame with the updated values
        df2 = pd.DataFrame(data2)
        df2.set_index ('Model', inplace=True)       
        st.dataframe(df2)
        st.write("\n")
        st.write("\n")


        st.markdown("""<span style='color: red;'>*Following the initial parameters used to previously train the VGG16 model provides a dramatic improvement.</span>""", unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")




    with tab3: # Pre-trained models with Pytorch (Yannik)
        plantpad_url = "http://plantpad.samlab.cn"
        st.write("""We used a pre-trained model from [www.plantpad.samlab.cn](%s) which provides models for plant disease diagnosis.
                 These models have been trained on image data (421,314 images) consisting of 63 plant species and 310 kinds of plant diseases. 
                 We employed a ResNet50 model for transfer learning by using it as the base model for feature extraction on which we added a 3-layer classifier CNN.""" % plantpad_url)
        st.write("")
        st.write("")
        st.write("We again used the same inital training parameters as for the TL models in Keras and kept the base model layers frozen:")
        st.markdown(parameters.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.image("src/visualization/Transfer_Learning_PyTorch_ResNet50/ResNet50_all-frozen_lr-1e-3.png")
        st.write("")
        st.write("")
        st.write("""Although the training and validation accuracies are above 0.98, the validation loss shows high fluctuations instead
                 of decreasing. This suggests that the model is not learning properly. As a next step, we decreased the learning rate to 1e-4.""")
        st.write("")
        st.write("")
        st.image("src/visualization/Transfer_Learning_PyTorch_ResNet50/ResNet50_all-frozen_lr-1e-4.png")
        st.write("")
        st.write("")
        st.write("""Lowering the learning rate improves the validation loss on an absolute scale but the fluctuations during training 
                 are still observable. This raises the question wether the architecture itself 
""")

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
    plantpad_url = "http://plantpad.samlab.cn"
    st.write("### Predict your plant")
    st.write("")
    st.write("You can select between three final models:")
    st.markdown("- ResNet50 (PyTorch) based on the pre-trained model from [www.plantpad.samlab.cn](%s)" % plantpad_url)
    st.markdown("- VGG16 (Keras): all layers unfrozen, fine-tuned with a learning rate of 1e-5 and img size of 224x224")
    st.markdown("- MobileNetV2 (Keras): all layers unfrozen, a learning rate of 1e-3 and img size of 256x256")


    # Dropdown menu for selecting a trained model
    model_files = [f for f in os.listdir("src/models/") if f.endswith(".keras") or f.endswith(".pth")]
    selected_model_file = st.selectbox("Select a trained model:", model_files)

    # Drag-and-drop file uploader for image
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  width=300)

        # Load the selected model
        model_path = os.path.join("src/models", selected_model_file)

        # Calculate predictions and handle confidence probabilities
        if selected_model_file.endswith(".keras"):
            model = load_keras_model(model_path)
            preprocessed_image = preprocess_image(image, model)
            predictions = model.predict(preprocessed_image)  # No softmax needed
            predicted_idx = np.argmax(predictions[0])
            top_predictions = predictions[0].argsort()[-5:][::-1]
            probabilities = predictions[0]  # Use raw probabilities as is
        elif selected_model_file.endswith(".pth"):
            model = load_pytorch_model(model_path)
            preprocessed_image = preprocess_image(image, model)
            predictions = model(preprocessed_image).detach().numpy()
            probabilities = np.exp(predictions[0]) / np.sum(np.exp(predictions[0]))  # Apply softmax for PyTorch
            predicted_idx = np.argmax(probabilities)
            top_predictions = probabilities.argsort()[-5:][::-1]


        # Load plant and disease class indices
        plant_indices = load_class_indices("src/streamlit/class_plant_indices.json")
        disease_indices = load_class_indices("src/streamlit/class_disease_indices.json")

        # Display predicted class name in a table
        st.subheader("Model Predictions")
        predicted_plant = plant_indices[str(predicted_idx)]  # Assuming diseases are indexed per plant
        predicted_disease = disease_indices[str(predicted_idx)]
        st.write("**Predicted Plant Type**:", predicted_plant)
        st.write("**Predicted Disease Type**:", predicted_disease)

        # Checkbox for displaying top predictions
        display_top_predictions = st.checkbox("Display top predictions")

        # Display top 5 predictions
        if display_top_predictions:
            st.subheader("Top Predictions")
            sorted_indices = top_predictions  # Already sorted
            seen_combinations = set()  # To avoid duplicate combinations
            top_pred_table = {
                "Plant Type": [],
                "Disease Type": [],
                "Confidence": []
            }
            for idx in sorted_indices:
                plant = plant_indices[str(idx)]
                disease = disease_indices[str(idx)]
                combination = (plant, disease)
                if combination not in seen_combinations:  # Avoid duplicates
                    seen_combinations.add(combination)
                    confidence = probabilities[idx]
                    top_pred_table["Plant Type"].append(plant)
                    top_pred_table["Disease Type"].append(disease)
                    top_pred_table["Confidence"].append(f"{confidence:.2%}")
                if len(top_pred_table["Plant Type"]) == 5:  # Stop at top 5
                    break

            st.table(top_pred_table)


    # Checkbox for Grad-CAM
    display_grad_cam = st.checkbox("Display Grad-CAM")

    # Generate and display Grad-CAM if selected
    if display_grad_cam:
        st.subheader("Grad-CAM Visualization")

        gradcam_model_path = f"src/models/plain_architectures/plain_{selected_model_file}"
        if selected_model_file.endswith(".keras"):
            gradcam_model = load_keras_model(gradcam_model_path)
            grad_cam_image = generate_grad_cam_keras(gradcam_model, image, layer_name=None)
            st.image(grad_cam_image, caption=f"Grad-CAM of {selected_model_file}", width=300)


        elif selected_model_file.endswith(".pth"):
            gradcam_model = load_pytorch_model(model_path)
            grad_cam_image = generate_grad_cam_pytorch(gradcam_model, image)
            st.image(grad_cam_image, caption=f"Grad-CAM of {selected_model_file}", width=300)

    