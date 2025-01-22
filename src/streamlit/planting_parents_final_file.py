import streamlit as st
import os
import pandas as pd

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
        st.image("src/visualization/Data_Exploration/Plants_&_Diseases_count.png", use_container_width=True)
    
    with tab2: # Plants & diseases
        st.write("")
        st.image("src/visualization/Data_Exploration/diseased_plant_samples.png")

        st.write("")
        st.image("src/visualization/Data_Exploration/healthy_plant_samples.png")

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
    
    st.write("#### Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["General distribution", "Train vs validation ditribution", "Confusion matrix", "Data Transformation"])


    with tab1:
        st.markdown("- Species")
        st.image("src/visualization/Data_Exploration/DExp_species_distribution.png")

        st.markdown("- Diseases")
        st.image("src/visualization/Data_Exploration/DExp_diseases_distribution.png")
    
        st.markdown("""<span style='color: red;'>*The data shows a reasonable inbalance with the tomato class, but the number of total images for the other categories is high enough to use all of them for the modelling.</span>""", unsafe_allow_html=True)

    with tab2:
        
        st.markdown("- Species")
        st.image("src/visualization/Data_Exploration/Dexp_TrainValid_species.png")

        st.markdown("- Diseases")
        st.image("src/visualization/Data_Exploration/DExp_TrainValid_diseases.png")

    with tab3:
        st.write("Confusion matrix between species and diseases")
        st.image("src/visualization/Data_Exploration/DExp_CM_species_diseases.png")
        st.markdown("""<span style='color: red;'>*Some species do not contain any disease examples and some species do not contain healthy examples.</span>""", unsafe_allow_html=True)


    with tab4:
        st.write("The data can be filtered in order to remove background or detect edges")
        st.markdown("1. Thresholding")
        st.image("src/visualization/Data_Exploration/DExp_thresholding_filter.png")

        st.markdown("2. Canny filtering")
        st.image("src/visualization/Data_Exploration/DExp_Canny_filter.png")

####################
# MODEL TRAINING   #
####################


elif page == pages[3]:
    st.write("### Model Training")
    
    tab1, tab2, tab3 = st.tabs(["Baseline CNN's", "Transfer Learning", "Pre-trained PyTorch"])


    with tab1: # Baseline CNN's (Ji)
        st.write("### Convolutional Neural Networks (CNNs) ")
        st.write('''
        In this section, we investigate the performance of Convolutional Neural Networks (CNNs), 
        a fundamental model choice for image analysis tasks 
        because of its ability to efficiently identify and interpret patterns and structures 
        at multiple levels of detail within visual data.
        ''')
        st.write('''
        To optimize the performance of our CNN model, we evaluated several parameters \
        that influence accuracy, generalization, and computational efficiency. 
        These parameters include dataset size, image resolution, learning rate, 
        data augmentation techniques, and the depth of the CNN architecture 
        (number of convolutional layers). By varying these parameters, 
        we aimed to identify the configurations that maximize model performance 
        for our image classification task.
        ''')
            

        st.write("### CNN Model Architecture ")
        st.write('''
        1. Convolutional Layers:
        These layers apply filters to the input data to extract features such as edges, textures, and other patterns. They help the network detect spatial features from images at different levels of abstraction.

        2. Batch Normalization Layers:
        These layers normalize the inputs of each layer to have a mean of 0 and a variance of 1. This speeds up training, stabilizes the learning process, and helps reduce the risk of overfitting.

        3. Max-Pooling Layers:
        These layers reduce the spatial dimensions of the feature maps by taking the maximum value from a pooling window (e.g., 2x2). This makes the model computationally efficient and helps it focus on the most important features while reducing sensitivity to spatial translations.

        4. Fully Connected Layers:
        These layers flatten the feature maps into a 1D vector and integrate the extracted features to perform the final classification or regression tasks. They are responsible for the decision-making process in the network.
                  ''')
        st.write('''

                 ''')
        st.write("### CNN Model: 2-layers (CNN-2x) vs 3-layers (CNN-3x) ")
        # Option 1
        col1, col2 = st.columns(2)
        with col1:
            st.image("Model-architecture_CNN-2.png", use_container_width=True)
        with col2:
            st.image("Model-architecture_CNN-3.png", use_container_width=True)
        # Option 2
        # st.image("Model-architecture_CNN-2.png", use_container_width=True)
        # st.image("Model-architecture_CNN-3.png", use_container_width=True)

############### Tab 1 - subtab ##################
        # subtab
        sub_tab = st.radio("Test parameters", 
                           ["Dataset size", "Image size", "Learning rate", "Augmentation","CNN layer"])
############### Tab 1 - subtab 1 ##################        
        if sub_tab == "Dataset size":
            st.write("In this section, we tested different pre-trained models")
            st.image("1_CNN_Dataset_graph+cm.png", use_container_width=True)
############### Tab 1 - subtab 2 ##################   
        elif sub_tab == "Image size":
            st.write("In this section, we tested different image sizes.")
############### Tab 1 - subtab 3 ##################  
        elif sub_tab == "Learning rate":
            st.write("In this section, we experimented with different learning rates.")
############### Tab 1 - subtab 4 ##################  
        elif sub_tab == "Augmentation":
            st.write("In this section, we tested various data augmentation techniques.")
############### Tab 1 - subtab 5 ##################  
        elif sub_tab == "CNN layer":
            st.write("In this section, we tested various data augmentation techniques.")



############### Tab 2 ##################
    with tab2: # Transfer Learning (Niels and Lara)
        st.write("In this section, we tested different pre-trained models")
        st.write("")


    with tab2: # Transfer Learning
        st.write("**Pre-trained models**")
        st.markdown("· MobileNetV2")
        st.markdown("· VGG16")
        st.markdown("· ResNet 101")
        st.markdown("· ResNet 50")
        st.markdown("· EfficientNetV2")
        st.write("\n")

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

        # All layers frozen (Niels)
        st.write("### **All layers frozen**")
        
        # MobileNetV2
        st.markdown("<h2 style='text-align: center; color: green;'>MobileNetV2 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/MobileNetV2_13-01-2025_Niels.png")
        st.write("\n")

        # VGG16
        st.markdown("<h2 style='text-align: center; color: green;'>VGG16 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/TL_VGG16_13-01-2025_Niels.png")
        st.write("\n")

        # ResNet 101
        st.markdown("<h2 style='text-align: center; color: green;'>ResNet 101 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/ResNet101_13-01-2025_Niels.png")
        st.write("\n")
        
        # ResNet 50
        st.markdown("<h2 style='text-align: center; color: green;'>ResNet 50 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/ResNet50_17-01-2025_Niels.png")
        st.markdown("""<span style='color: red;'>*Only 10 epochs were run on this model</span>""", unsafe_allow_html=True)
        st.write("\n")

        # EfficientNetV2-L
        st.markdown("<h2 style='text-align: center; color: green;'>EfficientNetV2-L </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/EfficientNetV2L_Niels_13.01.2025.png")
        st.markdown("""<span style='color: red;'>*Only 22 epochs were run on this model</span>""", unsafe_allow_html=True)
        st.write("\n")

        # Summary of metrics
        st.markdown("**Metrics summary**")
        data = {
        'Model': [
        'MobileNetV2', 'VGG16', 
        'ResNet 101', 'ResNet 50', 'EfficientNetV2-L'
        ],
        'Accuracy': [0.99, 0.97, 0.51, 0.39, 0.26],
        'Loss': [0.08, 0.09, 1.60, 2.00, 2.58],
        'Val-accuracy': [0.97, 0.96, 0.56, 0.48, 0.39],
        'Val-loss': [0.23, 0.14, 1.51, 1.75, 2.08],
        'Test Prediction Accuracy': [0.97, 0.97, "n.d.", "n.d.", "n.d."]
        }

        # Create DataFrame with the updated values
        df = pd.DataFrame(data)
        df.set_index ('Model', inplace=True)       
        st.dataframe(df)
        st.write("\n")
        st.write("\n")



        # LAYER UNFREEZING (Lara)
        st.write("### **Unfreezing of layers**")

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
        st.write("")

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
    st.write("### Upload an image to predict the plant type")
    st.write("This subpage should contain the actual app. Here, the user should chose")
    st.checkbox("between different models")
    st.checkbox("wether or not a Grad-CAM of the image should be shown")
    