import streamlit as st

import tensorflow as tf

import numpy as np

from PIL import Image



# Page config

st.set_page_config(

    page_title="Lung Cancer Detection",

    page_icon="🫁",

    layout="centered"

)



# Model load

@st.cache_resource

def load_model():

    model = tf.keras.models.load_model('model/best_model_v3.h5')

    return model



model = load_model()



# Class names

class_names = ['Benign', 'Malignant', 'Normal']

class_colors = {

    'Benign': '🟡',

    'Malignant': '🔴',

    'Normal': '🟢'

}

class_info = {

    'Benign': 'Benign tumor detected. Not immediately life-threatening but requires monitoring.',

    'Malignant': '⚠️ Malignant cancer detected. Immediate medical consultation recommended.',

    'Normal': 'No cancer detected. Lungs appear normal.'

}



# Header

st.title("🫁 Lung Cancer Detection System")

st.markdown("**Automated Detection and Classification from CT Scan Images Using Deep CNN**")

st.divider()



# Sidebar

with st.sidebar:

    st.header("ℹ️ About")

    st.write("This system uses VGG16-based CNN to detect and classify lung cancer from CT scan images.")

    st.divider()

    st.write("**Classes:**")

    st.write("🟢 Normal — No cancer")

    st.write("🟡 Benign — Non-cancerous tumor")

    st.write("🔴 Malignant — Cancerous tumor")

    st.divider()

    st.write("**Model:** VGG16 Transfer Learning")

    st.write("**Accuracy:** 80.72%")

    st.write("**Dataset:** IQ-OTHNCCD")



# Upload

st.subheader("📤 Upload CT Scan Image")

uploaded_file = st.file_uploader(

    "Choose a CT scan image",

    type=['jpg', 'jpeg', 'png']

)



if uploaded_file is not None:

    col1, col2 = st.columns(2)



    with col1:

        st.subheader("📷 Uploaded Image")

        image = Image.open(uploaded_file).convert('RGB')

        st.image(image, width='stretch')



    # Preprocess

    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)



    # Predict

    with st.spinner("Analyzing..."):

        predictions = model.predict(img_array)

        predicted_class = class_names[np.argmax(predictions)]

        confidence = np.max(predictions) * 100



    with col2:

        st.subheader("🔬 Prediction Result")

        st.markdown(f"### {class_colors[predicted_class]} {predicted_class}")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.info(class_info[predicted_class])



        st.subheader("📊 All Probabilities")

        for i, cls in enumerate(class_names):

            prob = predictions[0][i] * 100

            st.progress(int(prob), text=f"{cls}: {prob:.2f}%")



    st.divider()

    st.warning("⚠️ This tool is for research purposes only. Always consult a medical professional.")