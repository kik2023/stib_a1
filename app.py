import streamlit as st
import cv2
import joblib
from PIL import Image
import numpy as np

model = joblib.load("trained_model_KNN-Copy3.pkl")

st.image('753728_poster.jpg',width=300)
st.title('This doggo SEES through your cracks')
st.text('Crack Detector with KNN')

# Preprocessing and feature extraction
def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges
 
def extract_features(img):
    resized = cv2.resize(img, (64, 64))
    return resized.flatten().reshape(1, -1)

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    st.image(image, width=300, caption = "Photo well received")

    processed_image = preprocess_image(img_array)
    features = extract_features(processed_image)
    
    prediction = model.predict(features)[0]
    label = "Positive" if prediction == 1 else "Negative"
    st.success(f"**Prediction:** {label}")
