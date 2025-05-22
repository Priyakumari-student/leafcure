
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Leaf Disease Detection - PlantVillage (38 Classes)")
model = load_model("model/leaf_disease_model.h5")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    img_array = np.expand_dims(np.array(image)/255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    st.write(f"Predicted Class: {predicted_class}")
