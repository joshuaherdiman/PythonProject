import streamlit as st
from fastai.vision.all import *

st.title("Fish Species Identification")
st.text("Built by Joshua")

fish_model = load_learner("fish_species_prediction_model.pkl")

def extract_images(file_name):
    p = Path(file_name)
    species_name_parts = p.stem.split("_")

    final_species_name = " "
    length_parts = len(species_name_parts)-1
    for i in range(length_parts):
        final_species_name += species_name_parts[i]
        if i != length_parts:
            final_species_name += "_"

    return final_species_name

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = fish_model.predict(img)
    return pred_class

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"Prediction Result:")

st.text("Built with Streamlit and Fastai")