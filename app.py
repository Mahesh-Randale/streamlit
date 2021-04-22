import streamlit as st
from classifier import sign
from PIL import Image, ImageOps

st.title("Sign Language Recognition using CNN")
#st.header("Brain Tumor MRI Classification Example")
st.text("Upload Sign gesture")
uploaded_file = st.file_uploader("Choose a sign  ...", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded sign.', use_column_width=True)
	st.write("")
	st.write("Classifying...")
	label = sign(image, "model-bw.h5")
	st.write("Done")
	st.write(label)