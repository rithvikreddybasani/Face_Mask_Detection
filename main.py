import numpy as np
from tensorflow import keras
import streamlit as st
from PIL import Image

model = keras.models.load_model("mask_model.h5")

st.title('Face Mask Detection')
st.write('Upload a single image or multiple images to predictions!')

uploaded_images = st.file_uploader("Upload your images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image)
        
    
        image = image.convert('RGB')  
        image = image.resize((128, 128))
        
        
        image_array = np.array(image) / 255.0
        
    
        processed_image = np.reshape(image_array,[1,128,128,3])
        
    
        prediction_new_image = model.predict(processed_image)
        highest_index = np.argmax(prediction_new_image)

        
        st.image(uploaded_image, caption='Uploaded Image.', width=200)
        if highest_index == 0:
            st.write('The person in the image is not wearing a mask')
        else:
            st.write('The person in the image is wearing a mask')
