import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import random

@st.fragment
def AnimalRecognize():
    data = 'data'
    classes = []
    classes_dir = []

    image_shape = (64, 64, 3)

    model_path = 'NhanDienChoMeo.h5'
    model = load_model(model_path)

    for folderNames in os.listdir(data):
        classes.append(folderNames)
        classes_dir.append(os.path.join(data, folderNames))

    st.title('AniReco')
    st.subheader('Chọn một ảnh giống với con vật này')
    uploaded_file = st.file_uploader('', type=['jpg', 'png', 'jpeg'], label_visibility='collapsed')

    def classify_image(image):
        # Tiền xử lý hình ảnh và dự đoán
        image = preprocess_image(image, image_shape)
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        class_name = classes[class_index]
        return class_name

    def preprocess_image(image, input_shape):
        # Tiền xử lý hình ảnh
        img = Image.fromarray(image)
        img = img.resize(input_shape[:2])
        img = np.array(img) / 255.0
        img = img[np.newaxis, ...]
        return img

    # Define a fixed size for both images
    IMAGE_SIZE = (300, 300)

    # Adjust the ratio as needed, e.g., [3, 2] for a wider left column
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        def choose_random_image():
            random_class = random.choice(classes)
            random_image = random.choice(os.listdir(os.path.join(data, random_class)))
            image_path = os.path.join(data, random_class, random_image)
            img = Image.open(image_path)
            st.session_state.reference_image = img.resize(IMAGE_SIZE)
            st.session_state.reference_class = random_class

        if 'reference_image' not in st.session_state or 'reference_class' not in st.session_state:
            choose_random_image()
        
        st.image(st.session_state.reference_image, caption='Hình ảnh tham chiếu', use_column_width=True)

        if st.button('Chọn ảnh ngẫu nhiên') and st.session_state.reference_image:
            choose_random_image()
            st.rerun(scope='fragment')

    with col2:

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.session_state.user_image = img.resize(IMAGE_SIZE)
            st.image(st.session_state.user_image, caption='Hình ảnh đã chọn', use_column_width=True)

    # Add this dictionary with Vietnamese translations
    vietnamese_classes = {
        'dogs': 'Chó',
        'cats': 'Mèo',
        'chickens': 'Gà',
        'pigs': 'Heo',
        'ducks': 'Vịt',
        # Add more translations as needed
    }

    @st.dialog("Cùng xem nhé!")
    def dialogVideo():
        video_file = open(f"videos\\{reference_class}\\video_1.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes, autoplay=True)

    with col3:

        if st.button('Nhận dạng và So sánh'):
            if uploaded_file is None:
                st.warning('Vui lòng chọn một hình ảnh trước khi nhận dạng.')
            else:
                user_class = classify_image(np.array(st.session_state.user_image))
                reference_class = st.session_state.reference_class
                
                if user_class == reference_class:
                    vietnamese_class = vietnamese_classes.get(user_class, user_class)
                    st.success(f'Chúc mừng! Bạn đã chọn đúng loại. Đây là một con {vietnamese_class}.')
                    st.button("Cùng xem video con vật này nhé!", on_click=dialogVideo)
                else:
                    st.error('Ôi không, sai rồi. Hãy chọn lại ảnh ngẫu nhiên khác hoặc chọn một ảnh khác.')

def main_app():
    AnimalRecognize()

def welcome_page():
    st.title("Chào mừng đến với AniReco!")
    st.write("Hãy nhập tên của bạn để bắt đầu.")
    
    def start_app():
        if st.session_state.name:
            st.session_state.user_name = st.session_state.name
            st.session_state.page = "main"
            #st.rerun()
        else:
            st.warning("Vui lòng nhập tên của bạn.")
    
    # Use a session state variable to store the name
    st.text_input("Tên của bạn:", key="name", on_change=start_app)
    
    if st.button("Bắt đầu"):
        start_app()

# Main app logic
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "main":
        st.sidebar.write(f"Xin chào, {st.session_state.user_name}!")
        main_app()
