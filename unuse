import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image

# Đường dẫn tới mô hình đã được huấn luyện
model_path = 'NhanDienChoMeo.h5'

# Load mô hình đã được huấn luyện
model = load_model(model_path)

# Dictionary để ánh xạ từ chỉ mục lớp sang tên lớp
class_names = {0: 'cat', 1: 'dog'}

# Kích thước hình ảnh đầu vào
input_shape = (64, 64, 3)

def preprocess_image(image, input_shape):
    # Tiền xử lý hình ảnh
    img = Image.fromarray(image)
    img = img.resize(input_shape[:2])
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return img

def classify_image(image):
    # Tiền xử lý hình ảnh và dự đoán
    image = preprocess_image(image, input_shape)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    
    return class_name


# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("PhamKimDon_210501025")

# Đặt kích thước cửa sổ
window.geometry("500x500")

# Đặt màu nền và font chữ cho cửa sổ giao diện
window.configure(bg='white')
title_font = ('Arial', 14, 'bold')

# Tạo nhãn tên đề tài
title_label = tk.Label(window, text="Nhận diện chó, mèo", font=title_font, bg='white', fg='black')
title_label.pack(pady=10)

# Hàm xử lý sự kiện khi nhấp vào nút "Upload Image"
def browse_image():
    # Chọn hình ảnh từ máy tính
    file_path = filedialog.askopenfilename()
    if file_path:
        # Hiển thị hình ảnh đã chọn
        img = Image.open(file_path)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        # Nhận dạng và hiển thị kết quả
        img_cv = cv2.imread(file_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Chuyển đổi không gian màu BGR sang RGB
        class_name = classify_image(img_cv)
        result_label.configure(text="This is a " + class_name)
        
#Hàm xử lý sự kiện khi nhấp vào nút "Open Camera"
def open_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán lớp của hình ảnh từ camera
        class_name = classify_image(frame)
        
        # Hiển thị kết quả trên video từ camera
        confidence = 1
        cv2.putText(frame, f'{class_name}: {confidence * 100:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tạo nút "Upload Image" để chọn hình ảnh từ máy tính
upload_button = tk.Button(window, text="Upload Image", command=browse_image)
upload_button.pack(pady=10)

# Tạo nút "Open Camera" để mở camera và nhận diện trực tiếp từ camera
camera_button = tk.Button(window, text="Open Camera", command=open_camera)
camera_button.pack(pady=10)

# Tạo nhãn để hiển thị kết quả
result_label = tk.Label(window, text="", font=('Arial', 12), bg='white', fg='black')
result_label.pack(pady=10)

# Tạo nhãn để hiển thị hình ảnh
image_label = tk.Label(window, bg='white')
image_label.pack(pady=10)

# Chạy giao diện
window.mainloop()