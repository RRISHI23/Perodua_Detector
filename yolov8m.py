import streamlit as st
import os
from PIL import Image, ExifTags
from ultralytics import YOLO
import numpy as np
import torch

# Paths to models
yolo_model_path = "best.pt"  # YOLOv8 detection model
alza_classification_model_path = "alzaxveloz.pt"  # Alza classification model
aruz_classification_model_path = "aruzxrush.pt"  # Aruz classification model

# Load YOLOv8 models
yolo_model = YOLO(yolo_model_path)  # YOLOv8 detection model
alza_classification_model = YOLO(alza_classification_model_path)  # Alza classification model
aruz_classification_model = YOLO(aruz_classification_model_path)  # Aruz classification model

# Classes for YOLOv8 object detection
YOLO_CLASSES = ['Alza', 'Aruz', 'Myvi', 'Axia', 'Bezza']

# Classes for the classification models
CLASSIFICATION_CLASSES = {
    "alza": ['Alza', 'Veloz'],
    "aruz": ['Aruz', 'Rush']
}

# Confidence threshold for YOLO detection
YOLO_CONFIDENCE_THRESHOLD = 0.87

# Prediction function for YOLOv8
def detect_with_yolo(img_path, model):
    results = model.predict(img_path)
    predictions = results[0]
    if len(predictions.boxes) > 0:
        box = predictions.boxes[0]
        class_id = int(box.cls)
        confidence = box.conf.item()
        return class_id, confidence
    return None, 0.0

def classify_car_model(img_path, model, classification_classes):
    results = model.predict(img_path)
    probabilities = results[0].probs
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    elif hasattr(probabilities, "data"):
        probabilities = probabilities.data
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
    else:
        raise ValueError(f"Unsupported probabilities type: {type(probabilities)}")
    class_idx = np.argmax(probabilities)
    confidence = probabilities[class_idx]
    return classification_classes[class_idx], confidence

# Streamlit UI
st.set_page_config(
    page_title="Perodua Car Model Detector",
    page_icon="🚗",
    layout="wide",
)

# Sidebar Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Perodua Detector", "Car Models", "About Me", "FAQs"])

if page == "Perodua Detector":
    st.title("🚗 Perodua Car Model Detector")
    st.subheader("Identify your Perodua car model using AI.")

    # Add instructions for the user
    st.subheader("Follow these steps to predict your Perodua Car Model:")
    st.write("1. **Click Browse files**")
    st.write("2. **Upload your preferred Perodua Car Model** (Alza, Aruz, Myvi, Axia, or Bezza)")
    st.write("3. **Click the Predict Car Model** button to get your prediction.")
    st.markdown("💡 **Note:** Heavily modified cars will not be detected.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a Car Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image temporarily
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Correct the orientation using EXIF data (if available)
        img = Image.open(uploaded_file)
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None and orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        # Resize and display the corrected image
        img_resized = img.resize((224, 224))
        st.image(img_resized, caption="Uploaded Image")

        # Predict button
        if st.button("🔍 Predict Car Model"):
            status_placeholder = st.empty()  # Create a placeholder for the status message
            status_placeholder.markdown("<h4 style='color: #1abc9c;'>Analyzing the image...</h4>", unsafe_allow_html=True)

            # Detect with YOLOv8
            class_id, confidence = detect_with_yolo(file_path, yolo_model)

            # Update the status to "Analyzed" after processing
            status_placeholder.markdown("<h4 style='color: #1abc9c;'>Analyzed</h4>", unsafe_allow_html=True)

            if confidence < YOLO_CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ Detected as Unknown. Confidence level: {confidence * 100:.2f}%")
            elif class_id is not None:
                detected_class = YOLO_CLASSES[class_id]
                st.write(f"🚗 Detected: **Perodua {detected_class}** with {confidence * 100:.2f}% confidence.")

                # Further classify for specific models
                if detected_class == "Alza":
                    st.write("🔍 Performing detailed classification for Alza...")
                    final_class, final_confidence = classify_car_model(file_path, alza_classification_model, CLASSIFICATION_CLASSES["alza"])
                    st.write(f"🔹 Final Classification: **Perodua {final_class}** with {final_confidence * 100:.2f}% confidence.")

                elif detected_class == "Aruz":
                    st.write("🔍 Performing detailed classification for Aruz...")
                    final_class, final_confidence = classify_car_model(file_path, aruz_classification_model, CLASSIFICATION_CLASSES["aruz"])
                    st.write(f"🔹 Final Classification: **Perodua {final_class}** with {final_confidence * 100:.2f}% confidence.")

                elif detected_class in ["Myvi", "Axia", "Bezza"]:
                    st.write(f"🔹 Perodua {detected_class} detected.")
            else:
                st.warning(f"⚠️ No car detected. Confidence level: {confidence * 100:.2f}%")

        # Clean up the uploaded file
        os.remove(file_path)
    else:
        st.info("Please upload an image to start the prediction.")

elif page == "Car Models":
    st.title("🚘 Perodua Car Models")
    models_info = {
        "Alza": (
            "The Perodua Alza is a versatile family MPV designed for comfort and practicality. Known for its spacious interior, "
            "it seats up to seven passengers, making it perfect for family trips. The Alza comes equipped with advanced safety "
            "features such as ABS and EBD and offers a smooth and fuel-efficient drive."
        ),
        "Aruz": (
            "The Perodua Aruz is a sporty SUV built for adventure. With a robust design, it handles both city roads and rough terrains "
            "with ease. Offering spacious seating for seven, the Aruz includes features like Vehicle Stability Control and Hill Start Assist."
        ),
        "Myvi": (
            "The Perodua Myvi is Malaysia's favorite compact car. It combines fuel efficiency, practicality, and a stylish design. "
            "Its compact size makes it ideal for city driving while still offering a comfortable interior for passengers."
        ),
        "Axia": (
            "The Perodua Axia is an affordable hatchback known for its exceptional fuel efficiency. Its compact design makes it "
            "perfect for urban driving, while its modern features ensure a comfortable ride."
        ),
        "Bezza": (
            "The Perodua Bezza is a sleek sedan offering great mileage and comfort. With advanced safety features and ample trunk space, "
            "it's a reliable choice for families and professionals alike."
        ),
    }

    for model, description in models_info.items():
        st.subheader(f"🚗 {model}")
        st.write(description)

elif page == "About Me":
    st.title("📖 About Me")
    st.markdown(
        """
        The automotive industry is witnessing a remarkable transformation driven by advancements in artificial intelligence and deep learning. 
        This project explores the exciting world of YOLOv8, a state-of-the-art deep learning framework for object detection and recognition.

        The system initially recognizes the existence of cars in the provided data and then classifies the precise model of each detected car. 
        By leveraging deep neural networks, inspired by the human brain, we achieve unprecedented accuracy in car detection and model recognition.

        This project includes a user-friendly web interface that allows users to upload images and receive predictions about car models. 
        The project not only highlights advancements in AI and deep learning but also sets the stage for future innovations in the automotive industry.
        """
    )

elif page == "FAQs":
    st.title("📋 FAQs")
    st.write("1. **What types of images can I upload?**")
    st.write("   You can upload images in JPG, JPEG, or PNG format. Ensure the image is well-lit, not blurry, and contains the full car for better detection.")
    st.write("2. **What car models does the app detect?**")
    st.write("   The app detects the following Perodua models: Alza, Aruz, Myvi, Axia, Bezza.")
    st.write("3. **Can the app detect modified cars?**")
    st.write("   Heavily modified cars may not be detected accurately as they deviate from the trained dataset.")
    st.write("4. **What should I do if the app doesn't detect my car?**")
    st.write("   Ensure the image is clear and well-lit, contains the full car without obstructions, and matches one of the supported Perodua models.")
    st.write("5. **Can I upload multiple images at once?**")
    st.write("   No, the app only supports one image upload at a time. Please upload images one by one.")
    st.write("6. **Why is the detected car model confidence level low?**")
    st.write("   A low confidence level can occur due to poor image quality, obstructed views, or untrained models.")
