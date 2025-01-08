import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torch  # Ensure PyTorch is imported

# Paths to models
yolo_model_path = 'C:/Users/Rrish/runs/detect/train5/weights/best.pt'  # YOLOv8 detection model
alza_classification_model_path = "C:/Users/Rrish/OneDrive/Desktop/FYP/alzaxveloz.pt"  # Alza-specific classification model
aruz_classification_model_path = "C:/Users/Rrish/OneDrive/Desktop/FYP/aruzxrush.pt"  # Aruz-specific classification model

# Load YOLOv8 models
yolo_model = YOLO(yolo_model_path)  # YOLOv8 detection model
alza_classification_model = YOLO(alza_classification_model_path)  # Alza-specific classification model
aruz_classification_model = YOLO(aruz_classification_model_path)  # Aruz-specific classification model

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
    results = model.predict(img_path)  # Run inference
    predictions = results[0]  # Get results for the first image
    if len(predictions.boxes) > 0:  # If objects detected
        box = predictions.boxes[0]
        class_id = int(box.cls)
        confidence = box.conf.item()
        return class_id, confidence
    return None, 0.0  # No detection with 0 confidence

def classify_car_model(img_path, model, classification_classes):
    # Use YOLOv8's predict method directly for classification
    results = model.predict(img_path)  # YOLOv8 automatically processes the image

    # Extract probabilities from results[0].probs
    probabilities = results[0].probs

    # Convert `Probs` to a NumPy array or Tensor
    if isinstance(probabilities, torch.Tensor):  # Handle PyTorch tensor
        probabilities = probabilities.cpu().numpy()  # Convert to NumPy
    elif hasattr(probabilities, "data"):  # Handle Probs or similar object
        probabilities = probabilities.data
        if isinstance(probabilities, torch.Tensor):  # Convert if PyTorch tensor
            probabilities = probabilities.cpu().numpy()
    else:
        raise ValueError(f"Unsupported probabilities type: {type(probabilities)}")

    # Get the class with the highest probability
    class_idx = np.argmax(probabilities)
    confidence = probabilities[class_idx]

    return classification_classes[class_idx], confidence

# Streamlit UI
st.set_page_config(
    page_title="Perodua Car Model Detector",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Perodua Car Model Detector")
st.subheader("Identify your Perodua car model using AI.")

# File uploader
uploaded_file = st.file_uploader("Upload a Car Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Resize and display the uploaded image
    img = Image.open(uploaded_file)
    img_resized = img.resize((224, 224))  # Resize the image for display
    st.image(img_resized, caption="Uploaded Image", use_container_width=False)

    # Predict button
    if st.button("🔍 Predict Car Model"):
        st.markdown("<h4 style='color: #1abc9c;'>Analyzing the image...</h4>", unsafe_allow_html=True)

        # Detect with YOLOv8
        class_id, confidence = detect_with_yolo(file_path, yolo_model)

        if confidence < YOLO_CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Detected as Unknown. Confidence level: {confidence * 100:.2f}%")
        elif class_id is not None:
            detected_class = YOLO_CLASSES[class_id]
            st.write(f"🚗 Detected: **{detected_class}** with {confidence * 100:.2f}% confidence.")

            # Further classify for specific models
            if detected_class == "Alza":
                st.write("🔍 Performing detailed classification for Alza...")
                final_class, final_confidence = classify_car_model(file_path, alza_classification_model, CLASSIFICATION_CLASSES["alza"])
                st.write(f"🔹 Final Classification: **{final_class}** with {final_confidence * 100:.2f}% confidence.")

            elif detected_class == "Aruz":
                st.write("🔍 Performing detailed classification for Aruz...")
                final_class, final_confidence = classify_car_model(file_path, aruz_classification_model, CLASSIFICATION_CLASSES["aruz"])
                st.write(f"🔹 Final Classification: **{final_class}** with {final_confidence * 100:.2f}% confidence.")

            elif detected_class in ["Myvi", "Axia", "Bezza"]:
                st.write(f"🔹 {detected_class} detected.")
        else:
            st.warning(f"⚠️ No car detected. Confidence level: {confidence * 100:.2f}%")

    # Clean up the uploaded file
    os.remove(file_path)
else:
    st.info("Please upload an image to start the prediction.")

# Footer
st.markdown(
    """
    <hr style="border:1px solid #16a085;">
    <footer style="text-align:center; color:gray; font-size:14px;">
        <p>Powered by YOLOv8 | Designed for Perodua Enthusiasts</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
