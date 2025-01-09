import streamlit as st
import os
from PIL import Image
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
    # Use YOLOv8's predict method directly for classification
    results = model.predict(img_path)  

    # Extract probabilities from results[0].probs
    probabilities = results[0].probs

    # Convert `Probs` to a NumPy array or Tensor
    if isinstance(probabilities, torch.Tensor):  
        probabilities = probabilities.cpu().numpy()  
    elif hasattr(probabilities, "data"):  
        probabilities = probabilities.data
        if isinstance(probabilities, torch.Tensor):  
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


# Add instructions for the user
st.subheader("Follow these steps to predict your Perodua Car Model:")
st.write("1. **Click Browse files**")
st.write("2. **Upload your preferred Perodua Car Model** (Alza, Aruz, Myvi, Axia, or Bezza)")
st.write("3. **Click the Predict Car Model** button to get your prediction.")


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
    st.image(img_resized, caption="Uploaded Image")

    # Predict button
    if st.button("🔍 Predict Car Model"):
        st.markdown("<h4 style='color: #1abc9c;'>Analyzing the image...</h4>", unsafe_allow_html=True)

        # Detect with YOLOv8
        class_id, confidence = detect_with_yolo(file_path, yolo_model)

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
