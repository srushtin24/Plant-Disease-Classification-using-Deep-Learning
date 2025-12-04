# from google.cloud import storage
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# model = None
# interpreter = None
# input_index = None
# output_index = None

# class_names = ["Early Blight", "Late Blight", "Healthy"]

# BUCKET_NAME = "potato-disease-mp" # Here you need to put the name of your GCP bucket


# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     blob.download_to_filename(destination_file_name)

#     print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


# def predict(request):
#     global model
#     if model is None:
#         download_blob(
#             BUCKET_NAME,
#             "models/potatoes3.h5",
#             "/tmp/potatoes3.h5",
#         )
#         model = tf.keras.models.load_model("/tmp/potatoes3.h5", compile=True)

#     image = request.files["file"]

#     image = np.array(
#         Image.open(image).convert("RGB").resize((256, 256)) # image resizing
#     )

#     image = image # normalize the image in 0 to 1 range

#     img_array = tf.expand_dims(image, 0)
#     predictions = model.predict(img_array)

#     print("Predictions:",predictions)

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)

#     return {"class": predicted_class, "confidence": confidence}


from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Global variables for model loading
model = None

# --- Configuration ---
BUCKET_NAME = "potato-disease-mp" 
MODEL_BLOB = "models/potatoes4.h5"
LOCAL_MODEL_PATH = "/tmp/potatoes4.h5"
IMAGE_SIZE = (256, 256)

class_names = [
    "Potato___Early_blight", 
    "Potato___Late_blight", 
    "Potato___healthy"
]

# --- DISEASE INFORMATION DATA STORE (GLOBAL) ---
# This data is external to the model and must be defined here.
disease_info = {
    "Potato___Early_blight": {
        "Description": "Early Blight (Alternaria solani) appears as dark, concentric rings (bull's-eye pattern) on older, lower leaves. It spreads slowly and is treatable.",
        "Symptoms": "Dark, target-like spots with distinct concentric rings, usually surrounded by a yellow halo.",
        "Prevention": "Rotate crops annually. Use fungicides preventatively. Ensure good air circulation.",
        "Treatment": "Apply fungicides containing chlorothalonil or copper-based treatments." # <-- FOCUS ON THIS
    },
    "Potato___Late_blight": {
        "Description": "Late Blight (Phytophthora infestans) is highly destructive, appearing as irregular dark, water-soaked spots. Requires time-sensitive, aggressive treatment.",
        "Symptoms": "Large, irregular, black/brown, water-soaked spots that quickly spread. Fuzzy white mold may appear on leaf undersides in high humidity.",
        "Prevention": "Plant resistant potato varieties. Destroy infected debris immediately. Use wider plant spacing.",
        "Treatment": "Requires aggressive treatment with systemic fungicides (e.g., mefenoxam) immediately upon identification." # <-- FOCUS ON THIS
    },
    "Potato___healthy": {
        "Description": "The potato plant is vigorous and free from common diseases. Focus on continued maintenance.",
        "Symptoms": "Vibrant green leaves, full turgor, and strong plant structure.",
        "Prevention": "Maintain balanced soil nutrition and consistent watering practices.",
        "Treatment": "No treatment required. Focus on continued monitoring and good care." # <-- FOCUS ON THIS
    }
}


# --- Helper Function for GCS Download (Remains the same) ---
def download_blob(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        return True
    except Exception as e:
        print(f"Error downloading blob {source_blob_name}: {e}")
        return False

# --- Prediction Function (Entry Point) ---
def predict(request):
    """
    HTTP Cloud Function to classify a submitted potato leaf image, 
    returning ONLY the class, confidence, and treatment.
    """
    global model

    # 1. Model Loading
    if model is None:
        if not download_blob(BUCKET_NAME, MODEL_BLOB, LOCAL_MODEL_PATH):
             return {"error": "Failed to load potato model from GCS."}, 500
        
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH) 
        print("Potato ResNet model loaded into memory.")


    # 2. Image Handling
    if "file" not in request.files:
        return {"error": "Missing file upload. Please submit a file under the key 'file'."}, 400
        
    image_file = request.files["file"]

    image = np.array(
        Image.open(image_file).convert("RGB").resize(IMAGE_SIZE)
    )
    
    img_array = tf.expand_dims(image.astype('float32'), 0)
    
    # 3. Prediction
    predictions = model(img_array).numpy()[0] 

    # 4. Result Interpretation
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence_max = round(100 * float(np.max(predictions)), 2)

    # 5. Final Return (SIMPLIFIED OUTPUT)
    
    # Retrieve only the 'Treatment' string from the disease_info dictionary
    treatment_info = disease_info[predicted_class]['Treatment']
    
    return {
        "class": predicted_class, # <-- Class
        "confidence": confidence_max, # <-- Confidence
        "treatment": treatment_info # <-- Treatment
    }