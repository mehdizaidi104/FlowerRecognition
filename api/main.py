import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os

# --- CONFIG ----------------
IMAGE_SIZE = (256, 256)
MODEL_PATH = "/app/models/flower_model.keras" # Path from docker-compose volume
# These MUST match the order from your trainer's LabelEncoder
CLASS_NAMES = ["Daffodil", "Rose", "Snowdrop", "Sunflower", "WoodAnemone"]

# ---------------- MODEL LOADING ----------------
model = None

app = FastAPI(title="Flower Recognition API")

# --- ADD CORS MIDDLEWARE ---
# This allows your frontend (on http://localhost) to talk to
# this API (on http://localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

@app.on_event("startup")
def load_trained_model():
    """Load the model on startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}.")
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")

# ---------------- HELPERS ----------------
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- API APP ----------------
@app.get("/")
def read_root():
    if model is None:
        return {"message": "API is running, but the model is not loaded."}
    return {"message": "Flower Recognition API is running and model is loaded."}

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    
    preds = model.predict(input_data)
    probs = tf.nn.softmax(preds[0]).numpy()
    
    top_idx = int(np.argmax(probs))
    flower_name = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])
    
    return {
        "prediction": flower_name,
        "confidence": confidence,
        "class_probabilities": dict(zip(CLASS_NAMES, probs.astype(float)))
    }