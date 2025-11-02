import os
import psycopg2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import time

# --- Configs ---
DB_NAME = "flower_db"
DB_USER = "myuser"
DB_PASS = "mypassword"
DB_HOST = "db" # The service name!
DB_PORT = "5432"
IMAGE_SIZE = (256, 256)
MODEL_SAVE_PATH = "/app/models/flower_model.keras" # Path from docker-compose volume

def wait_for_db():
    """Waits for the database to be ready."""
    print("Trainer: Waiting for database...")
    while True:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
            )
            conn.close()
            print("Trainer: Database is ready!")
            break
        except psycopg2.OperationalError:
            print("Trainer: DB not ready, retrying in 1s...")
            time.sleep(1)

def load_data_from_db():
    """Fetches image paths and labels from the database."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    cur.execute("SELECT file_path, label FROM images WHERE dataset_type = 'train'")
    train_data = cur.fetchall()
    
    cur.execute("SELECT file_path, label FROM images WHERE dataset_type = 'test'")
    test_data = cur.fetchall()
    
    cur.close()
    conn.close()
    
    # Process data
    train_paths, train_labels = zip(*train_data)
    test_paths, test_labels = zip(*test_data)
    
    # Encode labels
    encoder = LabelEncoder()
    all_labels = sorted(list(set(train_labels + test_labels)))
    encoder.fit(all_labels)
    
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    
    class_names = list(encoder.classes_)
    print(f"Trainer: Classes found: {class_names}")
    
    # This is CORRECT
    num_classes = len(class_names)
    return train_paths, y_train, test_paths, y_test, num_classes

def preprocess_image(image_path):
    """Loads and preprocesses a single image."""
    try:
        img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32) # Return empty image on error

def data_generator(paths, labels, batch_size=8):
    """A generator to feed data to Keras."""
    idx = 0
    while True:
        batch_paths = paths[idx : idx + batch_size]
        batch_labels = labels[idx : idx + batch_size]
        
        batch_images = np.array([preprocess_image(p) for p in batch_paths])
        
        yield batch_images, batch_labels
        
        idx += batch_size
        if idx >= len(paths):
            idx = 0 # Loop back to start

def build_model(num_classes):
    """Builds the Keras model (from your notebook)."""
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256,256,3)),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    learning_rate = 0.00001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    wait_for_db()
    
    print("Trainer: Loading data from database...")
    train_paths, y_train, test_paths, y_test, num_classes = load_data_from_db()
    
    print(f"Trainer: Building model for {num_classes} classes...")
    model = build_model(num_classes)
    model.summary()
    
    # Create generators
    batch_size = 8
    train_gen = data_generator(train_paths, y_train, batch_size)
    test_gen = data_generator(test_paths, y_test, batch_size)
    
    print("Trainer: Starting model training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(train_paths) // batch_size),
        epochs=20, # From your notebook
        validation_data=test_gen,
        validation_steps=max(1, len(test_paths) // batch_size)
    )
    
    # Ensure model save directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    print(f"Trainer: Training complete. Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)