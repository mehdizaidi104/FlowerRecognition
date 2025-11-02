import os
import shutil
import psycopg2
import time

# --- DB Config (must match docker-compose) ---
DB_NAME = "flower_db"
DB_USER = "myuser"
DB_PASS = "mypassword"
DB_HOST = "db" # This is the service name from docker-compose
DB_PORT = "5432"

# --- Paths (must match docker-compose volumes) ---
SOURCE_DATA_FOLDER = "/app/Flower-Dataset"
TARGET_IMAGE_VOLUME = "/app/images"

def wait_for_db():
    """Waits for the database to be ready."""
    print("Loader: Waiting for database...")
    while True:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
            )
            conn.close()
            print("Loader: Database is ready!")
            break
        except psycopg2.OperationalError:
            print("Loader: DB not ready, retrying in 1s...")
            time.sleep(1)

def setup_database():
    """Creates the 'images' table in the database."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()
    
    cur.execute("DROP TABLE IF EXISTS images;")
    cur.execute("""
        CREATE TABLE images (
            id SERIAL PRIMARY KEY,
            label VARCHAR(100) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            dataset_type VARCHAR(10) NOT NULL -- e.g., 'train' or 'test'
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Loader: Table 'images' created.")
    return True

def process_images():
    """Copies images to the volume and inserts metadata into the DB."""

    # --- ADD THIS DEBUG LINE ---
    print(f"Loader: DEBUG: Contents of {SOURCE_DATA_FOLDER}: {os.listdir(SOURCE_DATA_FOLDER)}")
    # --- END DEBUG LINE ---

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    if not os.path.exists(TARGET_IMAGE_VOLUME):
        os.makedirs(TARGET_IMAGE_VOLUME)

    print("Loader: Populating database and image volume...")
    labels = [d for d in os.listdir(SOURCE_DATA_FOLDER) if os.path.isdir(os.path.join(SOURCE_DATA_FOLDER, d))]
    
    for label in labels:
        print(f"Loader: Processing label: {label}")
        source_dir = os.path.join(SOURCE_DATA_FOLDER, label)
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_file = os.path.join(source_dir, filename)
                
                # Create a unique filename to avoid clashes
                unique_filename = f"{label.lower()}_{filename}"
                target_file_path = os.path.join(TARGET_IMAGE_VOLUME, unique_filename)
                
                # Copy file to shared volume
                shutil.copy(source_file, target_file_path)
                
                # Insert metadata into DB
                # We'll tag 80% as train, 20% as test (basic split)
                dataset_type = 'train' if hash(filename) % 5 > 0 else 'test'
                db_path = os.path.join("/app/images", unique_filename) # Path *inside* the container
                
                cur.execute(
                    "INSERT INTO images (label, file_path, dataset_type) VALUES (%s, %s, %s)",
                    (label, db_path, dataset_type)
                )

    conn.commit()
    cur.close()
    conn.close()
    print("Loader: Data loading complete.")

if __name__ == "__main__":
    wait_for_db()
    if setup_database():
        process_images()