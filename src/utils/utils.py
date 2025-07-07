import os
import numpy as np
from PIL import Image

EMOTION_LABELS = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def load_images_from_directory(base_dir, image_size=(48, 48)):
    X = []
    y = []

    for label_name, label_id in EMOTION_LABELS.items():
        folder = os.path.join(base_dir, label_name)
        if not os.path.exists(folder):
            print(f"No folder: {folder}")
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                img = Image.open(file_path).convert("L")
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                X.append(img_array)
                y.append(label_id)
            except Exception as e:
                print(f"Error: {e}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    return X, y
