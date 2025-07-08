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


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col