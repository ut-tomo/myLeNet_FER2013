# main.py
from utils.utils import load_images_from_directory
import os

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

train_dir = os.path.join(PARENT_DIR, "train")
test_dir = os.path.join(PARENT_DIR, "test")

X_train, y_train = load_images_from_directory(train_dir)
X_test, y_test = load_images_from_directory(test_dir)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
