import tensorflow as tf
import numpy as np
from PIL import Image

class ImageUploader:
    def __init__(self):
        pass

    def process_image(self, image_path):
        # Load image
        img = Image.open(image_path).convert('L')
        # Resize to 28x28 pixels
        img = img.resize((28, 28))
        # Convert to numpy array
        img_array = np.array(img)
        # Invert colors if background is black
        if np.mean(img_array) < 128:
            img_array = 255 - img_array
        # Normalize pixel values
        img_array = img_array / 255.0
        # Expand dimensions to match model input
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        return img_array.astype('float32')