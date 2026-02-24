import numpy as np
from PIL import Image

def preprocess_image(image, size):
    """
    Preprocess MRI image for deep learning models
    - Resize image
    - Normalize pixel values
    - Expand dimensions for model input
    """

    # Resize image
    image = image.resize(size)

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize (0 to 1)
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
import numpy as np
from PIL import Image

def preprocess_grayscale(image, size):
    """
    Preprocess image for U-Net (grayscale, normalized)
    """
    image = image.convert("L")          # Convert to grayscale
    image = image.resize(size)

    image = np.array(image) / 255.0     # Normalize
    image = np.expand_dims(image, axis=-1)  # (H, W, 1)
    image = np.expand_dims(image, axis=0)   # (1, H, W, 1)

    return image
