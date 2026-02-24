import tensorflow as tf
import numpy as np

def load_model(model_path):
    """
    Load a trained deep learning model
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def cnn_predict(model, image_array):
    """
    Predict tumor presence using CNN model
    Returns probability value
    """
    prediction = model.predict(image_array)
    return float(prediction[0][0])


def unet_predict(model, image_array):
    """
    Predict tumor region mask using U-Net / Attention U-Net
    Returns segmentation mask
    """
    mask = model.predict(image_array)
    return mask[0]


def srgan_enhance(model, image_array):
    """
    Enhance MRI image using SRGAN
    SRGAN expects 64x64 input
    """
    enhanced_image = model.predict(image_array)
    return enhanced_image[0]
