import numpy as np
from PIL import Image

def overlay_mask(image, mask, alpha=0.4):
    image_np = np.array(image)

    # Ensure RGB
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)

    # Prepare mask
    mask = np.squeeze(mask)
    mask = (mask > 0).astype(np.uint8)

    # Resize mask to image size
    mask_pil = Image.fromarray(mask * 255)
    mask_pil = mask_pil.resize(
        (image_np.shape[1], image_np.shape[0]),
        resample=Image.NEAREST
    )
    resized_mask = np.array(mask_pil) > 0

    # ✅ ONLY tumor area green
    output = image_np.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)

    output[resized_mask] = (
        (1 - alpha) * image_np[resized_mask] +
        alpha * green
    ).astype(np.uint8)

    return output, resized_mask
