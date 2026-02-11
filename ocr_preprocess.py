from PIL import Image, ImageFilter
import numpy as np

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    # Convert to grayscale
    gray = image.convert("L")

    # Increase contrast using simple threshold
    gray_np = np.array(gray)
    threshold = gray_np.mean()
    binary = (gray_np > threshold) * 255
    binary = binary.astype("uint8")

    processed = Image.fromarray(binary)

    # Optional: slight sharpening
    processed = processed.filter(ImageFilter.SHARPEN)

    return processed
