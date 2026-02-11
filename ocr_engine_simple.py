import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\tanishkas\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

def extract_text(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text
