import pytesseract
from PIL import Image

# Update this if your path differs
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\OWNER\\sAppData\\Local\Programs\\Tesseract-OCR"
print(pytesseract.get_tesseract_version())
