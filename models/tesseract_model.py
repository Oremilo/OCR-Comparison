import pytesseract
import time
from PIL import Image

class TesseractModel:
    def __init__(self):
        # Configure optimal PSM/OEM settings
        # OEM 3: Default, based on what is available.
        # PSM 3: Fully automatic page segmentation, but no OSD.
        self.custom_config = r'--oem 3 --psm 3'
        
    def predict(self, image_path):
        """
        Runs Tesseract OCR on an image and returns extracted text and inference time.
        """
        start_time = time.perf_counter()
        text = ""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=self.custom_config)
        except Exception as e:
            print(f"Tesseract Error on {image_path}: {e}")
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        return text, inference_time
