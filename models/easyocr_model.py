import easyocr
import time

class EasyOCRModel:
    def __init__(self):
        # Initialize with English, automatically uses GPU if available
        self.reader = easyocr.Reader(['en'], gpu=True)
        
    def predict(self, image_path):
        """
        Runs EasyOCR on an image and returns extracted text and inference time.
        """
        start_time = time.perf_counter()
        text = ""
        try:
            result = self.reader.readtext(image_path, detail=0, paragraph=False)
            text = " ".join(result)
        except Exception as e:
            print(f"EasyOCR Error on {image_path}: {e}")
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        return text, inference_time
