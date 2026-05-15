from paddleocr import PaddleOCR
import time
import logging

class PaddleOCRModel:
    def __init__(self):
        # use angle classification, English model
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        
    def predict(self, image_path):
        """
        Runs PaddleOCR on an image and returns extracted text and inference time.
        """
        start_time = time.perf_counter()
        text = ""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    # line format: [[box_coords], (text, confidence)]
                    text_lines.append(line[1][0])
            text = " ".join(text_lines)
        except Exception as e:
            print(f"PaddleOCR Error on {image_path}: {e}")
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        return text, inference_time
