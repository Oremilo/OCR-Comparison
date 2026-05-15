import keras_ocr
import time

class KerasOCRModel:
    def __init__(self):
        # Initialize pipeline once to avoid repeated model loading
        self.pipeline = keras_ocr.pipeline.Pipeline()
        
    def predict(self, image_path):
        """
        Runs Keras-OCR on an image and returns extracted text and inference time.
        """
        start_time = time.perf_counter()
        text = ""
        try:
            # keras-ocr expects a list of images, tools.read reads the image correctly
            images = [keras_ocr.tools.read(image_path)]
            predictions = self.pipeline.recognize(images)[0]
            # predictions is a list of (text, box) tuples
            text = " ".join([word_text for word_text, box in predictions])
        except Exception as e:
            print(f"Keras-OCR Error on {image_path}: {e}")
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        return text, inference_time
