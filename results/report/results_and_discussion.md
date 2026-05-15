# Results and Discussion

Based on the comprehensive OCR benchmarking over the SROIE 2019 dataset, the following key insights were observed:

- Model 'EasyOCR' achieved the highest Character Accuracy at 78.22% and Word Accuracy at 44.45%.
- Model 'EasyOCR' produced the highest F1-score (0.310), indicating the best balance between precision and recall.
- Models PaddleOCR, Tesseract achieved exceptionally low scores (<5%), suggesting missing underlying dependencies or models failing to initialize properly in this environment.

## Analysis
The Accuracy metrics (Character and Word Accuracy) measure exact match rates, while Error Rates (CER and WER) quantify the edit distance needed to correct the predicted text. Precision, Recall, and F1-score are calculated using a Bag-of-Words approach to measure the engines' ability to retrieve the correct vocabulary regardless of ordering. The visualizations and tables clearly demonstrate the comparative performance.