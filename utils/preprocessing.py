import re

def normalize_text(text):
    """
    Normalizes text for OCR evaluation.
    - lowercase text
    - remove extra spaces
    - strip line breaks consistently
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove duplicate whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
