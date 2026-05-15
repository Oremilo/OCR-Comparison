import jiwer
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(gt_text, pred_text):
    """
    Calculate OCR metrics between ground truth and prediction.
    """
    if not gt_text and not pred_text:
        return 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
        
    if not gt_text:
        return 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
        
    if not pred_text:
        return 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
        
    # CER / WER
    try:
        cer = jiwer.cer(gt_text, pred_text)
        wer = jiwer.wer(gt_text, pred_text)
    except:
        cer = 1.0
        wer = 1.0
        
    # Character-level accuracy and Word-level accuracy
    char_acc = max(0.0, 1.0 - cer)
    word_acc = max(0.0, 1.0 - wer)
    
    # Tokenize for precision, recall, F1 (macro approach on bag of words)
    gt_words = set(gt_text.split())
    pred_words = set(pred_text.split())
    
    all_words = list(gt_words.union(pred_words))
    if not all_words:
        precision, recall, f1 = 1.0, 1.0, 1.0
    else:
        y_true = [1 if w in gt_words else 0 for w in all_words]
        y_pred = [1 if w in pred_words else 0 for w in all_words]
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
    return char_acc, word_acc, cer, wer, precision, recall, f1
