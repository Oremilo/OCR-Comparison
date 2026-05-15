import os
import pandas as pd
from tqdm import tqdm

from utils.parser import get_dataset_files
from utils.preprocessing import normalize_text
from utils.metrics import calculate_metrics
from utils.visualization import generate_plots

from models.easyocr_model import EasyOCRModel
from models.tesseract_model import TesseractModel
# from models.kerasocr_model import KerasOCRModel
from models.paddleocr_model import PaddleOCRModel

def main():
    # Dataset path relative to this script if run in this directory, or absolute as requested
    dataset_path = r"c:\Users\himan\Downloads\archive (1)\SROIE2019"
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    plots_dir = os.path.join(results_dir, "plots")
    
    # Ensure result directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading dataset...")
    # For benchmarking, we use the test set
    test_data = get_dataset_files(dataset_path)
    if not test_data:
        print("No data found. Please check the dataset path.")
        return
        
    print(f"Found {len(test_data)} images in test set.")
    
    # Initialize models
    print("Initializing models...")
    models = {
        'Tesseract': TesseractModel(),
        'EasyOCR': EasyOCRModel(),
        # 'KerasOCR': KerasOCRModel(),
        'PaddleOCR': PaddleOCRModel()
    }
    
    all_results = []
    
    # Benchmarking Loop
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}...")
        
        # We iterate over the test data and calculate metrics for each image
        for data in tqdm(test_data, desc=model_name):
            image_name = data['image_name']
            image_path = data['image_path']
            gt_text_raw = data['gt_text']
            
            # Get Prediction
            pred_text_raw, inf_time = model.predict(image_path)
            
            # Normalize texts
            gt_text = normalize_text(gt_text_raw)
            pred_text = normalize_text(pred_text_raw)
            
            # Calculate Evaluation Metrics
            char_acc, word_acc, cer, wer, precision, recall, f1 = calculate_metrics(gt_text, pred_text)
            
            # Store Results
            all_results.append({
                'image_name': image_name,
                'model': model_name,
                'ground_truth': gt_text,
                'predicted_text': pred_text,
                'char_accuracy': char_acc,
                'word_accuracy': word_acc,
                'CER': cer,
                'WER': wer,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'inference_time': inf_time
            })
            
    # Save Detailed Results
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(results_dir, "benchmark_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed results saved to {results_csv_path}")
    
    # Generate and Save Summary
    summary_df = results_df.groupby('model').agg({
        'char_accuracy': 'mean',
        'word_accuracy': 'mean',
        'CER': 'mean',
        'WER': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'inference_time': 'mean'
    }).reset_index()
    
    summary_df.rename(columns={'inference_time': 'avg_inference_time'}, inplace=True)
    summary_csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to {summary_csv_path}")
    
    # Generate Visualization Plots
    print("Generating visual plots...")
    generate_plots(summary_df, plots_dir)
    print(f"Plots saved to {plots_dir}")
    print("Benchmarking pipeline completed successfully!")

if __name__ == "__main__":
    main()
