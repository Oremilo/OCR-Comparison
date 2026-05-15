import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy Comparison Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='model', y='char_accuracy')
    plt.title('Character Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()

    # 2. CER Comparison Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='model', y='CER')
    plt.title('Character Error Rate (CER) Comparison')
    plt.ylabel('CER')
    plt.savefig(os.path.join(output_dir, 'cer_comparison.png'))
    plt.close()

    # 3. WER Comparison Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='model', y='WER')
    plt.title('Word Error Rate (WER) Comparison')
    plt.ylabel('WER')
    plt.savefig(os.path.join(output_dir, 'wer_comparison.png'))
    plt.close()

    # 4. Inference Time Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='model', y='avg_inference_time')
    plt.title('Average Inference Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'))
    plt.close()

    # 5. Precision/Recall/F1 Grouped Chart
    metrics_melted = summary_df.melt(id_vars='model', value_vars=['precision', 'recall', 'f1'], 
                                     var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_melted, x='model', y='Score', hue='Metric')
    plt.title('Precision, Recall, and F1-Score Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, 'prf1_comparison.png'))
    plt.close()
