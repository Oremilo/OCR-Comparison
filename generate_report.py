import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import math
from fpdf import FPDF

# Configuration
RESULTS_DIR = "results"
PLOTS_LIGHT_DIR = os.path.join(RESULTS_DIR, "plots", "light")
PLOTS_DARK_DIR = os.path.join(RESULTS_DIR, "plots", "dark")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
REPORT_DIR = os.path.join(RESULTS_DIR, "report")
EXPORTS_DIR = os.path.join(RESULTS_DIR, "exports")

# Create directories
for d in [PLOTS_LIGHT_DIR, PLOTS_DARK_DIR, TABLES_DIR, REPORT_DIR, EXPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
REPORT_HTML = os.path.join(EXPORTS_DIR, "ocr_benchmark_report.html")

PALETTE = sns.color_palette("colorblind")

def save_plot(fig, filename, is_dark=False):
    out_dir = PLOTS_DARK_DIR if is_dark else PLOTS_LIGHT_DIR
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Save to buffer for base64 (always useful for HTML)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return filepath, img_str

def set_theme(is_dark=False):
    if is_dark:
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
        plt.rcParams.update({
            "axes.facecolor": "#1e1e1e",
            "figure.facecolor": "#1e1e1e",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white"
        })
    else:
        plt.style.use('default')
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def create_bar_chart(df, x_col, y_col, title, ylabel, filename, is_dark=False):
    set_theme(is_dark)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=df, x=x_col, y=y_col, hue=x_col, palette=PALETTE[:len(df)], ax=ax, legend=False)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    color='white' if is_dark else 'black')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("OCR Model", fontsize=12)
    if max(df[y_col]) <= 1.0 and "Time" not in title and "CER" not in title and "WER" not in title:
        ax.set_ylim(0, 1.1)
        
    return save_plot(fig, filename, is_dark)

def create_grouped_bar_chart(df, title, filename, is_dark=False):
    set_theme(is_dark)
    metrics_melted = df.melt(id_vars='model', value_vars=['precision', 'recall', 'f1'], 
                                     var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=metrics_melted, x='model', y='Score', hue='Metric', palette=PALETTE, ax=ax)
    
    for p in ax.patches:
        val = p.get_height()
        if math.isnan(val):
            continue
        ax.annotate(f"{val:.3f}", 
                    (p.get_x() + p.get_width() / 2., val), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=9,
                    color='white' if is_dark else 'black')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("OCR Model", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return save_plot(fig, filename, is_dark)

def create_radar_chart(df, filename, is_dark=False):
    set_theme(is_dark)
    categories = ['char_accuracy', 'word_accuracy', 'precision', 'recall', 'f1']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], ['Char Acc', 'Word Acc', 'Precision', 'Recall', 'F1-Score'], color='lightgrey' if is_dark else 'grey', size=11)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="lightgrey" if is_dark else "grey", size=8)
    plt.ylim(0, 1)
    
    for i, row in df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['model'], color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.title("Overall Performance Radar Chart", size=16, fontweight='bold', pad=20)
    
    return save_plot(fig, filename, is_dark)

def create_heatmap(df, filename, is_dark=False):
    set_theme(is_dark)
    metrics = ['char_accuracy', 'word_accuracy', 'CER', 'WER', 'precision', 'recall', 'f1']
    heatmap_data = df.set_index('model')[metrics]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu" if not is_dark else "viridis", linewidths=.5, ax=ax)
    plt.title("Model Performance Heatmap", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Model", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.xticks(rotation=45)
    
    return save_plot(fig, filename, is_dark)

def create_boxplot(results_df, metric, title, ylabel, filename, is_dark=False):
    set_theme(is_dark)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=results_df, x='model', y=metric, hue='model', palette=PALETTE[:len(results_df['model'].unique())], ax=ax, width=0.5, showfliers=False, legend=False)
    sns.violinplot(data=results_df, x='model', y=metric, hue='model', palette=PALETTE[:len(results_df['model'].unique())], ax=ax, inner=None, color=".8", alpha=0.3, legend=False)
    sns.stripplot(data=results_df, x='model', y=metric, hue='model', palette=PALETTE[:len(results_df['model'].unique())], size=2, linewidth=0, ax=ax, alpha=0.5, legend=False)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("OCR Model", fontsize=12)
    
    return save_plot(fig, filename, is_dark)

def generate_insights(summary_df):
    insights = []
    
    best_acc = summary_df.loc[summary_df['char_accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['f1'].idxmax()]
    
    insights.append(f"Model '{best_acc['model']}' achieved the highest Character Accuracy at {best_acc['char_accuracy']:.2%} and Word Accuracy at {best_acc['word_accuracy']:.2%}.")
    insights.append(f"Model '{best_f1['model']}' produced the highest F1-score ({best_f1['f1']:.3f}), indicating the best balance between precision and recall.")
    
    failed_models = summary_df[summary_df['char_accuracy'] < 0.05]['model'].tolist()
    if failed_models:
        insights.append(f"Models {', '.join(failed_models)} achieved exceptionally low scores (<5%), suggesting missing underlying dependencies or models failing to initialize properly in this environment.")
        
    return insights

def export_tables(summary_df):
    # Save as CSV
    summary_df.to_csv(os.path.join(TABLES_DIR, "benchmark_metrics.csv"), index=False)
    
    # Save as LaTeX
    latex_str = summary_df.style.format(precision=3).to_latex()
    with open(os.path.join(TABLES_DIR, "benchmark_metrics.tex"), "w") as f:
        f.write(latex_str)

def export_markdown_latex_report(insights):
    md_content = "# Results and Discussion\n\n"
    md_content += "Based on the comprehensive OCR benchmarking over the SROIE 2019 dataset, the following key insights were observed:\n\n"
    for ins in insights:
        md_content += f"- {ins}\n"
        
    md_content += "\n## Analysis\nThe Accuracy metrics (Character and Word Accuracy) measure exact match rates, while Error Rates (CER and WER) quantify the edit distance needed to correct the predicted text. Precision, Recall, and F1-score are calculated using a Bag-of-Words approach to measure the engines' ability to retrieve the correct vocabulary regardless of ordering. The visualizations and tables clearly demonstrate the comparative performance."
    
    with open(os.path.join(REPORT_DIR, "results_and_discussion.md"), "w") as f:
        f.write(md_content)
        
    tex_content = "\\section*{Results and Discussion}\n\n"
    tex_content += "Based on the comprehensive OCR benchmarking over the SROIE 2019 dataset, the following key insights were observed:\n\n\\begin{itemize}\n"
    for ins in insights:
        # Escape percentages for LaTeX
        ins_tex = ins.replace("%", "\\%")
        tex_content += f"    \\item {ins_tex}\n"
    tex_content += "\\end{itemize}\n\n"
    tex_content += "The Accuracy metrics (Character and Word Accuracy) measure exact match rates, while Error Rates (CER and WER) quantify the edit distance needed to correct the predicted text. Precision, Recall, and F1-score are calculated using a Bag-of-Words approach to measure the engines' ability to retrieve the correct vocabulary regardless of ordering. The visualizations and tables clearly demonstrate the comparative performance.\n"
    
    with open(os.path.join(REPORT_DIR, "results_and_discussion.tex"), "w") as f:
        f.write(tex_content)
        
    return md_content

def generate_pdf_report(md_content, light_plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(200, 10, txt="OCR Benchmark Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt="Executive Summary & Key Insights", ln=True)
    pdf.set_font("Helvetica", size=12)
    
    # Simple markdown to text conversion
    lines = md_content.split('\n')
    for line in lines:
        if line.startswith('#'): continue
        if line.startswith('-'):
            pdf.multi_cell(190, 8, txt=f"  - {line[2:]}")
        elif line.strip() != "":
            pdf.multi_cell(190, 8, txt=line)
            pdf.ln(2)
            
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt="Performance Visualizations", ln=True)
    pdf.ln(5)
    
    # Insert images
    img_width = 160
    # Center X
    x = (210 - img_width) / 2
    
    pdf.image(light_plots['acc'][0], x=x, w=img_width)
    pdf.ln(5)
    pdf.image(light_plots['radar'][0], x=x+20, w=120)
    
    pdf.add_page()
    pdf.image(light_plots['prf1'][0], x=x, w=img_width)
    pdf.ln(5)
    pdf.image(light_plots['cer'][0], x=x, w=img_width)
    
    out_pdf = os.path.join(EXPORTS_DIR, "benchmark_report.pdf")
    pdf.output(out_pdf)

def main():
    print("Loading data...")
    if not os.path.exists(SUMMARY_CSV) or not os.path.exists(RESULTS_CSV):
        print("Error: Benchmark result CSVs not found in 'results/' directory.")
        return
        
    summary_df = pd.read_csv(SUMMARY_CSV)
    results_df = pd.read_csv(RESULTS_CSV)
    summary_df = summary_df.fillna(0)
    
    print("Exporting publication-ready tables...")
    export_tables(summary_df)
    
    print("Generating visualizations (Light & Dark themes)...")
    light_plots = {}
    dark_plots = {}
    
    for is_dark in [False, True]:
        plots_dict = dark_plots if is_dark else light_plots
        plots_dict['acc'] = create_bar_chart(summary_df, 'model', 'char_accuracy', 'Character Accuracy Comparison', 'Accuracy', 'accuracy_comparison.png', is_dark)
        plots_dict['cer'] = create_bar_chart(summary_df, 'model', 'CER', 'Character Error Rate (CER) Comparison', 'CER (Lower is better)', 'cer_comparison.png', is_dark)
        plots_dict['wer'] = create_bar_chart(summary_df, 'model', 'WER', 'Word Error Rate (WER) Comparison', 'WER (Lower is better)', 'wer_comparison.png', is_dark)
        plots_dict['time'] = create_bar_chart(summary_df, 'model', 'avg_inference_time', 'Average Inference Time Comparison', 'Time (Seconds)', 'inference_time_comparison.png', is_dark)
        plots_dict['prf1'] = create_grouped_bar_chart(summary_df, 'Precision, Recall, and F1-Score Comparison', 'prf1_comparison.png', is_dark)
        plots_dict['radar'] = create_radar_chart(summary_df, 'radar_comparison.png', is_dark)
        plots_dict['heatmap'] = create_heatmap(summary_df, 'heatmap_comparison.png', is_dark)
        plots_dict['box_cer'] = create_boxplot(results_df, 'CER', 'CER Distribution per Image', 'CER', 'box_cer.png', is_dark)
        plots_dict['box_time'] = create_boxplot(results_df, 'inference_time', 'Inference Time Distribution', 'Time (Seconds)', 'box_time.png', is_dark)
    
    print("Generating insights & reports...")
    insights = generate_insights(summary_df)
    md_content = export_markdown_latex_report(insights)
    
    print("Compiling PDF report...")
    generate_pdf_report(md_content, light_plots)
    
    print("Generating HTML dashboard...")
    table_html = summary_df.to_html(classes='table table-striped table-hover table-bordered', float_format='%.4f', index=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Benchmark Analysis Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }}
            .container {{ max-width: 1200px; margin-top: 40px; background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }}
            h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; }}
            .plot-container {{ text-align: center; margin-bottom: 40px; padding: 20px; border: 1px solid #e1e8ed; border-radius: 8px; background-color: #fff; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
            .insights-card {{ background-color: #e8f4f8; border-left: 5px solid #3498db; padding: 20px; margin-bottom: 30px; border-radius: 4px; }}
            .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OCR Benchmark Analysis Report - SROIE 2019</h1>
            <div class="insights-card">
                <h3>Executive Summary & Key Insights</h3>
                <ul>{"".join(f"<li>{ins}</li>" for ins in insights)}</ul>
            </div>
            <h2>1. Summary Metrics Table</h2>
            <div class="table-responsive">{table_html}</div>
            <h2>2. Visualizations</h2>
            <div class="row">
                <div class="col-md-6"><div class="plot-container"><h4>Character Accuracy</h4><img src="data:image/png;base64,{light_plots['acc'][1]}"></div></div>
                <div class="col-md-6"><div class="plot-container"><h4>Overall Performance Radar</h4><img src="data:image/png;base64,{light_plots['radar'][1]}"></div></div>
            </div>
            <div class="row">
                <div class="col-md-12"><div class="plot-container"><h4>Performance Heatmap</h4><img src="data:image/png;base64,{light_plots['heatmap'][1]}"></div></div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(REPORT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"All exports completed successfully in {EXPORTS_DIR}")

if __name__ == "__main__":
    main()
