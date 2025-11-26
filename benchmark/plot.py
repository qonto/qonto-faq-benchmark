import os
import json
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def create_plot():
    """
    Reads all benchmark JSON files, plots the data, and saves it as an SVG image.
    """
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    all_data = []

    # Find all .json files in the benchmark directory
    for filename in os.listdir(benchmark_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(benchmark_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                model_name = data.get('model')
                for measurement in data.get('measurements', []):
                    all_data.append({
                        'model': model_name,
                        'top_k': measurement.get('top_k'),
                        'information_assimilation': 100 * measurement.get('information_assimilation')
                    })

    if not all_data:
        print("No benchmark data found. Exiting.")
        return

    df = pd.DataFrame(all_data)
    df = df[df.model != 'question_only']

    # Order models by their information assimilation at top_k=1
    df_top_k_1 = df[df['top_k'] == 1].sort_values(by='information_assimilation')
    ordered_models = df_top_k_1['model'].tolist()

    # Create a palette dictionary, ensuring order
    palette = {model: color_for_model(model) for i, model in enumerate(ordered_models)}

    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create the line plot
    sns.lineplot(
        data=df,
        x='top_k',
        y='information_assimilation',
        hue='model',
        hue_order=ordered_models, # Apply the custom order
        palette=palette,
        marker='o',
        ax=ax
    )

    # Customize the plot
    ax.set_title('Model Performance: Information Assimilation vs. Top K Documents', fontsize=16)
    ax.set_xlabel('Top K Documents', fontsize=12)
    ax.set_ylabel('Information Assimilation', fontsize=12)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(benchmark_dir, 'plot.svg')
    fig.savefig(output_path, format='svg')
    print(f"Plot saved to {output_path}")

def color_for_model(model):
    """
    Returns a unique color for each model.
    """
    palette = {
        'Qwen': '#6F4AFF',
        'BM25': '#1F77B4',
        'Jina': '#17BECF',
        'Mistral': '#FF7F0E',
        'Cohere': '#9467BD',
        'OpenAI': '#2CA02C',
        'gemini': '#D62728',
        'Voyage': '#8C564B',
        'Gemma': '#E377C2',
    }
    for key, color in palette.items():
        if key in model:
            return color
    # Deterministic fallback based on model name for unseen labels.
    rand = random.Random(model)
    return '#' + ''.join(rand.choices('0123456789abcdef', k=6))

if __name__ == '__main__':
    create_plot()
