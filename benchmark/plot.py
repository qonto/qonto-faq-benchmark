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
    colors = {
        'Qwen': '#605bec',
        'BM25': '#2ca02c',
        'Jina': '#258ec8',
        'Mistral': '#ff7f0e',
        'Cohere': '#39594d',
        'OpenAI': '#5ea3c4',
        'Google': '#ea4335',
        'Voyage': '#819640',
    }
    for key in colors:
        if key in model:
            color = colors[key]
            return find_similar_color_hex(color)
    # Random color.
    return '#' + ''.join(random.choices('0123456789abcdef', k=6))

def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple."""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert hex string to RGB tuple
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Convert an RGB tuple back to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def find_similar_color_hex(hex_color, max_variation=40):
    """
    Find a random color near the given hex color.

    Parameters:
    - hex_color: A string representing the color in hex format (e.g., "#RRGGBB").
    - max_variation: The maximum amount of variation allowed for each component.

    Returns:
    - A string representing the new color in hex format.
    """
    rgb_color = hex_to_rgb(hex_color)
    similar_rgb = find_similar_color(rgb_color, max_variation)
    return rgb_to_hex(similar_rgb)

def find_similar_color(rgb_color, max_variation=30):
    """
    Find a random color near the given RGB color.

    Parameters:
    - rgb_color: A tuple of (R, G, B) where each component is between 0 and 255.
    - max_variation: The maximum amount of variation allowed for each component.

    Returns:
    - A tuple representing the new RGB color.
    """
    R, G, B = rgb_color
    # Generate random variations for each color component
    new_R = max(0, min(255, R + random.randint(-max_variation, max_variation)))
    new_G = max(0, min(255, G + random.randint(-max_variation, max_variation)))
    new_B = max(0, min(255, B + random.randint(-max_variation, max_variation)))
    return (new_R, new_G, new_B)

if __name__ == '__main__':
    create_plot()
