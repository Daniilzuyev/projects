# work with graphs (heatmap, histplot, boxplot, scatter)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(filename: str, folder="../output/graphs"):
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{filename}"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved: {path}")

def set_style():
    sns.set(style='whitegrid', palette='colorblind')
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
