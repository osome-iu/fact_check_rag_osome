"""
Purpose:
    Generate scatter plot comparing F1 macro scores between 6k and 12k datasets.

    This script creates a scatter plot showing the relationship between F1 macro-avg
    performance on 6k vs 12k claim datasets across all models and configurations.
    The plot includes a diagonal reference line and correlation analysis.

Input:
    - data/cleaned/classification_reports/multi_class_all_summary_0.25.parquet: 6k dataset results
    - data/cleaned/classification_reports/multi_class_all_summary_0.50.parquet: 12k dataset results

Output:
    - figures/6k_vs_12k_performance_comparison.pdf: Scatter plot comparing F1 scores

Author: Matthew DeVerna
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import spearmanr

# Set current directory to script location
os.chdir(Path(__file__).resolve().parent)

# Constants
DATA_DIR = Path("../../data/cleaned/classification_reports/").resolve()
FP_6K = DATA_DIR / "multi_class_all_summary_0.25.parquet"
FP_12K = DATA_DIR / "multi_class_all_summary_0.50.parquet"
OUTPUT_PATH = Path("../../figures/si_6k_vs_12k_performance_comparison.pdf").resolve()

# Plot styling constants
FIGURE_SIZE = (4, 4)
SCATTER_ALPHA = 0.7
SCATTER_SIZE = 30
SCATTER_COLOR = "#1f77b4"
DIAGONAL_COLOR = "k"
DIAGONAL_LINESTYLE = "--"
DIAGONAL_LINEWIDTH = 0.5
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.7
DPI = 600


def load_and_merge_data():
    """Load 6k and 12k performance data and merge on matching configurations."""
    # Load datasets
    df_6k = pd.read_parquet(FP_6K)
    df_12k = pd.read_parquet(FP_12K)

    # Merge on model configuration columns
    combined_df = pd.merge(
        df_6k[["model", "num_rag_items", "used_web_search", "f1-score-macro-avg"]],
        df_12k[["model", "num_rag_items", "used_web_search", "f1-score-macro-avg"]],
        on=["model", "num_rag_items", "used_web_search"],
        suffixes=("_6k", "_12k"),
    )

    return combined_df


def create_performance_comparison_plot():
    """Create scatter plot comparing F1 scores between 6k and 12k datasets."""
    # Load data
    combined_df = load_and_merge_data()

    # Calculate Spearman's correlation
    rho, p_value = spearmanr(
        combined_df["f1-score-macro-avg_6k"],
        combined_df["f1-score-macro-avg_12k"]
    )

    # Format p-value
    if p_value < 0.001:
        p_str = r"$p < 0.001$"
    elif p_value < 0.01:
        p_str = r"$p < 0.01$"
    elif p_value < 0.05:
        p_str = r"$p < 0.05$"
    else:
        p_str = fr"$p = {p_value:.2f}$"

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create scatter plot
    sns.scatterplot(
        data=combined_df,
        x="f1-score-macro-avg_6k",
        y="f1-score-macro-avg_12k",
        alpha=SCATTER_ALPHA,
        s=SCATTER_SIZE,
        color=SCATTER_COLOR,
        legend=False,
        ax=ax,
    )

    # Set axis labels and limits
    ax.set_xlabel("macro F1 (6k)")
    ax.set_ylabel("macro F1 (12k)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(fr"$\rho = {rho:.3f}$, {p_str}")

    # Add perfect diagonal line
    ax.plot(
        [0, 1],
        [0, 1],
        color=DIAGONAL_COLOR,
        linestyle=DIAGONAL_LINESTYLE,
        linewidth=DIAGONAL_LINEWIDTH,
        zorder=-1,
    )

    # Add grid
    ax.grid(True, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Define output paths
    pdf_path = OUTPUT_PATH
    png_path = Path(str(OUTPUT_PATH).replace(".pdf", ".png"))

    # Tight layout and save in both formats
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print("Figure saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"Spearman correlation: {rho:.6f}, p-value: {p_value:.6f}")


def main():
    """Main function to generate the 6k vs 12k performance comparison figure."""
    create_performance_comparison_plot()


if __name__ == "__main__":
    main()
