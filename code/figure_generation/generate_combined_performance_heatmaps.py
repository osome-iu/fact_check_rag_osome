"""
Purpose:
    Generate a combined figure showing F1-macro performance heatmaps for fact-checking models
    across years (2007-2024). The figure includes two vertically stacked heatmaps:
    1. Zero-shot performance (top)
    2. RAG k=6 and web search enabled performance (bottom)

    Models are ordered from best to worst overall performance on y-axis.
    Years 2007-2024 are shown on x-axis.
    Orange boxes highlight the maximum value for each year.

Input:
    - multi_class_all_summary_0.25_by_year.parquet: Performance metrics by year for all models

Output:
    - combined_performance_heatmaps.pdf: Combined heatmap figure saved to figures/ directory

Author: Matthew DeVerna
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set working directory to script location
script_dir = Path(__file__).parent
data_dir = script_dir / "../../data/cleaned/classification_reports"
figures_dir = script_dir / "../../figures"

# Constants
OUTPUT_FILENAME = "si_combined_performance_heatmaps"

# Model name mapping for cleaner display
MODELS_NAME_MAP = {
    "DeepSeek-V3": "DeepSeek-V3",
    "Llama-3.2-3B-Instruct-Turbo": "Llama 3.2 3B",
    "Llama-3.2-11B-Vision-Instruct-Turbo": "Llama 3.2 11B",
    "Llama-3.2-90B-Vision-Instruct-Turbo": "Llama 3.2 90B",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0 Pro",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-4o-2024-11-20": "GPT-4o",
    "DeepSeek-R1": "DeepSeek-R1",
    "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
    "gpt-4o-search-preview-2025-03-11": "GPT-4o Search",
    "gpt-4o-mini-search-preview-2025-03-11": "GPT-4o mini Search",
    "o1-2024-12-17": "o1",
    "o3-mini-2025-01-31": "o3-mini",
}


def load_data():
    """Load the year-based performance data."""
    data_path = data_dir / "multi_class_all_summary_0.25_by_year.parquet"
    multi_year_025 = pd.read_parquet(data_path)

    print(f"Data loaded: {multi_year_025.shape}")
    print(f"Years available: {sorted(multi_year_025.factcheck_year.unique())}")
    print(f"Models available: {len(multi_year_025.model.unique())}")
    print(f"Models available: {multi_year_025.model.unique()}")

    return multi_year_025


def prepare_data(multi_year_025):
    """Filter and prepare data for both heatmaps."""
    # Filter for only the models in the mapping
    selected_models = list(MODELS_NAME_MAP.keys())

    # Filter for RAG k=6 configuration
    rag_6_data = multi_year_025[
        (multi_year_025["num_rag_items"] == 6.0)
        & (multi_year_025["model"].isin(selected_models))
    ].copy()

    # Filter for zero-shot configuration
    zero_shot_data = multi_year_025[
        (multi_year_025["num_rag_items"] == 0.0)
        & (multi_year_025["model"].isin(selected_models))
    ].copy()

    # Remove rows for gemini models that do not use web search
    # Exception: gemini-2.0-flash-lite and gemini-2.0-pro-exp-02-05 could not be tested with search
    exceptions = ["gemini-2.0-flash-lite", "gemini-2.0-pro-exp-02-05"]
    keep_rows = (
        (~rag_6_data["model"].str.contains("gemini"))  # Keep all non-gemini models
        | (rag_6_data["used_web_search"] == True)  # Keep gemini models with web search
        | (rag_6_data["model"].isin(exceptions))  # Keep the two exception models
    )
    rag_6_data = rag_6_data[keep_rows]

    keep_rows_zero = (
        (~zero_shot_data["model"].str.contains("gemini"))
        | (zero_shot_data["used_web_search"] == True)
        | (zero_shot_data["model"].isin(exceptions))
    )
    zero_shot_data = zero_shot_data[keep_rows_zero]

    print(f"RAG k=6 data after filtering: {rag_6_data.shape}")
    print(f"Zero-shot data after filtering: {zero_shot_data.shape}")

    return rag_6_data, zero_shot_data


def create_heatmap_data(data, config_name, add_search_suffix=False):
    """Create pivot table for heatmap and sort models by performance."""
    # Create pivot table
    heatmap_data = data.pivot_table(
        values="f1-score-macro-avg",
        index="model",
        columns="factcheck_year",
        aggfunc="first",
    )

    # Map model names to display names
    if add_search_suffix:
        # For RAG heatmap, check web search usage and modify names accordingly
        display_names = {}
        for orig_model in heatmap_data.index:
            base_name = MODELS_NAME_MAP.get(orig_model, orig_model)

            # Check if this model uses web search in the data
            model_data = data[data["model"] == orig_model]
            uses_web_search = (
                model_data["used_web_search"].iloc[0] if not model_data.empty else False
            )

            # Add " + Search" if uses web search and doesn't already contain "search"
            if uses_web_search and "search" not in base_name.lower():
                display_names[orig_model] = base_name + " + Search"
            else:
                display_names[orig_model] = base_name

        heatmap_data.index = heatmap_data.index.map(display_names)
    else:
        heatmap_data.index = heatmap_data.index.map(MODELS_NAME_MAP)

    # Calculate overall performance for each model to sort y-axis
    model_avg_performance = heatmap_data.mean(axis=1, skipna=True).sort_values(
        ascending=False
    )

    # Reorder heatmap data by model performance (best to worst)
    heatmap_data_sorted = heatmap_data.reindex(model_avg_performance.index)

    print(f"\n{config_name} - Top 5 models by average F1-macro:")
    for i, (model, score) in enumerate(model_avg_performance.head().items(), 1):
        print(f"{i}. {model}: {score:.3f}")

    return heatmap_data_sorted


def add_max_value_boxes(ax, heatmap_data):
    """Add orange boxes around the maximum value for each year."""
    for col_idx, year in enumerate(heatmap_data.columns):
        # Find the row index with maximum value for this year
        col_data = heatmap_data[year].dropna()
        if len(col_data) > 0:
            max_row_name = col_data.idxmax()
            max_row_idx = heatmap_data.index.get_loc(max_row_name)

            # Add orange rectangle around the cell
            rect = plt.Rectangle(
                (col_idx, max_row_idx),
                1,
                1,
                fill=False,
                edgecolor="orange",
                linewidth=2,
            )
            ax.add_patch(rect)


def generate_combined_figure(heatmap_data_zero, heatmap_data_rag):
    """Generate the combined heatmap figure."""
    fig_width = 7
    fig_height = 8
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    # Create zero-shot heatmap (top) - no colorbar
    im1 = sns.heatmap(
        heatmap_data_zero,
        annot=True,
        fmt=".2f",
        cmap="seismic_r",
        cbar=False,  # Disable individual colorbar
        linewidths=0.5,
        annot_kws={"size": 8},
        vmin=0,
        vmax=1,
        ax=ax1,
    )

    ax1.set_title("Zero-shot")
    ax1.set_xlabel("")
    ax1.set_ylabel(None)
    ax1.tick_params(axis="x", labelbottom=True, labelsize=8)
    ax1.tick_params(axis="y", rotation=0, labelsize=8)

    # Add orange boxes for max values
    add_max_value_boxes(ax1, heatmap_data_zero)

    # Create RAG k=6 heatmap (bottom) - no colorbar
    im2 = sns.heatmap(
        heatmap_data_rag,
        annot=True,
        fmt=".2f",
        cmap="seismic_r",
        cbar=False,  # Disable individual colorbar
        linewidths=0.5,
        annot_kws={"size": 8},
        vmin=0,
        vmax=1,
        ax=ax2,
    )

    ax2.set_title(r"$k$=6")
    ax2.set_xlabel("Factcheck Year")
    ax2.set_ylabel(None)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)
    ax2.tick_params(axis="x", labelsize=8)

    # Add orange boxes for max values
    add_max_value_boxes(ax2, heatmap_data_rag)

    # Create shared colorbar at the bottom
    # Position: [left, bottom, width, height] - centered and narrower
    cbar_ax = fig.add_axes([0.35, -0.04, 0.3, 0.025])

    # Create colorbar using the second heatmap's mappable
    cbar = fig.colorbar(im2.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("F1 macro", size=8)
    cbar.ax.tick_params(labelsize=8)

    # Adjust layout with more space for colorbar
    plt.subplots_adjust(bottom=0.08, hspace=0.25)  # Make room for colorbar

    return fig


def main():
    """Main function to generate and save the combined heatmap figure."""
    # Ensure figures directory exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    multi_year_data = load_data()

    # Prepare data for both configurations
    rag_6_data, zero_shot_data = prepare_data(
        multi_year_data,
    )

    # Create heatmap data for both configurations
    heatmap_data_zero = create_heatmap_data(
        zero_shot_data, "Zero-shot", add_search_suffix=True
    )
    heatmap_data_rag = create_heatmap_data(
        rag_6_data, "RAG k=6", add_search_suffix=True
    )

    # Generate combined figure
    fig = generate_combined_figure(heatmap_data_zero, heatmap_data_rag)

    # Save figure in both formats
    pdf_path = figures_dir / f"{OUTPUT_FILENAME}.pdf"
    png_path = figures_dir / f"{OUTPUT_FILENAME}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=600)
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    print("\nCombined heatmap figure saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
