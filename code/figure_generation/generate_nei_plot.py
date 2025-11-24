#!/usr/bin/env python3
"""
Purpose:
    Generate a 2x2 panel figure analyzing the use of the "Not Enough Information" (NEI)
    label in fact-checking model evaluations. The figure includes:
    1. NEI detection performance metrics across model families
    2. Confusion matrix showing NEI vs verdict prediction patterns
    3. NEI response rates by veracity labels
    4. Top-k retrieval accuracy by veracity labels

Input:
    - nei_performance_metrics.parquet: NEI detection performance metrics
    - nei_confusion_summary.parquet: Confusion matrix data for NEI detection
    - nei_given_props.parquet: NEI response proportions by veracity labels
    - topk_accuracy_by_veracity.parquet: Retrieval accuracy for different veracity labels

Output:
    - nei_analysis_plot.pdf: Multi-panel figure saved to figures/ directory
    - nei_analysis_plot.png: Same figure in PNG format for easier viewing

Author: Matthew DeVerna
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
DATA_DIR = "../../data/cleaned"
FIGURES_DIR = "../../figures"
OUTPUT_FILENAME = "nei_analysis_plot"

# Selected models to include in the analysis
SELECTED_MODELS = [
    "DeepSeek-R1",
    "DeepSeek-V3",
    "Llama-3.2-3B-Instruct-Turbo",
    "Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama-3.2-90B-Vision-Instruct-Turbo",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-pro-exp-02-05",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview-2025-03-11",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
]


def load_data():
    """Load the required data files."""
    performance_df = pd.read_parquet(
        os.path.join(DATA_DIR, "nei_performance_metrics.parquet"),
    )
    confusion_df = pd.read_parquet(
        os.path.join(DATA_DIR, "nei_confusion_summary.parquet"),
    )
    nei_given_props_df = pd.read_parquet(
        os.path.join(DATA_DIR, "nei_given_props.parquet"),
    )
    topk_accuracy_df = pd.read_parquet(
        os.path.join(DATA_DIR, "topk_accuracy_by_veracity.parquet"),
    )

    return performance_df, confusion_df, nei_given_props_df, topk_accuracy_df


def prepare_data(performance_df, confusion_df, nei_given_props_df, topk_accuracy_df):
    """Prepare and transform data for plotting."""
    # Filter data to include only selected models
    performance_df = performance_df[
        performance_df["model"].isin(SELECTED_MODELS)
    ].copy()
    confusion_df = confusion_df[confusion_df["model"].isin(SELECTED_MODELS)].copy()
    nei_given_props_df = nei_given_props_df[
        nei_given_props_df["model"].isin(SELECTED_MODELS)
    ].copy()

    # Prepare NEI given props data
    nei_given_props_melt = nei_given_props_df.melt(
        id_vars=["model", "num_rag_items", "used_web_search"],
        value_vars=[
            "prop_nei_overall",
            "prop_nei_false",
            "prop_nei_half_true",
            "prop_nei_mostly_false",
            "prop_nei_mostly_true",
            "prop_nei_pants_on_fire",
            "prop_nei_true",
        ],
        value_name="proportion",
    )

    # Map variable names to readable labels
    nei_given_props_melt["variable"] = nei_given_props_melt["variable"].map(
        {
            "prop_nei_overall": "overall",
            "prop_nei_pants_on_fire": "pants on fire",
            "prop_nei_false": "false",
            "prop_nei_mostly_false": "mostly false",
            "prop_nei_half_true": "half true",
            "prop_nei_mostly_true": "mostly true",
            "prop_nei_true": "true",
        }
    )

    # Convert to ordered categorical
    nei_given_props_melt["variable"] = pd.Categorical(
        nei_given_props_melt["variable"],
        categories=[
            "overall",
            "pants on fire",
            "false",
            "mostly false",
            "half true",
            "mostly true",
            "true",
        ],
        ordered=True,
    )

    # Remove sample-based filtering - we'll use all data

    # Melt performance data
    performance_melt = performance_df.melt(
        id_vars=["model", "num_rag_items", "used_web_search"],
        value_vars=["precision", "recall", "f1_score", "accuracy"],
        var_name="metric",
        value_name="value",
    )

    # Add model family mapping
    model_family_map = {
        "DeepSeek-R1": "DeepSeek",
        "DeepSeek-V3": "DeepSeek",
        "Llama-3.2-11B-Vision-Instruct-Turbo": "Meta",
        "Llama-3.2-3B-Instruct-Turbo": "Meta",
        "Llama-3.2-90B-Vision-Instruct-Turbo": "Meta",
        "gemini-1.5-flash-002": "Google",
        "gemini-1.5-flash-8b-001": "Google",
        "gemini-1.5-pro-002": "Google",
        "gemini-2.0-flash": "Google",
        "gemini-2.0-flash-lite": "Google",
        "gemini-2.0-flash-thinking-exp-01-21": "Google",
        "gemini-2.0-pro-exp-02-05": "Google",
        "gpt-4o-2024-08-06": "OpenAI",
        "gpt-4o-2024-11-20": "OpenAI",
        "gpt-4o-mini-2024-07-18": "OpenAI",
        "gpt-4o-mini-search-preview-2025-03-11": "OpenAI",
        "gpt-4o-search-preview-2025-03-11": "OpenAI",
        "o1-2024-12-17": "OpenAI",
        "o3-mini-2025-01-31": "OpenAI",
    }

    performance_melt["model_family"] = performance_melt["model"].map(model_family_map)
    nei_given_props_melt["model_family"] = nei_given_props_melt["model"].map(
        model_family_map
    )
    confusion_df["model_family"] = confusion_df["model"].map(model_family_map)

    # Prepare top-k accuracy data - average across k values
    topk_accuracy_avg = (
        topk_accuracy_df.groupby("veracity_label")["topk_accuracy"].mean().reset_index()
    )

    # Order to match panel 3 (excluding "overall")
    # Using the actual labels: ['False' 'Half true' 'Mostly false' 'Mostly true' 'Pants on fire' 'True']
    veracity_order = [
        "Pants on fire",
        "False",
        "Mostly false",
        "Half true",
        "Mostly true",
        "True",
    ]

    # Sort dataframe by this order
    topk_accuracy_avg["order"] = topk_accuracy_avg["veracity_label"].map(
        {label: i for i, label in enumerate(veracity_order)}
    )
    topk_accuracy_avg = topk_accuracy_avg.sort_values("order")

    return performance_melt, nei_given_props_melt, confusion_df, topk_accuracy_avg


def create_confusion_matrix(confusion_df):
    """Create confusion matrix with proportions."""
    # Calculate mean confusion matrix values across all selected models
    confusion_mean = confusion_df[
        ["true_positive", "false_positive", "true_negative", "false_negative"]
    ].mean()

    # Calculate total for proportion calculation
    total = confusion_mean.sum()

    # Create confusion matrix with proportions
    conf_matrix = pd.DataFrame(
        {
            "predicted NEI": [
                confusion_mean["true_positive"] / total,
                confusion_mean["false_positive"] / total,
            ],
            "predicted verdict": [
                confusion_mean["false_negative"] / total,
                confusion_mean["true_negative"] / total,
            ],
        },
        index=["NEI", "summary present"],
    )

    return conf_matrix


def generate_figure(
    performance_melt, nei_given_props_melt, conf_matrix, topk_accuracy_avg
):
    """Generate the main NEI analysis figure."""
    modelfam_color_map = {
        "OpenAI": "salmon",
        "Google": "royalblue",
        "Meta": "gold",
        "DeepSeek": "darkslategray",
    }

    # Create 2x2 subplot layout
    width = 7
    height = 5
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(width, height))

    # Top-left: Performance plots
    sns.stripplot(
        data=performance_melt.sort_values("model_family"),
        x="metric",
        y="value",
        hue="model_family",
        order=["precision", "recall", "f1_score", "accuracy"],
        dodge=True,
        alpha=0.9,
        ax=ax1,
        palette=modelfam_color_map,
        legend=True,
    )

    # Update x-axis labels to make F1 score more readable
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(["precision", "recall", "F1 score", "accuracy"])

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.set_xlabel(None)
    ax1.set_ylabel("metric value")
    ax1.set_ylim(-0.05, None)
    ax1.tick_params(axis="x", rotation=0)

    # Remove legend from this axis - will add shared legend
    ax1.legend().set_visible(False)

    # Top-right: Confusion matrix
    # Convert proportions to percentages for display
    conf_matrix_pct = conf_matrix * 100

    # Create custom annotations with % symbols
    annot_labels = conf_matrix_pct.round(0).astype(int).astype(str) + "%"

    sns.heatmap(
        conf_matrix_pct,
        annot=annot_labels,
        fmt="",
        cmap="Reds",
        ax=ax2,
        square=True,
        cbar=False,
    )
    ax2.tick_params(
        axis="x",
        labelsize=10,
        rotation=0,
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    ax2.tick_params(axis="y", labelsize=10, rotation=0)

    # Add line breaks to confusion matrix labels with spaces and make lowercase
    x_labels = [
        label.get_text().replace(" ", "\n").lower().replace("nei", "NEI")
        for label in ax2.get_xticklabels()
    ]
    y_labels = [
        label.get_text().replace(" ", "\n").lower().replace("nei", "NEI")
        for label in ax2.get_yticklabels()
    ]
    ax2.set_xticklabels(x_labels, fontsize=8)
    ax2.set_yticklabels(y_labels, fontsize=8)

    # Bottom-left: NEI response rates by veracity
    sns.stripplot(
        data=nei_given_props_melt.sort_values("model_family"),
        x="variable",
        y="proportion",
        hue="model_family",
        dodge=True,
        alpha=0.9,
        ax=ax3,
        palette=modelfam_color_map,
        legend=False,
    )

    ax3.tick_params(axis="x", rotation=90)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    ax3.set_xlabel(None)
    ax3.set_ylabel("NEI response rate")
    ax3.set_ylim(-0.02, 0.35)

    # Bottom-right: Top-k retrieval accuracy by veracity
    bars = ax4.bar(
        range(len(topk_accuracy_avg)),
        topk_accuracy_avg["topk_accuracy"],
        color="darkslategrey",
    )
    ax4.set_xticks(range(len(topk_accuracy_avg)))
    # Make all veracity labels lowercase
    lowercase_labels = [
        str(label).lower() for label in topk_accuracy_avg["veracity_label"]
    ]
    ax4.set_xticklabels(lowercase_labels, rotation=90)
    ax4.set_ylabel(r"average top-$k$" + "\nretrieval accuracy")
    ax4.set_xlabel(None)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="y", linestyle="--", alpha=0.7)
    ax4.set_axisbelow(True)  # Ensure grid lines go behind bars

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, topk_accuracy_avg["topk_accuracy"])):
        ax4.text(i, value + 0.002, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    # Set y-axis limits to show full range
    ax4.set_ylim(0.85, 1.0)

    # Add panel markers
    ax1.text(
        -0.2, 1.1, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="normal"
    )
    ax2.text(
        -0.4, 1.1, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="normal"
    )
    ax3.text(
        -0.2, 1.1, "(c)", transform=ax3.transAxes, fontsize=12, fontweight="normal"
    )
    ax4.text(
        -0.35, 1.1, "(d)", transform=ax4.transAxes, fontsize=12, fontweight="normal"
    )

    # Add shared legend for model families above left two panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.28, 1.05),
        ncol=2,
        fontsize="medium",
        frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig


def main():
    """Main function to generate and save the NEI analysis figure."""
    # Load data
    performance_df, confusion_df, nei_given_props_df, topk_accuracy_df = load_data()

    # Prepare data
    performance_melt, nei_given_props_melt, confusion_df, topk_accuracy_avg = (
        prepare_data(performance_df, confusion_df, nei_given_props_df, topk_accuracy_df)
    )

    # Create confusion matrix
    conf_matrix = create_confusion_matrix(confusion_df)

    # Generate figure
    fig = generate_figure(
        performance_melt, nei_given_props_melt, conf_matrix, topk_accuracy_avg
    )

    # Save figure
    pdf_path = os.path.join(FIGURES_DIR, f"{OUTPUT_FILENAME}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{OUTPUT_FILENAME}.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to: {pdf_path}")

    # Also save as PNG for easier viewing
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Figure also saved to: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
