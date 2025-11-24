"""
Generate a model performance comparison plot showing F1 scores across different RAG settings.

This script creates a comprehensive visualization with:
- Main panel showing model performance with RAG configurations (0, 3, 6, 9 items)
- Top-k retrieval accuracy bar chart
- Retrieval rank distribution strip plot

Input:
    - multi_class_all_summary_0.25.parquet: Performance metrics for all models (~6k claims)
    - retrieval_ranks.parquet: Retrieval ranking data for RAG accuracy analysis

Output:
    - base_model_performance_comparison.pdf: Multi-panel figure saved to figures/ directory

Author: Matthew DeVerna
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Set font size
plt.rcParams["font.size"] = 10

# Change to script directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
CLASSIFICATION_REPORT_PATH = (
    "../../data/cleaned/classification_reports/multi_class_all_summary_0.25.parquet"
)
RETRIEVAL_RANKS_PATH = "../../data/cleaned/retrieval_ranks.parquet"
TOPK_ACCURACY_PATH = "../../data/cleaned/topk_accuracy.parquet"
OUTPUT_PATH = "../../figures/base_model_performance_comparison_one_column.pdf"

# Configuration
SELECTED_MODELS = [
    "DeepSeek-V3",
    "Llama-3.2-3B-Instruct-Turbo",
    "Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama-3.2-90B-Vision-Instruct-Turbo",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-pro-exp-02-05",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
]

MODEL_NAME_MAP = {
    "DeepSeek-V3": "DeepSeek-V3",
    "Llama-3.2-3B-Instruct-Turbo": "Llama 3.2 3B",
    "Llama-3.2-11B-Vision-Instruct-Turbo": "Llama 3.2 11B",
    "Llama-3.2-90B-Vision-Instruct-Turbo": "Llama 3.2 90B",
    "gemini-2.0-flash-lite": "Gemini 2.0\nFlash Lite",
    "gemini-2.0-flash": "Gemini 2.0\nFlash",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0\nPro",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-4o-2024-11-20": "GPT-4o",
}

MODEL_FAMILY_COLORS = {
    "OpenAI": "salmon",
    "Google": "royalblue",
    "Meta": "gold",
    "DeepSeek": "darkslategray",
}

K_MARKERS = {
    0: "o",  # Circle for zero shot
    3: "P",  # Plus for k=3
    6: "D",  # Diamond for k=6
    9: "^",  # Triangle up for k=9
}

K_LABELS = {0: r"$k=0$", 3: r"$k=3$", 6: r"$k=6$", 9: r"$k=9$"}
RETRIEVAL_THRESHOLDS = [3, 6, 9]


def get_model_family(model_name):
    """Determine model family from model name."""
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "Meta"
    elif "gemini" in model_lower:
        return "Google"
    elif "gpt" in model_lower:
        return "OpenAI"
    elif "deepseek" in model_lower:
        return "DeepSeek"
    else:
        return "other"


def load_data():
    """Load and prepare all required datasets."""
    # Load performance data
    performance_df = pd.read_parquet(CLASSIFICATION_REPORT_PATH)

    # Load retrieval ranking data
    retrieval_df = pd.read_parquet(RETRIEVAL_RANKS_PATH)

    # Load pre-calculated top-k accuracy
    topk_accuracy_df = pd.read_parquet(TOPK_ACCURACY_PATH)

    return performance_df, retrieval_df, topk_accuracy_df


def prepare_performance_data(performance_df):
    """Filter and prepare performance data for selected models."""
    filtered_df = performance_df[
        (performance_df["model"].isin(SELECTED_MODELS))
        & (performance_df["used_web_search"] == False)
    ].copy()

    filtered_df["model"] = pd.Categorical(
        filtered_df["model"], categories=SELECTED_MODELS, ordered=True
    )

    return filtered_df


def prepare_retrieval_accuracy(topk_accuracy_df):
    """Convert top-k accuracy dataframe to dictionary for plotting."""
    # Convert to dictionary for compatibility with existing plotting code
    recall_dict = {}
    for _, row in topk_accuracy_df.iterrows():
        # Round for plotting purposes
        recall_dict[row["k"]] = round(row["topk_accuracy"], 3)

    return recall_dict


def plot_model_points(ax, group, ypos, x_column):
    """Plot scatter points for a single model across different k values."""
    model_family = get_model_family(group.iloc[0]["model"])
    color = MODEL_FAMILY_COLORS.get(model_family, "black")

    for _, row in group.iterrows():
        k_val = int(row["num_rag_items"])
        marker = K_MARKERS.get(k_val, "o")
        ax.scatter(
            row[x_column],
            ypos,
            marker=marker,
            color=color,
            s=40,
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
            alpha=0.7,
        )


def add_mean_line_and_bracket(ax, group, ypos, x_column):
    """Add mean line and comparison bracket for RAG vs zero-shot."""
    # Plot vertical mean line for k=3,6,9 if they exist
    rag_group = group[group["num_rag_items"].isin([3, 6, 9])]
    if len(rag_group) < 2:
        return

    mean_val = rag_group[x_column].mean()
    ymin = ypos - 0.25
    ymax = ypos + 0.25

    # Draw mean line
    ax.vlines(
        x=mean_val,
        ymin=ymin,
        ymax=ymax,
        color="k",
        linestyles="-",
        zorder=-3,
        linewidth=0.5,
    )

    # Add mean value annotation
    y_text = ymax + 0.05
    ax.text(
        mean_val,
        y_text,
        f"{mean_val:.2f}",
        color="k",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    # Add horizontal bracket below connecting k=0 to mean
    zero_shot_group = group[group["num_rag_items"] == 0]
    if zero_shot_group.empty:
        return

    zero_val = zero_shot_group[x_column].iloc[0]
    bracket_y = ypos - 0.4
    tick_height = 0.05

    # Draw horizontal bracket line
    ax.hlines(
        y=bracket_y,
        xmin=zero_val,
        xmax=mean_val,
        color="k",
        linestyles="-",
        linewidth=0.5,
        zorder=-2,
    )

    # Draw vertical ticks at both ends
    for x_val in [zero_val, mean_val]:
        ax.vlines(
            x=x_val,
            ymin=bracket_y,
            ymax=bracket_y + tick_height,
            color="gray",
            linestyles="-",
            linewidth=0.5,
            alpha=0.7,
            zorder=-2,
        )

    # Add difference annotation
    diff_val = mean_val - zero_val
    bracket_center_x = (zero_val + mean_val) / 2
    ax.text(
        bracket_center_x,
        bracket_y + 0.08,
        f"{diff_val:+.2f}",
        color="k",
        ha="center",
        va="bottom",
        fontsize=7,
    )


def setup_performance_axes(ax, unique_models, x_column):
    """Configure axes styling for performance plot."""
    # Ensure grid lines are not clipped
    for line in ax.get_xgridlines():
        line.set_clip_on(False)

    # Set up y-axis with model positions
    ax.set_yticks(range(len(unique_models)))
    ax.set_yticklabels(
        [MODEL_NAME_MAP.get(model, model) for model in unique_models], fontsize=10
    )

    ax.set_xlabel(x_column.replace("-", " "))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(unique_models) - 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_performance_plot(ax, df, x_column="f1-score-macro-avg"):
    """Create the main model performance plot."""
    # Get unique models from categorical or unique values
    unique_models = (
        df["model"].cat.categories
        if hasattr(df["model"], "cat")
        else df["model"].unique()
    )

    for i, model in enumerate(unique_models):
        group = df[df["model"] == model].sort_values("num_rag_items")
        if len(group) == 0:
            continue

        plot_model_points(ax, group, i, x_column)
        add_mean_line_and_bracket(ax, group, i, x_column)

    setup_performance_axes(ax, unique_models, x_column)
    return ax


def create_legend_elements():
    """Create legend elements for k values and model families."""
    # Create k value legend elements
    k_legend_elements = [
        Line2D(
            [0],
            [0],
            marker=K_MARKERS[k_val],
            color="gray",
            linestyle="None",
            markersize=5,
            label=K_LABELS[k_val],
        )
        for k_val in [0, 3, 6, 9]
    ]

    # Create model family legend elements
    family_legend_elements = [
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=model_family,
        )
        for model_family, color in MODEL_FAMILY_COLORS.items()
    ]

    return k_legend_elements + family_legend_elements


def create_retrieval_rank_plot(ax, retrieval_df):
    """Create retrieval rank distribution strip plot."""
    sns.stripplot(
        data=retrieval_df,
        x="rank",
        ax=ax,
        color="darkslategray",
        alpha=0.5,
        size=2,
        zorder=1,
        jitter=0.2,
    )

    # Add mean and median markers
    mean_val = retrieval_df["rank"].mean()
    ax.plot(mean_val, 0, color="darkorange", marker="o", markersize=5, zorder=2)

    median_val = retrieval_df["rank"].median()
    ax.plot(median_val, 0, color="crimson", marker="s", markersize=5, zorder=2)

    # Configure axes
    ax.set_xlabel("retrieval rank", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-0.3, 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="dotted", alpha=0.7)


def create_retrieval_accuracy_plot(ax, recall_dict):
    """Create top-k retrieval accuracy bar chart."""
    keys = sorted(recall_dict.keys())
    values = [recall_dict[k] for k in keys]

    ax.barh(keys, values, color="darkslategray", height=1.5)

    # Annotate each bar with its recall value
    for k, val in zip(keys, values):
        ax.text(val + 0.01, k, f"{val:.2f}", va="center", ha="left", fontsize=8)

    # Configure axes
    ax.set_xlabel("top-$k$ retrieval accuracy", fontsize=9)
    ax.set_ylabel("$k$", fontsize=9)
    ax.set_yticks(keys)
    ax.set_yticklabels(keys)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="dotted", alpha=0.7)


def create_comprehensive_figure():
    """Create the complete model performance comparison figure."""
    # Load and prepare data
    performance_df, retrieval_df, topk_accuracy_df = load_data()
    selected_df = prepare_performance_data(performance_df)
    recall_dict = prepare_retrieval_accuracy(topk_accuracy_df)

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(4, 8))
    gs = GridSpec(nrows=3, ncols=1, height_ratios=[0.8, 0.1, 0.1])

    ax1 = fig.add_subplot(gs[0, 0])  # Main performance plot
    ax2 = fig.add_subplot(gs[1, 0])  # Retrieval rank distribution
    ax3 = fig.add_subplot(gs[2, 0])  # Top-k retrieval accuracy

    # Create main performance plot
    create_performance_plot(ax1, selected_df, "f1-score-macro-avg")
    ax1.set_xlabel("macro F1 score", fontsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", linestyle="dotted", alpha=0.7, linewidth=0.5)

    # Add comprehensive legend
    legend_handles = create_legend_elements()
    ax1.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        frameon=True,
        fontsize=8,
        ncol=2,
    )

    # Create retrieval analysis plots
    create_retrieval_rank_plot(ax2, retrieval_df)
    create_retrieval_accuracy_plot(ax3, recall_dict)

    # Add panel annotations
    ax1.text(
        -0.08, 1.05, "(a)", transform=ax1.transAxes, fontsize=10, va="top", ha="right"
    )
    ax2.text(
        -0.15, 1.1, "(b)", transform=ax2.transAxes, fontsize=10, va="top", ha="right"
    )
    ax3.text(
        -0.15, 1.1, "(c)", transform=ax3.transAxes, fontsize=10, va="top", ha="right"
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    return fig


def main():
    """Main function to generate the model performance comparison figure."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Define output paths
    pdf_path = OUTPUT_PATH
    png_path = OUTPUT_PATH.replace(".pdf", ".png")

    # Create and save figure in both formats
    fig = create_comprehensive_figure()
    fig.savefig(pdf_path, bbox_inches="tight", dpi=900)
    fig.savefig(png_path, bbox_inches="tight", dpi=900)

    print("Model performance comparison plot saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
