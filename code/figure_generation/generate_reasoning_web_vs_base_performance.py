"""
Generate a performance comparison plot showing F1 scores for fact-checking models
across different settings (zero-shot vs RAG).

This script creates a comprehensive visualization with two panels:
1. Reasoning models vs their corresponding base models
2. Web search models vs their corresponding base models

Input:
    - multi_class_all_summary_0.25.parquet: Performance metrics for all models (~6k claims)

Output:
    - reasoning_web_vs_base_performance.pdf: Multi-panel figure saved to figures/ directory

Author: Matthew DeVerna
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Set font size
plt.rcParams["font.size"] = 10

# Change to script directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
DATA_PATH = (
    "../../data/cleaned/classification_reports/multi_class_all_summary_0.25.parquet"
)
OUTPUT_PATH = "../../figures/reasoning_web_vs_base_performance.pdf"

SCATTER_ALPHA = 0.75

# Model configurations
BASE_MODELS = [
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

ADDITIONAL_MODELS = [
    "DeepSeek-R1",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
]

# Model name mappings for display
MODEL_NAME_MAP = {
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

# Model family color mapping
MODEL_FAMILY_COLORS = {
    "OpenAI": "salmon",
    "Google": "royalblue",
    # "Meta": "gold",
    "DeepSeek": "darkslategray",
}

# Panel configurations
REASONING_PANEL_ORDER = [
    "DeepSeek-R1",
    "gemini-2.0-flash-thinking-exp-01-21",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
]

REASONING_BASE_MODEL_CONFIG = [
    {
        "model": "gpt-4o-2024-11-20",
        "y_models": ["o1-2024-12-17", "o3-mini-2025-01-31"],
    },
    {
        "model": "gemini-2.0-flash",
        "y_models": ["gemini-2.0-flash-thinking-exp-01-21"],
    },
    {
        "model": "DeepSeek-V3",
        "y_models": ["DeepSeek-R1"],
    },
]

SEARCH_PANEL_ORDER = [
    ("gemini-2.0-flash-thinking-exp-01-21", True),
    ("gemini-2.0-flash", True),
    ("gpt-4o-mini-search-preview-2025-03-11", True),
    ("gpt-4o-search-preview-2025-03-11", True),
]

SEARCH_BASE_MODEL_CONFIG = [
    {
        "model": "gpt-4o-2024-11-20",
        "y_model": ("gpt-4o-search-preview-2025-03-11", True),
    },
    {
        "model": "gpt-4o-mini-2024-07-18",
        "y_model": ("gpt-4o-mini-search-preview-2025-03-11", True),
    },
    {
        "model": "gemini-2.0-flash",
        "y_model": ("gemini-2.0-flash", True),
    },
    {
        "model": "gemini-2.0-flash-thinking-exp-01-21",
        "y_model": ("gemini-2.0-flash-thinking-exp-01-21", True),
    },
]


def get_model_family(model_name):
    """Determine model family from model name."""
    model_lower = model_name.lower()
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "OpenAI"
    elif "deepseek" in model_lower:
        return "DeepSeek"
    elif "llama" in model_lower:
        return "Meta"
    else:
        return "Google"


def format_difference(value):
    """Format difference value with appropriate decimal places."""
    if abs(value) < 0.01:
        return f"{value:+.3f}"
    else:
        return f"{value:+.2f}"


def load_and_prepare_data():
    """Load and prepare performance data for analysis."""
    df = pd.read_parquet(DATA_PATH)

    # Filter for selected models
    all_models = BASE_MODELS + ADDITIONAL_MODELS
    selected_df = df[df["model"].isin(all_models)].copy()

    # Set categorical ordering for consistent plotting
    selected_df["model"] = pd.Categorical(
        selected_df["model"], categories=all_models, ordered=True
    )

    return selected_df


def prepare_f1_comparison_data(df):
    """Prepare F1 score data for zero-shot vs RAG comparison."""
    # Extract RAG performance (k=6)
    f1_rag = df[df["num_rag_items"] == 6][
        ["model", "used_web_search", "f1-score-macro-avg"]
    ].copy()
    f1_rag.rename(columns={"f1-score-macro-avg": "f1_macro"}, inplace=True)
    f1_rag["setting"] = "rag"

    # Extract zero-shot performance
    f1_zero = df[df["num_rag_items"] == 0][
        ["model", "used_web_search", "f1-score-macro-avg"]
    ].copy()
    f1_zero.rename(columns={"f1-score-macro-avg": "f1_macro"}, inplace=True)
    f1_zero["setting"] = "zero_shot"

    # Combine datasets
    return pd.concat([f1_rag, f1_zero], ignore_index=True)


def filter_reasoning_data(f1_df):
    """Filter data for reasoning models panel."""
    return f1_df[
        (f1_df["used_web_search"] == False)
        & (f1_df["model"].isin(REASONING_PANEL_ORDER))
    ].copy()


def filter_search_data(f1_df):
    """Filter data for search models panel."""
    search_models = [model for model, _ in SEARCH_PANEL_ORDER]
    return f1_df[f1_df["model"].isin(search_models)].copy()


def draw_bracket(ax, x1, x2, y, annotation_text=None):
    """Draw a horizontal bracket between two x positions at a given y position."""
    vertical_padding = 0.2
    height = 0.05
    linewidth = 0.8
    color = "black"

    # Calculate y positions
    horizontal_line_y = y + vertical_padding + height
    tick_end_y = y + vertical_padding

    # Draw horizontal line
    ax.plot(
        [x1, x2],
        [horizontal_line_y, horizontal_line_y],
        color=color,
        linewidth=linewidth,
        zorder=4,
    )

    # Draw vertical ticks
    for x_val in [x1, x2]:
        ax.plot(
            [x_val, x_val],
            [horizontal_line_y, tick_end_y],
            color=color,
            linewidth=linewidth,
            zorder=4,
        )

    # Add text annotation if provided
    if annotation_text is not None:
        text_x = (x1 + x2) / 2
        text_y = horizontal_line_y + 0.02
        ax.text(
            text_x,
            text_y,
            annotation_text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            zorder=5,
        )


def draw_bracket_below(ax, x1, x2, y, annotation_text=None):
    """Draw a horizontal bracket below the data points."""
    vertical_padding = 0.2
    height = 0.05
    linewidth = 0.8
    color = "black"

    # Calculate y positions (below the data point)
    horizontal_line_y = y - vertical_padding - height
    tick_end_y = y - vertical_padding

    # Draw horizontal line
    ax.plot(
        [x1, x2],
        [horizontal_line_y, horizontal_line_y],
        color=color,
        linewidth=linewidth,
        zorder=4,
    )

    # Draw vertical ticks
    for x_val in [x1, x2]:
        ax.plot(
            [x_val, x_val],
            [horizontal_line_y, tick_end_y],
            color=color,
            linewidth=linewidth,
            zorder=4,
        )

    # Add text annotation if provided
    if annotation_text is not None:
        text_x = (x1 + x2) / 2
        text_y = horizontal_line_y + 0.01
        ax.text(
            text_x,
            text_y,
            annotation_text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            zorder=5,
        )


def get_model_performance_values(panel_df, model_info, is_web_search):
    """Extract zero-shot and RAG performance values for a model."""
    if is_web_search:
        model, web_search_flag = model_info
        sub = panel_df[
            (panel_df["model"] == model)
            & (panel_df["used_web_search"] == web_search_flag)
        ]
    else:
        model = model_info
        sub = panel_df[panel_df["model"] == model]

    # Check if both zero-shot and RAG data exist
    zero_data = sub[sub["setting"] == "zero_shot"]
    rag_data = sub[sub["setting"] == "rag"]

    if zero_data.empty or rag_data.empty:
        return None, None

    return zero_data["f1_macro"].iloc[0], rag_data["f1_macro"].iloc[0]


def plot_model_points(
    ax, zero_val, rag_val, y_pos, model_name, show_zero_label, show_rag_label
):
    """Plot zero-shot and RAG performance points for a model."""
    color = MODEL_FAMILY_COLORS.get(get_model_family(model_name), "gray")

    # Plot zero-shot performance
    ax.scatter(
        zero_val,
        y_pos,
        marker="o",
        color=color,
        zorder=3,
        label="zero shot" if show_zero_label else "",
        s=30,
        alpha=SCATTER_ALPHA,
    )

    # Plot RAG performance
    ax.scatter(
        rag_val,
        y_pos,
        marker="D",
        color=color,
        zorder=3,
        label="RAG" if show_rag_label else "",
        alpha=SCATTER_ALPHA,
    )


def plot_base_model_markers(ax, f1_df, vertical_lines, models_in_panel, is_web_search):
    """Plot base model reference markers."""
    for vline_data in vertical_lines:
        base_model = vline_data["model"]

        # Get zero-shot performance for base model
        base_data = f1_df[
            (f1_df["model"] == base_model)
            & (f1_df["used_web_search"] == False)
            & (f1_df["setting"] == "zero_shot")
        ]

        if base_data.empty:
            continue

        x_pos = base_data["f1_macro"].iloc[0]
        color = MODEL_FAMILY_COLORS.get(get_model_family(base_model), "gray")

        if is_web_search:
            target_model_info = vline_data["y_model"]
            try:
                y_pos = [
                    i
                    for i, model_info in enumerate(models_in_panel)
                    if model_info == target_model_info
                ][0]
                ax.scatter(
                    x_pos,
                    y_pos,
                    marker="v",
                    color=color,
                    zorder=3,
                    s=50,
                    alpha=SCATTER_ALPHA,
                )
            except IndexError:
                continue
        else:
            y_models = vline_data["y_models"]
            for target_model in y_models:
                try:
                    y_pos = models_in_panel.index(target_model)
                    ax.scatter(
                        x_pos,
                        y_pos,
                        marker="v",
                        color=color,
                        zorder=3,
                        s=50,
                        alpha=SCATTER_ALPHA,
                    )
                except ValueError:
                    continue


def add_comparison_brackets(
    ax, f1_df, panel_df, models_in_panel, vertical_lines, is_web_search
):
    """Add brackets showing performance comparisons."""
    for i, model_info in enumerate(models_in_panel):
        zero_val, rag_val = get_model_performance_values(
            panel_df, model_info, is_web_search
        )

        if zero_val is None or rag_val is None:
            continue

        # Find corresponding base model performance
        base_val = None
        if vertical_lines:
            for vline_data in vertical_lines:
                base_model = vline_data["model"]

                # Check if this model corresponds to this base model
                model_matches = False
                if is_web_search:
                    target_model_info = vline_data["y_model"]
                    model_matches = model_info == target_model_info
                else:
                    model = model_info
                    y_models = vline_data["y_models"]
                    model_matches = model in y_models

                if model_matches:
                    # Get base model zero-shot performance
                    base_data = f1_df[
                        (f1_df["model"] == base_model)
                        & (f1_df["used_web_search"] == False)
                        & (f1_df["setting"] == "zero_shot")
                    ]
                    if not base_data.empty:
                        base_val = base_data["f1_macro"].iloc[0]
                    break

        # Draw brackets if base model performance is available
        if base_val is not None:
            base_to_zero_diff = zero_val - base_val
            zero_to_rag_diff = rag_val - zero_val

            # Bracket 1: base model vs reasoning/search zero-shot (above)
            draw_bracket(
                ax,
                base_val,
                zero_val,
                i,
                annotation_text=format_difference(base_to_zero_diff),
            )

            # Bracket 2: reasoning/search zero-shot vs RAG (below)
            draw_bracket_below(
                ax,
                zero_val,
                rag_val,
                i,
                annotation_text=format_difference(zero_to_rag_diff),
            )


def create_y_axis_labels(models_in_panel, is_web_search):
    """Create appropriate y-axis labels for the panel."""
    if is_web_search:
        labels = []
        for model_info in models_in_panel:
            model, web_search_flag = model_info
            base_label = MODEL_NAME_MAP.get(model, model)

            if "gemini" in model.lower():
                suffix = " + Search" if web_search_flag else " (base)"
                labels.append(base_label + suffix)
            elif "search" in model.lower():
                labels.append(base_label)
            else:
                labels.append(base_label + " (base)")
        return labels
    else:
        return [MODEL_NAME_MAP.get(m, m) for m in models_in_panel]


def setup_panel_axes(ax, models_in_panel, is_web_search, xlabel=False, buffer=0.0):
    """Configure axes styling and labels for a panel."""
    # Grid and basic styling
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="dotted", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y-axis configuration
    ax.set_yticks(np.arange(len(models_in_panel)))
    labels = create_y_axis_labels(models_in_panel, is_web_search)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-buffer, len(models_in_panel) - 1 + buffer)

    # X-axis configuration
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    if xlabel:
        ax.set_xlabel("macro F1")


def get_models_in_panel(custom_order, panel_df, is_web_search):
    """Get list of models that exist in the panel data."""
    if is_web_search:
        return [m for m in custom_order if any(panel_df["model"] == m[0])]
    else:
        return [m for m in custom_order if m in panel_df["model"].unique()]


def plot_panel(
    ax,
    f1_df,
    panel_df,
    custom_order,
    vertical_lines,
    is_web_search=False,
    xlabel=False,
    buffer=0.0,
):
    """Plot a panel showing zero-shot vs RAG performance for a set of models."""
    models_in_panel = get_models_in_panel(custom_order, panel_df, is_web_search)

    # Track legend labels
    show_zero_label, show_rag_label = True, True

    # Plot model performance points
    for i, model_info in enumerate(models_in_panel):
        zero_val, rag_val = get_model_performance_values(
            panel_df, model_info, is_web_search
        )

        if zero_val is None or rag_val is None:
            continue

        model_name = model_info if not is_web_search else model_info[0]
        plot_model_points(
            ax, zero_val, rag_val, i, model_name, show_zero_label, show_rag_label
        )

        show_zero_label, show_rag_label = False, False

    # Add base model markers and comparison brackets
    plot_base_model_markers(ax, f1_df, vertical_lines, models_in_panel, is_web_search)
    add_comparison_brackets(
        ax, f1_df, panel_df, models_in_panel, vertical_lines, is_web_search
    )

    # Configure axes
    setup_panel_axes(ax, models_in_panel, is_web_search, xlabel, buffer)


def create_legend():
    """Create comprehensive legend for the figure."""
    # Marker shapes
    marker_elements = [
        Line2D(
            [],
            [],
            marker="v",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="baseline",
            linestyle="None",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label=r"$k=0$",
            linestyle="None",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label=r"$k=6$",
            linestyle="None",
        ),
    ]

    # Model family colors
    family_elements = [
        Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=family,
        )
        for family, color in MODEL_FAMILY_COLORS.items()
    ]

    return marker_elements + family_elements


def create_comprehensive_figure():
    """Create the complete reasoning vs web search performance comparison figure."""
    # Load and prepare data
    df = load_and_prepare_data()
    f1_df = prepare_f1_comparison_data(df)

    # Filter data for each panel
    reasoning_df = filter_reasoning_data(f1_df)
    search_df = filter_search_data(f1_df)

    # Create figure with two panels
    fig = plt.figure(figsize=(6, 4.5))
    gs = GridSpec(2, 1, height_ratios=[0.5, 0.5], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])  # Reasoning panel
    ax2 = fig.add_subplot(gs[1])  # Search panel

    # Plot panels
    plot_panel(
        ax1,
        f1_df,
        reasoning_df,
        REASONING_PANEL_ORDER,
        REASONING_BASE_MODEL_CONFIG,
        buffer=0.4,
    )

    plot_panel(
        ax2,
        f1_df,
        search_df,
        SEARCH_PANEL_ORDER,
        SEARCH_BASE_MODEL_CONFIG,
        is_web_search=True,
        xlabel=True,
        buffer=0.4,
    )

    # Configure panel-specific settings
    ax1.set_xticklabels([])  # Remove x-axis labels for top panel
    ax1.set_ylabel("reasoning")
    ax2.set_ylabel("search")

    # Add comprehensive legend
    legend_handles = create_legend()
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.1, 0.9),
        fontsize=9,
        frameon=True,
    )

    # Add panel labels
    ax1.text(
        -0.5, 1.05, "(a)", transform=ax1.transAxes, fontsize=10, va="top", ha="right"
    )
    ax2.text(
        -0.5, 1.05, "(b)", transform=ax2.transAxes, fontsize=10, va="top", ha="right"
    )

    return fig


def main():
    """Main function to generate the reasoning vs web search performance comparison figure."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Define output paths
    pdf_path = OUTPUT_PATH
    png_path = OUTPUT_PATH.replace(".pdf", ".png")

    # Create and save figure in both formats
    fig = create_comprehensive_figure()
    fig.savefig(pdf_path, bbox_inches="tight", dpi=900)
    fig.savefig(png_path, bbox_inches="tight", dpi=900)

    print("Performance plot saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
