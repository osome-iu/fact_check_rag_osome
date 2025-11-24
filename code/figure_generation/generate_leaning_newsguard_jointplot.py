"""
Generate 2D histogram with marginal KDE plots showing political leaning vs NewsGuard score.

This script creates a visualization with:
- 2D histogram showing density of cited domains by leaning score and NewsGuard score
- Marginal KDE distributions for both dimensions
- NewsGuard credibility category annotations and percentages

Purpose:
    Visualizes the quality and political bias distribution of sources cited by LLMs
    when fact-checking claims with RAG (k=6).

Input:
    - data/cleaned/enriched_web_urls.parquet: Enriched domain information with scores

Output:
    - figures/leaning_newsguard_jointplot.pdf: 2D histogram with marginal distributions

Author: Matthew DeVerna
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colors
from matplotlib.ticker import StrMethodFormatter

# Set rc param fontsize to 10
plt.rcParams["font.size"] = 10

# Change to script directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
DATA_PATH = "../../data/cleaned/enriched_web_urls.parquet"
OUTPUT_PATH = "../../figures/leaning_newsguard_jointplot.pdf"

# Configuration
MODEL_NAME_MAP = {
    "gpt-4o-search-preview-2025-03-11": "GPT-4o Search",
    "gpt-4o-mini-search-preview-2025-03-11": "GPT-4o mini Search",
}

NEWSGUARD_CATEGORIES = [
    (75, "Generally\nCredible"),
    (60, "Credible\nwith Exceptions"),
    (40, "Proceed\nwith Caution"),
]

NEWSGUARD_CATEGORY_NAMES = [
    "Highly/Generally Credible",
    "Credible with Exceptions",
    "Proceed with Caution",
    "Proceed with Maximum Caution",
]

# Styling constants
KDE_COLOR = "midnightblue"
REFERENCE_LINE_STYLE = {"color": "black", "linestyle": "--", "linewidth": 0.75}

# Dual marginal colors (colorblind-friendly)
ALL_DATA_COLOR = "#003366"  # Blue
EXCLUDED_PF_COLOR = "#E31B23"  # Red


def get_ng_category(score):
    """Define NewsGuard category based on score."""
    if score >= 75:
        return "Highly/Generally Credible"
    elif score >= 60:
        return "Credible with Exceptions"
    elif score >= 40:
        return "Proceed with Caution"
    else:
        return "Proceed with Maximum Caution"


def load_data():
    """Load and return the required dataset."""
    df_domain_info = pd.read_parquet(DATA_PATH)
    return df_domain_info


def prepare_data(df_domain_info, exclude_politifact=False, dual_marginals=False):
    """Prepare and filter data for visualization."""
    df_domain_info["clean_model"] = df_domain_info["model"].map(MODEL_NAME_MAP)

    df_lean_quality = (
        df_domain_info[
            [
                "clean_model",
                "num_rag_items",
                "domain",
                "leaning_score",
                "newsguard_score",
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )

    source_df = df_lean_quality[df_lean_quality["num_rag_items"] == 6].copy()

    # Handle different modes
    if dual_marginals:
        # For dual marginals, return both all data and excluded PolitiFact data
        all_data_df = source_df.copy()
        excluded_pf_df = source_df[source_df["domain"] != "politifact.com"].copy()

        # Calculate percentages for both datasets
        all_data_df["ng_category"] = all_data_df["newsguard_score"].apply(
            get_ng_category
        )
        excluded_pf_df["ng_category"] = excluded_pf_df["newsguard_score"].apply(
            get_ng_category
        )

        all_percentages = (
            all_data_df["ng_category"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .to_dict()
        )
        excluded_percentages = (
            excluded_pf_df["ng_category"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .to_dict()
        )

        return all_data_df, excluded_pf_df, all_percentages, excluded_percentages

    else:
        # Original behavior
        if exclude_politifact:
            source_df = source_df[source_df["domain"] != "politifact.com"].copy()

        # Calculate NewsGuard category percentages
        source_df["ng_category"] = source_df["newsguard_score"].apply(get_ng_category)
        category_percentages = (
            source_df["ng_category"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .to_dict()
        )

        return source_df, category_percentages


def setup_axes_style(ax, remove_spines=None, remove_ticks=None):
    """Apply common styling to axes."""
    if remove_spines:
        for spine in remove_spines:
            ax.spines[spine].set_visible(False)

    if remove_ticks:
        if "x" in remove_ticks:
            ax.set_xticks([])
        if "y" in remove_ticks:
            ax.set_yticks([])

    ax.grid(False)


def create_top_kde_panel(fig, gs, source_df, excluded_pf_df=None):
    """Create the top KDE panel for political leaning scores."""
    ax_kde = fig.add_subplot(gs[0, 0])

    if excluded_pf_df is not None:
        # Dual marginal mode - plot both distributions
        sns.kdeplot(
            data=source_df,
            x="leaning_score",
            fill=False,
            ax=ax_kde,
            color=ALL_DATA_COLOR,
            linewidth=2,
            label="All data",
        )
        sns.kdeplot(
            data=excluded_pf_df,
            x="leaning_score",
            fill=False,
            ax=ax_kde,
            color=EXCLUDED_PF_COLOR,
            linewidth=2,
            linestyle="--",
            label="Excluding PolitiFact",
        )
    else:
        # Single distribution mode
        sns.kdeplot(
            data=source_df,
            x="leaning_score",
            fill=True,
            ax=ax_kde,
            color=KDE_COLOR,
            linewidth=2,
        )

    # Styling
    setup_axes_style(
        ax_kde, remove_spines=["top", "left", "right"], remove_ticks=["x", "y"]
    )
    ax_kde.set_xlabel("")
    ax_kde.set_ylabel("")
    ax_kde.set_xlim(-1, 1.01)
    ax_kde.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax_kde.set_xticklabels([])

    # Add reference line and percentages
    ax_kde.axvline(0, **REFERENCE_LINE_STYLE)

    # Calculate and add percentage annotations
    if excluded_pf_df is not None:
        # Dual mode - show both percentages
        total_all = len(source_df)
        total_excl = len(excluded_pf_df)

        left_pct_all = round(
            100 * (source_df["leaning_score"] < 0).sum() / total_all, 1
        )
        right_pct_all = round(
            100 * (source_df["leaning_score"] >= 0).sum() / total_all, 1
        )
        left_pct_excl = round(
            100 * (excluded_pf_df["leaning_score"] < 0).sum() / total_excl, 1
        )
        right_pct_excl = round(
            100 * (excluded_pf_df["leaning_score"] >= 0).sum() / total_excl, 1
        )

        y_pos = ax_kde.get_ylim()[1] * 0.3
        ax_kde.text(
            -1,
            y_pos,
            f"{left_pct_all}% ({left_pct_excl}%)",
            fontsize=9,
            ha="left",
            va="center",
        )
        ax_kde.text(
            1,
            y_pos,
            f"{right_pct_all}% ({right_pct_excl}%)",
            fontsize=9,
            ha="right",
            va="center",
        )
    else:
        # Single mode - original behavior
        total = len(source_df)
        left_pct = round(100 * (source_df["leaning_score"] < 0).sum() / total, 1)
        right_pct = round(100 * (source_df["leaning_score"] >= 0).sum() / total, 1)

        y_pos = ax_kde.get_ylim()[1] * 0.3
        ax_kde.text(-1, y_pos, f"{left_pct}%", fontsize=9, ha="left", va="center")
        ax_kde.text(1, y_pos, f"{right_pct}%", fontsize=9, ha="right", va="center")

    return ax_kde


def create_main_histogram(fig, gs, source_df):
    """Create the main 2D histogram panel."""
    ax_hist2d = fig.add_subplot(gs[1, 0])

    # Create histogram
    x_intervals = np.linspace(-1, 1.01, 30)
    y_intervals = np.linspace(-1, 101, 30)

    h = ax_hist2d.hist2d(
        source_df["leaning_score"],
        source_df["newsguard_score"],
        bins=(x_intervals, y_intervals),
        norm=colors.LogNorm(vmin=1, vmax=1e4),
        cmap="viridis_r",
    )

    # Styling
    setup_axes_style(ax_hist2d, remove_spines=["right", "top"])
    ax_hist2d.set_xlabel("political leaning score")
    ax_hist2d.set_ylabel("newsguard score")
    ax_hist2d.set_axisbelow(True)
    ax_hist2d.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax_hist2d.set_ylim(0, 100)

    # Add reference lines and category labels
    ax_hist2d.axvline(0, **REFERENCE_LINE_STYLE, zorder=-1)

    for score, label in NEWSGUARD_CATEGORIES:
        ax_hist2d.axhline(score, **REFERENCE_LINE_STYLE, zorder=-1)
        ax_hist2d.text(1.0, score + 1, label, fontsize=9, ha="right", va="bottom")

    # Add bottom category label
    ax_hist2d.text(
        1.0, 1, "Proceed with\nMaximum Caution", fontsize=9, ha="right", va="bottom"
    )

    return ax_hist2d, h


def create_right_kde_panel(
    fig,
    gs,
    source_df,
    category_percentages,
    excluded_pf_df=None,
    excluded_percentages=None,
):
    """Create the right KDE panel for NewsGuard scores."""
    ax_marg_ng = fig.add_subplot(gs[1, 1])

    if excluded_pf_df is not None:
        # Dual marginal mode - plot both distributions
        sns.kdeplot(
            data=source_df,
            y="newsguard_score",
            fill=False,
            ax=ax_marg_ng,
            color=ALL_DATA_COLOR,
            linewidth=2,
            label="All data",
        )
        sns.kdeplot(
            data=excluded_pf_df,
            y="newsguard_score",
            fill=False,
            ax=ax_marg_ng,
            color=EXCLUDED_PF_COLOR,
            linewidth=2,
            linestyle="--",
            label="Excluding PolitiFact",
        )
    else:
        # Single distribution mode
        sns.kdeplot(
            data=source_df,
            y="newsguard_score",
            fill=True,
            ax=ax_marg_ng,
            color=KDE_COLOR,
            linewidth=2,
        )

    # Styling
    setup_axes_style(
        ax_marg_ng, remove_spines=["top", "bottom", "right"], remove_ticks=["x", "y"]
    )
    ax_marg_ng.set_xlabel(None)
    ax_marg_ng.set_ylabel(None)
    ax_marg_ng.set_ylim(0, 100)
    ax_marg_ng.set_xticklabels([])
    ax_marg_ng.set_yticklabels([])

    # Add category lines
    for score, _ in NEWSGUARD_CATEGORIES:
        ax_marg_ng.axhline(score, **REFERENCE_LINE_STYLE)

    # Add percentage annotations
    x_center = 0.1
    positions = [76, 61, 41, 1]

    if excluded_pf_df is not None:
        # Dual mode - show both percentages
        for pos, name in zip(positions, NEWSGUARD_CATEGORY_NAMES):
            all_pct = category_percentages.get(name, 0.0)
            excl_pct = excluded_percentages.get(name, 0.0)
            ax_marg_ng.text(
                x_center,
                pos,
                f"{all_pct}% ({excl_pct}%)",
                fontsize=9,
                ha="left",
                va="bottom",
            )
    else:
        # Single mode - original behavior
        for pos, name in zip(positions, NEWSGUARD_CATEGORY_NAMES):
            ax_marg_ng.text(
                x_center,
                pos,
                f"{category_percentages[name]}%",
                fontsize=9,
                ha="left",
                va="bottom",
            )

    return ax_marg_ng


def add_colorbar(fig, ax_hist2d, histogram_data):
    """Add colorbar below the main histogram."""
    hist_pos = ax_hist2d.get_position()

    cbar_ax = fig.add_axes(
        [
            hist_pos.x0 + hist_pos.width * 0.24,  # x position
            hist_pos.y0 - 0.12,  # y position
            hist_pos.width * 0.6,  # width
            0.015,  # height
        ]
    )

    cbar = fig.colorbar(histogram_data[3], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("count", fontsize=9, loc="center")
    cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    cbar.ax.tick_params(labelsize=8, rotation=0)

    return cbar


def create_2d_histogram_plot(
    df_domain_info, exclude_politifact=False, dual_marginals=False
):
    """Create the 2D histogram plot with marginal KDE distributions."""
    # Prepare data
    if dual_marginals:
        all_data_df, excluded_pf_df, all_percentages, excluded_percentages = (
            prepare_data(df_domain_info, dual_marginals=True)
        )
        # Use all data for the main histogram
        main_df = all_data_df
        main_percentages = all_percentages
    else:
        main_df, main_percentages = prepare_data(df_domain_info, exclude_politifact)
        excluded_pf_df = None
        excluded_percentages = None

    # Create fresh figure and clear any existing plots
    plt.clf()
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(
        nrows=2, ncols=2, width_ratios=[0.9, 0.1], height_ratios=[0.1, 0.9]
    )

    # Create panels
    ax_kde_top = create_top_kde_panel(fig, gs, main_df, excluded_pf_df)
    ax_hist2d, h = create_main_histogram(fig, gs, main_df)
    ax_kde_right = create_right_kde_panel(
        fig, gs, main_df, main_percentages, excluded_pf_df, excluded_percentages
    )

    # Add legend in top-right corner if dual marginals mode
    if dual_marginals:
        # Get legend from one of the KDE axes and place it on the figure
        handles, labels = ax_kde_top.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9
        )

    # Add colorbar and finalize layout
    add_colorbar(fig, ax_hist2d, h)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig


def main():
    """Main function to generate the 2D histogram figure."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate 2D histogram with marginal KDE plots showing political leaning vs NewsGuard score."
    )

    # Create mutually exclusive group for the two modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--exclude-politifact",
        action="store_true",
        help="Exclude politifact.com from the analysis",
    )
    mode_group.add_argument(
        "--dual-marginals",
        action="store_true",
        help="Show dual marginal distributions: all data vs. excluding PolitiFact",
    )

    args = parser.parse_args()

    # Load data
    df_domain_info = load_data()

    # Create 2D histogram plot
    fig = create_2d_histogram_plot(
        df_domain_info,
        exclude_politifact=args.exclude_politifact,
        dual_marginals=args.dual_marginals,
    )

    # Determine output path based on arguments
    if args.exclude_politifact:
        pdf_path = OUTPUT_PATH.replace(".pdf", "_exclude_politifact.pdf")
        png_path = OUTPUT_PATH.replace(".pdf", "_exclude_politifact.png")
    elif args.dual_marginals:
        pdf_path = OUTPUT_PATH.replace(".pdf", "_dual_marginals.pdf")
        png_path = OUTPUT_PATH.replace(".pdf", "_dual_marginals.png")
    else:
        pdf_path = OUTPUT_PATH
        png_path = OUTPUT_PATH.replace(".pdf", ".png")

    # Save figure in both formats
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    print("Figure saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
