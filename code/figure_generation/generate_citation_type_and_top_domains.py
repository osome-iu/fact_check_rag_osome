"""
Generate comprehensive citation analysis figure combining multiple visualizations.

This script creates a combined figure with multiple panels showing:
1. Top panel (full width): Point plot of citations by domain type
2. Bottom-left panel: Scatter plot of top citation domains for GPT-4o mini Search
3. Bottom-right panel: Scatter plot of top citation domains for GPT-4o Search

Purpose:
    Provides a comprehensive visualization of citation patterns and domain analysis
    for fact-checking LLM responses.

Input:
    - data/cleaned/enriched_web_urls.parquet: Enriched domain information

Output:
    - figures/generate_citation_type_and_top_domains.png: Combined visualization figure

Author: Matthew DeVerna
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Set rc param fontsize to 10
plt.rcParams["font.size"] = 10

# Change to script directory to ensure relative paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Constants
DATA_PATH = "../../data/cleaned/enriched_web_urls.parquet"
OUTPUT_PATH = "../../figures/citation_type_and_top_domains.pdf"

# Configuration mappings
MODEL_NAME_MAP = {
    "gpt-4o-mini-search-preview-2025-03-11": "GPT-4o mini Search",
    "gpt-4o-search-preview-2025-03-11": "GPT-4o Search",
}

MODEL_COLOR_MAP = {
    "GPT-4o mini Search": "#E31B23",
    "GPT-4o Search": "#262600",
}


def load_data():
    """Load and return the required dataset."""
    df_domain_info = pd.read_parquet(DATA_PATH)
    return df_domain_info


def create_domain_type_plot(ax, df_domain_info):
    """Create the point plot showing citations by domain type."""
    # Prepare domain type data
    df_domain_info["clean_model"] = df_domain_info["model"].map(MODEL_NAME_MAP)
    df_domain_type_counts = (
        df_domain_info.groupby(["model", "num_rag_items"], observed=True)["domain_type"]
        .value_counts()
        .to_frame("domain_count")
        .reset_index()
    )
    df_domain_type_counts["clean_model"] = df_domain_type_counts["model"].map(
        MODEL_NAME_MAP
    )
    df_domain_type_counts.domain_type = df_domain_type_counts.domain_type.str.replace(
        "fact_checking", "fact checking"
    )

    sns.pointplot(
        data=df_domain_type_counts,
        x="domain_count",
        y="domain_type",
        hue="clean_model",
        palette=MODEL_COLOR_MAP,
        linestyles="none",
        markers=[">", "<"],  # Use different markers for each model
        markersize=4,
        ax=ax,
        legend=True,
        alpha=0.75,
        dodge=0.3,
        errorbar=("ci", 95),  # You can specify errorbar type here (default is 95% CI)
        capsize=0.15,  # Controls the length of the error bar caps
        err_kws={"linewidth": 0.75},  # Use err_kws instead of errwidth
    )

    # Configure axes
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,d}"))
    ax.set_xlabel("number of citations")
    ax.set_ylabel("domain type")
    ax.grid(axis="both", linestyle="dotted", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set tick label font sizes
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Configure legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        title="",
        fontsize=8,
        bbox_to_anchor=(0.5, 1.2),
        loc="upper center",
        ncols=2,
        frameon=True,
    )


def create_top_domains_plots(ax2, ax3, df_domain_info):
    """Create scatter plots for top domains by model."""
    # Get top domains for each model
    domain_counts_list = []

    for ids, temp_df in df_domain_info.groupby(
        ["model", "num_rag_items"], observed=True
    ):
        model, num_rag_items = ids[0], ids[1]
        top10 = temp_df["domain"].value_counts().head(10)
        domain_counts_list.append(
            pd.DataFrame(
                {
                    "model": model,
                    "num_rag_items": num_rag_items,
                    "domain": top10.index,
                    "domain_count": top10.values,
                }
            )
        )

    df_top_domains = pd.concat(domain_counts_list).reset_index(drop=True)
    df_top_domains["clean_model"] = df_top_domains["model"].map(MODEL_NAME_MAP)
    df_top_domains = df_top_domains[
        ["clean_model", "num_rag_items", "domain", "domain_count"]
    ].copy()

    df_top_domains.num_rag_items = df_top_domains.num_rag_items.astype(int)

    gpt4o_top = df_top_domains.loc[
        (df_top_domains["clean_model"] == "GPT-4o Search")
    ].copy()
    gpt4o_mini_top = df_top_domains.loc[
        (df_top_domains["clean_model"] == "GPT-4o mini Search")
    ].copy()

    for ax, df_top, title in zip(
        [ax2, ax3], [gpt4o_mini_top, gpt4o_top], ["GPT-4o mini Search", "GPT-4o Search"]
    ):

        sns.scatterplot(
            data=df_top,
            x="domain_count",
            y="domain",
            style="num_rag_items",
            markers=["o", "D"],
            color=MODEL_COLOR_MAP[title],
            s=50,
            ax=ax,
            alpha=0.8,
            legend=True if title == "GPT-4o Search" else False,
        )

        # Configure axes
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,d}"))
        ax.set_xlabel("number of citations")
        ax.set_title(title, fontsize=10)

        # add x-axis grid lines
        ax.grid(axis="both", linestyle="dotted", alpha=0.7)
        ax.set_axisbelow(True)

        # Remove y-axis label
        ax.set_ylabel(None)

        # Set tick label font sizes
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Clean up spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Standard axis positioning for right panel

    handles, labels = ax3.get_legend_handles_labels()
    # Extract markers from the original handles and create grey versions
    grey_handles = [
        Line2D(
            [0],
            [0],
            marker=handle.get_marker(),
            color="grey",
            linestyle="None",
            markersize=6,
        )
        for handle in handles
    ]

    ax3.legend(
        handles=grey_handles,
        labels=labels,
        title=r"$k$",
        fontsize=8,
        ncols=2,
        loc="lower right",
        frameon=True,
    )


def main():
    """Main function to generate the combined citation analysis figure."""
    # Load data
    df_domain_info = load_data()

    # Create figure with gridspec
    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.35, 0.65],
    )

    # Create subplots
    ax1 = fig.add_subplot(gs[0, :])  # Top panel: Domain type (spans full width)
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left: GPT-4o mini domains
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom-right: GPT-4o domains

    # Create individual plots
    create_domain_type_plot(ax1, df_domain_info)
    create_top_domains_plots(ax2, ax3, df_domain_info)

    # Add panel markers
    ax1.text(-0.26, 1.15, "(a)", transform=ax1.transAxes, fontsize=10)
    ax2.text(-0.7, 1.1, "(b)", transform=ax2.transAxes, fontsize=10)
    ax3.text(-0.6, 1.1, "(c)", transform=ax3.transAxes, fontsize=10)

    # Adjust layout and save
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.7, hspace=0.45)

    # Define output paths
    pdf_path = OUTPUT_PATH
    png_path = OUTPUT_PATH.replace(".pdf", ".png")

    # Save in both formats
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    print("Figure saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
