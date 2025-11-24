"""
Purpose:
    Generate a comprehensive LaTeX table showing relative performance increases when
    using Retrieval-Augmented Generation (RAG) compared to baseline (k=0) configurations.
    Calculates percentage improvements in macro F1-score performance for models featured
    in figure generation scripts for research analysis.

    Covers two main comparison scenarios:
    1. Base model performance (k=3,6,9 vs k=0) from Figure 1 generation script
    2. Reasoning/web search model performance (k=6 vs k=0) from Figure 2 generation script

    Models are organized by family (DeepSeek, Llama, Gemini, GPT) with standard
    configurations listed before reasoning and web search variants.

Inputs:
    Reads cleaned classification report data from:
    - multi_class_all_summary_0.25.parquet: Overall performance metrics (~6k claims)

    Processes models from two figure generation scenarios:
    - Figure 1: 9 base models with k=3,6,9 comparisons (27 comparisons)
    - Figure 2: 8 reasoning/search models with k=6 comparisons (8 comparisons)

Output:
    Single LaTeX table file:
    - relative_increase_table.txt: Comprehensive table showing baseline F1 scores,
      RAG F1 scores, and percentage increases for all 35 model comparisons

    Table includes model type indicators (web search, reasoning capabilities) and
    maintains consistent ordering by model family and configuration type.

Author: Matthew DeVerna
"""

import os
import pandas as pd
from pathlib import Path

# Change working directory to script location
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Data paths (relative to script location)
MULTI_SUMMARY_PATH = Path(
    "../../data/cleaned/classification_reports/multi_class_all_summary_0.25.parquet"
)
OUTPUT_DIR = Path("../../tables")

# Configuration from figure generation scripts
OUTDATED_MODEL = "gpt-4o-2024-08-06"

# Figure 1 models (base model performance comparison)
FIGURE1_MODELS = [
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

# Figure 2 models (reasoning and search)
FIGURE2_REASONING_MODELS = [
    "DeepSeek-R1",
    "gemini-2.0-flash-thinking-exp-01-21",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
]

FIGURE2_SEARCH_MODELS = [
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-search-preview-2025-03-11",
]

# Model name mapping for display (consistent with other scripts)
MODEL_NAME_MAP = {
    "DeepSeek-R1": "DeepSeek-R1",
    "DeepSeek-V3": "DeepSeek-V3",
    "Llama-3.2-3B-Instruct-Turbo": "Llama 3.2 3B",
    "Llama-3.2-11B-Vision-Instruct-Turbo": "Llama 3.2 11B",
    "Llama-3.2-90B-Vision-Instruct-Turbo": "Llama 3.2 90B",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
    "gemini-2.0-pro-exp-02-05": "Gemini 2.0 Pro",
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-4o-search-preview-2025-03-11": "GPT-4o Search",
    "gpt-4o-mini-search-preview-2025-03-11": "GPT-4o mini Search",
    "o1-2024-12-17": "o1",
    "o3-mini-2025-01-31": "o3-mini",
}


def load_and_prepare_data():
    """Load and prepare performance data for analysis."""
    df = pd.read_parquet(MULTI_SUMMARY_PATH)

    # Filter outdated models
    df = df[df.model != OUTDATED_MODEL].reset_index(drop=True)

    return df


def get_baseline_performance(df, model, web_search):
    """Get baseline (k=0) F1 performance for a model."""
    baseline_data = df[
        (df["model"] == model)
        & (df["num_rag_items"] == 0.0)
        & (df["used_web_search"] == web_search)
    ]

    if baseline_data.empty:
        return None

    return baseline_data["f1-score-macro-avg"].iloc[0]


def get_rag_performance(df, model, k_value, web_search):
    """Get RAG (k > 0) F1 performance for a model."""
    rag_data = df[
        (df["model"] == model)
        & (df["num_rag_items"] == float(k_value))
        & (df["used_web_search"] == web_search)
    ]

    if rag_data.empty:
        return None

    return rag_data["f1-score-macro-avg"].iloc[0]


def calculate_relative_increase(baseline, rag_value):
    """Calculate percentage increase from baseline to RAG."""
    if baseline is None or rag_value is None or baseline == 0:
        return None

    return ((rag_value - baseline) / baseline) * 100


def generate_figure1_comparisons(df):
    """Generate comparisons for Figure 1 models (k=3,6,9 vs k=0)."""
    comparisons = []

    for model in FIGURE1_MODELS:
        baseline_f1 = get_baseline_performance(df, model, web_search=False)

        if baseline_f1 is None:
            continue

        # Compare k=3, k=6, k=9 against k=0
        for k_value in [3, 6, 9]:
            rag_f1 = get_rag_performance(df, model, k_value, web_search=False)

            if rag_f1 is not None:
                rel_increase = calculate_relative_increase(baseline_f1, rag_f1)

                comparisons.append(
                    {
                        "model": model,
                        "display_name": MODEL_NAME_MAP.get(model, model),
                        "used_web_search": False,
                        "is_reasoning": False,
                        "k_baseline": 0,
                        "k_rag": k_value,
                        "f1_baseline": baseline_f1,
                        "f1_rag": rag_f1,
                        "relative_increase": rel_increase,
                        "source_figure": "Figure 1",
                    }
                )

    return comparisons


def generate_figure2_comparisons(df):
    """Generate comparisons for Figure 2 models (k=6 vs k=0)."""
    comparisons = []

    # Reasoning models (no web search)
    for model in FIGURE2_REASONING_MODELS:
        baseline_f1 = get_baseline_performance(df, model, web_search=False)
        rag_f1 = get_rag_performance(df, model, 6, web_search=False)

        if baseline_f1 is not None and rag_f1 is not None:
            rel_increase = calculate_relative_increase(baseline_f1, rag_f1)

            comparisons.append(
                {
                    "model": model,
                    "display_name": MODEL_NAME_MAP.get(model, model),
                    "used_web_search": False,
                    "is_reasoning": True,
                    "k_baseline": 0,
                    "k_rag": 6,
                    "f1_baseline": baseline_f1,
                    "f1_rag": rag_f1,
                    "relative_increase": rel_increase,
                    "source_figure": "Figure 2",
                }
            )

    # Search models (with web search)
    for model in FIGURE2_SEARCH_MODELS:
        baseline_f1 = get_baseline_performance(df, model, web_search=True)
        rag_f1 = get_rag_performance(df, model, 6, web_search=True)

        if baseline_f1 is not None and rag_f1 is not None:
            rel_increase = calculate_relative_increase(baseline_f1, rag_f1)

            # Check if this is a reasoning model
            is_reasoning = model in FIGURE2_REASONING_MODELS

            comparisons.append(
                {
                    "model": model,
                    "display_name": MODEL_NAME_MAP.get(model, model),
                    "used_web_search": True,
                    "is_reasoning": is_reasoning,
                    "k_baseline": 0,
                    "k_rag": 6,
                    "f1_baseline": baseline_f1,
                    "f1_rag": rag_f1,
                    "relative_increase": rel_increase,
                    "source_figure": "Figure 2",
                }
            )

    return comparisons


def get_model_type_priority(used_web_search, is_reasoning):
    """Determine model type priority for sorting: standard -> reasoning -> web search."""
    if not used_web_search and not is_reasoning:
        return 1  # Standard models
    elif not used_web_search and is_reasoning:
        return 2  # Reasoning models (no web search)
    elif used_web_search:
        return 3  # Web search models (may also be reasoning)
    else:
        return 4  # Fallback


def sort_comparisons_by_family_and_type(comparisons_df):
    """Sort comparisons by specified model order and type (standard -> reasoning -> web)."""
    # Define the exact model order as requested
    model_order = [
        "DeepSeek-V3",
        "DeepSeek-R1",
        "Llama 3.2 3B",
        "Llama 3.2 11B",
        "Llama 3.2 90B",
        "Gemini 2.0 Flash Lite",
        "Gemini 2.0 Flash",
        "Gemini 2.0 Flash Thinking",
        "Gemini 2.0 Pro",
        "GPT-4o mini",
        "GPT-4o",
        "GPT-4o Search",
        "GPT-4o mini Search",
        "o1",
        "o3-mini",
    ]

    # Create ordered categorical for model names
    comparisons_df["model_order"] = pd.Categorical(
        comparisons_df["display_name"], categories=model_order, ordered=True
    )

    # Add type priority: standard -> reasoning -> web search
    comparisons_df["type_priority"] = comparisons_df.apply(
        lambda row: get_model_type_priority(row["used_web_search"], row["is_reasoning"]),
        axis=1,
    )

    # Sort by: model order -> type priority -> k value
    sorted_df = comparisons_df.sort_values(
        ["model_order", "type_priority", "k_rag"]
    ).reset_index(drop=True)

    # Remove the sorting helper columns
    sorted_df = sorted_df.drop(columns=["model_order", "type_priority"])

    return sorted_df


def format_table_for_latex(comparisons_df):
    """Format the comparisons dataframe for LaTeX output."""
    # Create formatted dataframe
    formatted_df = comparisons_df.copy()

    # Format web search and reasoning columns
    formatted_df["used_web_search"] = formatted_df["used_web_search"].replace(
        {True: r"\checkmark", False: ""}
    )
    formatted_df["is_reasoning"] = formatted_df["is_reasoning"].replace(
        {True: r"\checkmark", False: ""}
    )

    # Round numerical values
    formatted_df["f1_baseline"] = formatted_df["f1_baseline"].round(2)
    formatted_df["f1_rag"] = formatted_df["f1_rag"].round(2)
    formatted_df["relative_increase"] = formatted_df["relative_increase"].round(0)

    # Select and rename columns for output
    output_df = formatted_df[
        [
            "display_name",
            "used_web_search",
            "is_reasoning",
            "k_rag",
            "f1_baseline",
            "f1_rag",
            "relative_increase",
        ]
    ].rename(
        columns={
            "display_name": "Model",
            "used_web_search": r"\faSearch",
            "is_reasoning": r"\faBrain",
            "k_rag": r"$k$",
            "f1_baseline": "F1 (macro) Baseline",
            "f1_rag": "F1 (macro) RAG",
            "relative_increase": r"Increase (\%)",
        }
    )

    # Add summary row with mean (std) of relative increase
    mean_increase = comparisons_df["relative_increase"].mean()
    std_increase = comparisons_df["relative_increase"].std()

    summary_row = pd.DataFrame({
        "Model": ["Mean (SD)"],
        r"\faSearch": [pd.NA],
        r"\faBrain": [pd.NA],
        r"$k$": [pd.NA],
        "F1 (macro) Baseline": [pd.NA],
        "F1 (macro) RAG": [pd.NA],
        r"Increase (\%)": [f"{mean_increase:.1f} ({std_increase:.1f})"]
    })

    output_df = pd.concat([output_df, summary_row], ignore_index=True)

    return output_df


def main():
    """Generate relative increase table for RAG vs baseline performance."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading classification report data...")
    df = load_and_prepare_data()

    # Generate comparisons from both figures
    print("Generating Figure 1 comparisons...")
    figure1_comparisons = generate_figure1_comparisons(df)

    print("Generating Figure 2 comparisons...")
    figure2_comparisons = generate_figure2_comparisons(df)

    # Combine all comparisons
    all_comparisons = figure1_comparisons + figure2_comparisons

    if not all_comparisons:
        print("No valid comparisons found!")
        return

    # Convert to DataFrame
    comparisons_df = pd.DataFrame(all_comparisons)

    # Sort by model family and type for better organization
    comparisons_df = sort_comparisons_by_family_and_type(comparisons_df)

    # Format for LaTeX
    print("Formatting table for LaTeX...")
    latex_df = format_table_for_latex(comparisons_df)

    # Generate LaTeX table
    output_path = OUTPUT_DIR / "relative_increase_table.txt"

    # Format float columns individually since they need different precision
    latex_df_formatted = latex_df.copy()

    # Format F1 columns to 2 decimal places (skip if not numeric)
    latex_df_formatted["F1 (macro) Baseline"] = latex_df_formatted["F1 (macro) Baseline"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and pd.notna(x) else x
    )
    latex_df_formatted["F1 (macro) RAG"] = latex_df_formatted["F1 (macro) RAG"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and pd.notna(x) else x
    )

    # Format increase column to whole numbers (skip if not numeric or already a string)
    latex_df_formatted[r"Increase (\%)"] = latex_df_formatted[r"Increase (\%)"].apply(
        lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and pd.notna(x) else x
    )

    with open(output_path, "w") as f:
        f.write(
            latex_df_formatted.to_latex(
                column_format="llcc" + "r" * 4,
                index=False,
                escape=False,
            )
        )

    print(f"Relative increase table saved to {output_path}")
    print(f"Generated {len(all_comparisons)} comparisons")
    print(f"  - Figure 1: {len(figure1_comparisons)} comparisons")
    print(f"  - Figure 2: {len(figure2_comparisons)} comparisons")

    # Display summary statistics
    print("\nSummary statistics:")
    print(
        f"  - Mean relative increase: {comparisons_df['relative_increase'].mean():.1f}%"
    )
    print(
        f"  - Min relative increase: {comparisons_df['relative_increase'].min():.1f}%"
    )
    print(
        f"  - Max relative increase: {comparisons_df['relative_increase'].max():.1f}%"
    )
    print(f"  - Positive increases: {(comparisons_df['relative_increase'] > 0).sum()}")
    print(f"  - Negative increases: {(comparisons_df['relative_increase'] < 0).sum()}")


if __name__ == "__main__":
    main()
