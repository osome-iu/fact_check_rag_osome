"""Generate binary performance tables from fact-checking evaluation data.

Reads binary classification report data and generates LaTeX tables for research analysis:
- Best vs worst performance comparison by label
- Binary summary across all model configurations

Part of the fact-checking LLM evaluation pipeline.
"""

import os
import pandas as pd
from pathlib import Path

# Change working directory to script location
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Data paths (relative to script location)
BINARY_SUMMARY_PATH = Path(
    "../../data/cleaned/classification_reports/binary_class_all_summary_0.25.parquet"
)
BINARY_BY_LABEL_PATH = Path(
    "../../data/cleaned/classification_reports/binary_class_all_by_label_0.25.parquet"
)
OUTPUT_DIR = Path("../../tables")

# Configuration
OUTDATED_MODEL = "gpt-4o-2024-08-06"
PROPER_SUPPORT_VALUE = 6137


def main():
    """Generate binary performance tables for fact-checking evaluation results."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading binary classification report data...")
    binary_summary_df = pd.read_parquet(BINARY_SUMMARY_PATH)
    binary_by_label_df = pd.read_parquet(BINARY_BY_LABEL_PATH)

    # Filter outdated models
    binary_summary_df = binary_summary_df[
        binary_summary_df.model != OUTDATED_MODEL
    ].reset_index(drop=True)

    # Model name mapping for display
    models_name_map = {
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

    # Generate Table 1: Best vs Worst Performance by Label
    print("Generating best vs worst binary performance table...")
    generate_best_worst_table(binary_summary_df, binary_by_label_df, models_name_map)

    # Generate Table 2: Binary Summary
    print("Generating binary summary table...")
    generate_binary_summary_table(binary_summary_df, models_name_map)

    print(f"Tables saved to: {OUTPUT_DIR}")


def generate_best_worst_table(binary_summary_df, binary_by_label_df, models_name_map):
    """Generate table comparing best and worst performing model configurations by label."""

    # Find best and worst performing configurations
    best_model_setting_index = binary_summary_df.sort_values(
        by="f1-score-macro-avg", ascending=False
    ).index[0]
    worst_model_setting_index = binary_summary_df.sort_values(
        by="f1-score-macro-avg", ascending=True
    ).index[0]

    best_setting = binary_summary_df.loc[best_model_setting_index].to_dict()
    worst_setting = binary_summary_df.loc[worst_model_setting_index].to_dict()

    # Extract data for best and worst settings
    best_setting_df = binary_by_label_df[
        (binary_by_label_df["model"] == best_setting["model"])
        & (binary_by_label_df["num_rag_items"] == best_setting["num_rag_items"])
        & (binary_by_label_df["used_web_search"] == best_setting["used_web_search"])
    ]

    worst_setting_df = binary_by_label_df[
        (binary_by_label_df["model"] == worst_setting["model"])
        & (binary_by_label_df["num_rag_items"] == worst_setting["num_rag_items"])
        & (binary_by_label_df["used_web_search"] == worst_setting["used_web_search"])
    ]

    # Combine and process data
    selected_models = pd.concat([best_setting_df, worst_setting_df]).reset_index(
        drop=True
    )
    selected_models["model"] = selected_models["model"].map(models_name_map)

    # Validate support values
    support_values = list(selected_models.groupby("model")["support"].sum())
    assert all(
        support == PROPER_SUPPORT_VALUE for support in support_values
    ), "Support values do not match expected total"

    # Remove "Not enough information" label
    selected_models = selected_models[
        selected_models["label"] != "Not enough information"
    ].reset_index(drop=True)

    # Create ordered categories
    ordered_models = ["GPT-4o Search", "Llama 3.2 3B"]
    selected_models["model"] = pd.Categorical(
        selected_models["model"], categories=ordered_models, ordered=True
    )

    label_order = ["True", "False"]
    selected_models["label"] = pd.Categorical(
        selected_models["label"], categories=label_order, ordered=True
    )

    # Sort and format data
    selected_models = selected_models.sort_values(["model", "label"]).reset_index(
        drop=True
    )
    selected_models["num_rag_items"] = selected_models["num_rag_items"].astype(int)
    selected_models["used_web_search"] = selected_models["used_web_search"].replace(
        {True: r"\checkmark", False: ""}
    )

    # Generate LaTeX table
    output_path = OUTPUT_DIR / "best_worst_by_label_performance_binary.txt"
    with open(output_path, "w") as f:
        f.write(
            selected_models.rename(
                columns={
                    "model": "Model",
                    "num_rag_items": r"$k$",
                    "used_web_search": "Search",
                    "label": "Label",
                    "precision": "P",
                    "recall": "R",
                    "f1-score": "F1",
                    "support": "Sup.",
                }
            )[["Model", "Search", r"$k$", "Label", "P", "R", "F1", "Sup."]].to_latex(
                column_format="llc" + "r" * 6,
                float_format="%.2f",
                index=False,
                escape=False,
            )
        )


def generate_binary_summary_table(binary_summary_df, models_name_map):
    """Generate comprehensive summary table for all model configurations."""

    # Apply model name mapping
    binary_summary_df = binary_summary_df.copy()
    binary_summary_df["model"] = binary_summary_df["model"].map(models_name_map)

    # Define model order
    ordered_models = [
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

    # Create ordered categorical and sort
    binary_summary_df["model_ordered"] = pd.Categorical(
        binary_summary_df["model"], categories=ordered_models, ordered=True
    )
    binary_summary_df["num_rag_items"] = binary_summary_df["num_rag_items"].astype(int)
    binary_summary_df = binary_summary_df.sort_values(
        by=["model_ordered", "used_web_search", "num_rag_items"]
    )

    # Format web search column
    binary_summary_df["used_web_search"] = binary_summary_df["used_web_search"].replace(
        {True: r"\checkmark", False: ""}
    )

    # Add reasoning models column
    reasoning_models = ["DeepSeek-R1", "Gemini 2.0 Flash Thinking", "o1", "o3-mini"]
    binary_summary_df["reasoning_model"] = binary_summary_df["model"].apply(
        lambda x: r"\checkmark" if x in reasoning_models else ""
    )

    # Generate LaTeX table
    output_path = OUTPUT_DIR / "summary_df_binary.txt"
    with open(output_path, "w") as f:
        f.write(
            binary_summary_df.rename(
                columns={
                    "model": "Model",
                    "num_rag_items": r"$k$",
                    "used_web_search": r"\faSearch",
                    "reasoning_model": r"\faBrain",
                    "precision-macro-avg": "P (macro)",
                    "recall-macro-avg": "R (macro)",
                    "f1-score-macro-avg": "F1 (macro)",
                    "precision-weighted-avg": "P (wt)",
                    "recall-weighted-avg": "R (wt)",
                    "f1-score-weighted-avg": "F1 (wt)",
                    "accuracy": "Acc.",
                    "support": "Sup.",
                }
            )[
                [
                    "Model",
                    r"\faSearch",
                    r"\faBrain",
                    r"$k$",
                    "P (macro)",
                    "R (macro)",
                    "F1 (macro)",
                    "F1 (wt)",
                    "Acc.",
                    "Sup.",
                ]
            ]
            .round(2)
            .to_latex(
                column_format="llcc" + "r" * 6,
                float_format="%.2f",
                index=False,
                escape=False,
            )
        )


if __name__ == "__main__":
    main()