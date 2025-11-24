"""
Purpose:
    Generate LaTeX citation statistics table for search-enabled LLM fact-checks.
    Creates a table showing how often models cite web URLs and specifically cite
    the exact PolitiFact article URL in their fact-check responses.

    Citation statistics help understand model behavior regarding source attribution
    and whether models are referencing the original fact-check article or finding
    independent sources.

Inputs:
    Reads citation analysis data from:
    - web_url_citation_analysis.parquet: Citation counts and proportions by model
      and RAG configuration (k=0 vs k=6)

Output:
    Single LaTeX table file:
    - citation_statistics_table.txt: Table with columns showing model name,
      Curated RAG setting (k), count and percentage of fact-checks containing
      any URL, and count and percentage with exact match to PolitiFact article

    Format: "count (percentage)" - e.g., "3,861 (63)" for readability

Author: Matthew DeVerna
"""

import os
import pandas as pd
from pathlib import Path

# Change working directory to script location
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Data paths (relative to script location)
CITATION_DATA_PATH = Path("../../data/cleaned/web_url_citation_analysis.parquet")
OUTPUT_DIR = Path("../../tables")


def main():
    """Generate citation statistics table for search-enabled LLM fact-checks."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading citation analysis data...")
    df = pd.read_parquet(CITATION_DATA_PATH)

    # Model name mapping for display (consistent with other table scripts)
    models_name_map = {
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
        "gpt-4o-mini-search-preview-2025-03-11": "GPT-4o mini Search",
        "gpt-4o-search-preview-2025-03-11": "GPT-4o Search",
    }

    # Apply model name mapping
    df["model_display"] = df["model"].map(models_name_map)

    # Define model order (consistent with other scripts)
    ordered_models = [
        "Gemini 2.0 Flash",
        "Gemini 2.0 Flash Thinking",
        "GPT-4o mini Search",
        "GPT-4o Search",
    ]

    # Create ordered categorical and sort
    df["model_ordered"] = pd.Categorical(
        df["model_display"], categories=ordered_models, ordered=True
    )
    df = df.sort_values(by=["model_ordered", "num_rag_items"]).reset_index(drop=True)

    # Format k values as integers
    df["k"] = df["num_rag_items"].astype(int)

    # Calculate percentages (convert proportions to percentages)
    df["pct_any_url"] = (df["prop_any_web_url"] * 100).round(0).astype(int)
    df["pct_exact_match"] = (df["prop_exact_match"] * 100).round(0).astype(int)

    # Format combined count (percentage) columns
    df["any_url_formatted"] = df.apply(
        lambda row: f"{row['links_with_any_web_url']:,} ({row['pct_any_url']})",
        axis=1,
    )
    df["exact_match_formatted"] = df.apply(
        lambda row: f"{row['links_with_exact_match']:,} ({row['pct_exact_match']})",
        axis=1,
    )

    # Create output dataframe with bold headers
    output_df = df[
        ["model_display", "k", "any_url_formatted", "exact_match_formatted"]
    ].rename(
        columns={
            "model_display": r"\textbf{Model}",
            "k": r"\textbf{$k$}",
            "any_url_formatted": r"\textbf{Any URL (\%)}",
            "exact_match_formatted": r"\textbf{Exact Match (\%)}",
        }
    )

    # Generate LaTeX table
    output_path = OUTPUT_DIR / "citation_statistics_table.txt"
    print("Generating LaTeX table...")

    with open(output_path, "w") as f:
        f.write(
            output_df.to_latex(
                column_format="lrrr",
                index=False,
                escape=False,
            )
        )

    print(f"Citation statistics table saved to {output_path}")
    print(f"Generated table with {len(output_df)} rows")

    # Display summary
    print("\nSummary:")
    print(f"  Models: {len(df['model_display'].unique())}")
    print(f"  RAG settings per model: {len(df) // len(df['model_display'].unique())}")


if __name__ == "__main__":
    main()
