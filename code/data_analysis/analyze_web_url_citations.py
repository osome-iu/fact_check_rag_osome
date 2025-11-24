"""
Purpose:
    Calculate proportions of fact-checking responses that contain web URLs and those where URLs
    match the original fact-check link. Analysis is performed at the factcheck_analysis_link level
    for each model and num_rag_items combination.

Input:
    - fc_web_sources.parquet: A Parquet file containing web URL citations from fact-checking results.
      Expected columns: factcheck_analysis_link, model, web_url, url_title, num_rag_items,
      used_web_search, filename

Output:
    - web_url_citation_analysis.parquet: A Parquet file containing the analysis results with columns:
        - model: The name of the model used
        - num_rag_items: The number of RAG items used (if applicable)
        - total_factcheck_links: Total number of unique factcheck_analysis_links for this combination
        - links_with_any_web_url: Number of factcheck_analysis_links that have at least one web_url
        - links_with_exact_match: Number of factcheck_analysis_links where at least one web_url matches exactly
        - prop_any_web_url: Proportion of factcheck_analysis_links with any web_url
        - prop_exact_match: Proportion of factcheck_analysis_links with exact URL match

Author: Matthew DeVerna
"""

import os
import pandas as pd

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

CLEAN_DATA_DIR = "../../data/cleaned/"
WEB_SOURCES_PATH = os.path.join(CLEAN_DATA_DIR, "fc_web_sources.parquet")


def analyze_url_citations(df):
    """
    Analyze web URL citations for each model and num_rag_items combination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing web URL citation data

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with analysis results for each model/num_rag_items combination
    """
    # Clean web URLs by removing utm_source parameter and handling empty strings
    df = df.copy()
    df["web_url"] = (
        df["web_url"].str.replace("?utm_source=openai", "").replace("", None)
    )

    # Use the approach from the provided inspiration code
    link_records = []
    for ids, temp_df in df.groupby(
        ["model", "factcheck_analysis_link", "num_rag_items"]
    ):
        model, fc_link, k = ids
        link_records.append(
            {
                "model": model,
                "factcheck_analysis_link": fc_link,
                "num_rag_items": k,
                "web_url_returned": any(
                    isinstance(url, str) for url in temp_df["web_url"].to_list()
                ),
                "exact_match_found": any(
                    url == fc_link for url in temp_df["web_url"].to_list()
                ),
            }
        )

    link_analysis = pd.DataFrame.from_records(link_records)

    # Calculate summary statistics for each model/num_rag_items combination
    results = (
        link_analysis.groupby(["model", "num_rag_items"])
        .agg(
            {
                "factcheck_analysis_link": "count",
                "web_url_returned": "sum",
                "exact_match_found": "sum",
            }
        )
        .reset_index()
    )

    # Rename and calculate proportions
    results = results.rename(
        columns={
            "factcheck_analysis_link": "total_factcheck_links",
            "web_url_returned": "links_with_any_web_url",
            "exact_match_found": "links_with_exact_match",
        }
    )

    results["prop_any_web_url"] = (
        results["links_with_any_web_url"] / results["total_factcheck_links"]
    )
    results["prop_exact_match"] = (
        results["links_with_exact_match"] / results["total_factcheck_links"]
    )

    return results


if __name__ == "__main__":

    print("Loading web sources data...")
    df = pd.read_parquet(WEB_SOURCES_PATH)

    print(f"Loaded {len(df)} records")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique num_rag_items: {sorted(df['num_rag_items'].unique())}")

    print("\nAnalyzing web URL citations...")
    results_df = analyze_url_citations(df)

    print(f"\nAnalysis complete.")

    # Save results
    output_path = os.path.join(CLEAN_DATA_DIR, "web_url_citation_analysis.parquet")
    print(f"\nSaving results to: {output_path}")
    results_df.to_parquet(output_path, index=False)

    print("Script complete.")
