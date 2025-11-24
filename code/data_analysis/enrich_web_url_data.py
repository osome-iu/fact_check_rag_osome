"""
Purpose:
    Enrich web URL data from fact-checking results by adding domain-level
    information including political leaning scores, NewsGuard ratings, PC1 quality scores
    (from Lin et al. 2023), and domain type classifications. It filters to only non-null
    web URLs and creates an enriched dataset for further analysis.

    NOTE:
        The NewsGuard data file is proprietary and we are not allowed to share it in this repository.
        As a result, this script will skip the NewsGuard enrichment step and the output filename will
        reflect this exclusion.

        If you have access to NewsGuard data, you can run the script to include NewsGuard in the output
        file in the following way:
         1. Save the file in the right location with the proper filename (see NEWSGUARD_PATH variable)
         2. Ensure that the file has the columns:
            - domain
            - newsguard_rating
            - newsguard_score
         3. Re-run the script

Input:
    - fc_web_sources.parquet: Web URL citations from fact-checking results
    - domaindemo_political_leaning.csv.gz: Political leaning scores by domain
    - newsguard_2025-03.csv: NewsGuard ratings and scores by domain
    - lin10k_domain_pc1.csv: PC1 quality scores by domain from Lin et al. (2023)
        Ref: https://github.com/hauselin/domain-quality-ratings/tree/main
    - news_domains.csv: List of news-related domains

Output:
    - enriched_web_urls.parquet: Enriched dataset with columns:
        - Original columns from fc_web_sources.parquet
        - web_url_cleaned: URL with utm parameters removed
        - domain: Extracted domain name
        - leaning_score: Political leaning score (-1 to 1, left to right)
        - newsguard_rating: NewsGuard rating (T/F/N)
        - newsguard_score: NewsGuard score (0-100)
        - pc1_quality: PC1 quality score from Lin et al. (2023), 0-1 scale
        - domain_type: Classification (fact_checking, news, government, wikipedia, educational, research, other)

Author: Matthew DeVerna
"""

import os
import tldextract

import pandas as pd


from llm_fact.domain_patterns import (
    FACT_CHECK_PATTERNS,
    NEWS_PATTERNS,
    GOVERNMENT_PATTERNS,
    EDUCATIONAL_PATTERNS,
    RESEARCH_PATTERNS,
    NONPROFIT_PATTERNS,
)

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Data paths
CLEAN_DATA_DIR = "../../data/cleaned/"
INTERMEDIATE_DATA_DIR = "../../data/intermediate/"

WEB_SOURCES_PATH = os.path.join(CLEAN_DATA_DIR, "fc_web_sources.parquet")
POLITICAL_LEANING_PATH = os.path.join(
    INTERMEDIATE_DATA_DIR, "domaindemo_political_leaning.csv.gz"
)
NEWSGUARD_PATH = os.path.join(INTERMEDIATE_DATA_DIR, "newsguard_2025-03.csv")
PC1_QUALITY_PATH = os.path.join(INTERMEDIATE_DATA_DIR, "lin10k_domain_pc1.csv")
NEWS_DOMAINS_PATH = os.path.join(INTERMEDIATE_DATA_DIR, "news_domains.csv")


def clean_web_url(url):
    """
    Clean web URLs by removing utm parameters and handling empty strings.

    Parameters
    ----------
    url : str or None
        The web URL to clean

    Returns
    -------
    cleaned_url : str or None
        The cleaned URL with utm parameters removed, or None if empty
    """
    if pd.isna(url) or url == "":
        return None

    # Remove utm_source parameter and convert empty strings to None
    cleaned = (
        str(url).replace("?utm_source=openai", "").replace("&utm_source=openai", "")
    )
    return cleaned if cleaned != "" else None


def extract_domain(url):
    """Extract base domain from URL using tldextract (without subdomains)."""
    try:
        extracted = tldextract.extract(url)
        # Combine domain and suffix (e.g., 'example' + 'co.uk' = 'example.co.uk')
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        elif extracted.domain:
            return extracted.domain.lower()
        return None
    except Exception:
        return None


def classify_domain_type(domain, news_domains_set):
    """
    Classify domain type based on heuristics and news domain list.

    Parameters
    ----------
    domain : str
        The domain to classify
    news_domains_set : set
        Set of known news domains

    Returns
    -------
    domain_type : str
        One of: fact_checking, news, government, wikipedia, educational, research, other
    """

    domain_lower = domain.lower()

    # Fact-checking domains
    if any(pattern in domain_lower for pattern in FACT_CHECK_PATTERNS):
        return "fact_checking"

    # News domains (from provided list)
    if domain in news_domains_set:
        return "news"

    # Additional news patterns
    if any(pattern in domain_lower for pattern in NEWS_PATTERNS):
        # print(f"Classifying {domain} as news based on patterns")
        return "news"

    # Government domains (expanded)
    if any(pattern in domain_lower for pattern in GOVERNMENT_PATTERNS):
        return "government"

    # State/local government patterns
    if domain_lower.endswith(".us") and (
        "state" in domain_lower or "county" in domain_lower or "city" in domain_lower
    ):
        return "government"

    # Wikipedia domains
    if "wiki" in domain_lower:
        return "wikipedia"

    # Educational domains (expanded)
    if any(pattern in domain_lower for pattern in EDUCATIONAL_PATTERNS):
        return "educational"

    # Research/Think tank/Policy organizations
    if any(pattern in domain_lower for pattern in RESEARCH_PATTERNS):
        return "research"

    # Non-profit organizations (broad .org patterns)
    if any(pattern in domain_lower for pattern in NONPROFIT_PATTERNS):
        return "research"

    return "other"


def enrich_web_url_data():
    """
    Main function to enrich web URL data with domain-level information.

    Returns
    -------
    enriched_df : pd.DataFrame
        Enriched dataset with additional domain information
    newsguard_included : bool
        Whether NewsGuard data was included in the enrichment
    """
    print("Loading web sources data...")
    web_sources_df = pd.read_parquet(WEB_SOURCES_PATH)

    # Drop 'gemini-2.0-flash' model rows as these are all bad links
    print("Filtering out 'gemini-2.0-flash' model rows with bad links...")
    web_sources_df = web_sources_df[
        web_sources_df["model"] != "gemini-2.0-flash"
    ].reset_index(drop=True)

    # Filter to only non-null and non-empty web URLs
    print(f"Original data shape: {web_sources_df.shape}")
    web_sources_df = web_sources_df[
        (web_sources_df["web_url"].notna()) & (web_sources_df["web_url"] != "")
    ].copy()
    print(f"After filtering to non-null and non-empty web URLs: {web_sources_df.shape}")

    # Clean URLs and extract domains
    print("Cleaning URLs and extracting domains...")
    web_sources_df["web_url_cleaned"] = web_sources_df["web_url"].apply(clean_web_url)
    web_sources_df["domain"] = web_sources_df["web_url_cleaned"].apply(extract_domain)

    # Store initial row count for verification
    initial_row_count = len(web_sources_df)

    # Load auxiliary data
    print("Loading auxiliary data sources...")
    political_leaning_df = pd.read_csv(POLITICAL_LEANING_PATH)
    pc1_quality_df = pd.read_csv(PC1_QUALITY_PATH)
    pc1_quality_df = pc1_quality_df.rename(columns={"pc1": "pc1_quality"})
    news_domains_df = pd.read_csv(NEWS_DOMAINS_PATH)

    # Check if NewsGuard data exists
    newsguard_included = os.path.exists(NEWSGUARD_PATH)
    if newsguard_included:
        newsguard_df = pd.read_csv(NEWSGUARD_PATH)
    else:
        print(
            f"\nWARNING: NewsGuard data file not found at {NEWSGUARD_PATH}. "
            "Only Lin et al. (2023) data will be used.\n"
        )

    # Create news domains set for efficient lookup
    news_domains_set = set(news_domains_df["domain"].tolist())

    # Merge with political leaning data (left join preserves all rows)
    print("Merging with political leaning data...")
    enriched_df = web_sources_df.merge(political_leaning_df, on="domain", how="left")

    # Verify no rows were dropped
    assert (
        len(enriched_df) == initial_row_count
    ), f"Rows dropped in political leaning merge: {initial_row_count} -> {len(enriched_df)}"

    # Merge with NewsGuard data (left join preserves all rows) if available
    if newsguard_included:
        print("Merging with NewsGuard data...")
        enriched_df = enriched_df.merge(newsguard_df, on="domain", how="left")

        # Verify no rows were dropped
        assert (
            len(enriched_df) == initial_row_count
        ), f"Rows dropped in NewsGuard merge: {initial_row_count} -> {len(enriched_df)}"

    # Merge with PC1 quality data (left join preserves all rows)
    print("Merging with PC1 quality data...")
    enriched_df = enriched_df.merge(pc1_quality_df, on="domain", how="left")

    # Verify no rows were dropped
    assert (
        len(enriched_df) == initial_row_count
    ), f"Rows dropped in PC1 quality merge: {initial_row_count} -> {len(enriched_df)}"

    # Classify domain types
    print("Classifying domain types...")
    enriched_df["domain_type"] = enriched_df["domain"].apply(
        lambda x: classify_domain_type(x, news_domains_set)
    )

    # Final verification
    assert (
        len(enriched_df) == initial_row_count
    ), f"Rows dropped during enrichment: {initial_row_count} -> {len(enriched_df)}"
    print(
        f"✓ Row count verification passed: {len(enriched_df)} rows maintained throughout enrichment"
    )

    return enriched_df, newsguard_included


if __name__ == "__main__":

    print("Starting web URL data enrichment...")
    enriched_df, newsguard_included = enrich_web_url_data()

    print(f"\nEnrichment complete. Final shape: {enriched_df.shape}")
    print(f"Columns: {enriched_df.columns.tolist()}")

    # Display summary statistics
    print("\nAggregate summary statistics across models/tests:\n")
    print("\nDomain type distribution:")
    print(enriched_df["domain_type"].value_counts())

    print("\nPolitical leaning coverage:")
    print(f"Domains with leaning scores: {enriched_df['leaning_score'].notna().sum()}")
    print(f"Total domains: {len(enriched_df)}")
    print(
        f"Proportion: {enriched_df['leaning_score'].notna().mean():.2} "
        "of domains have political leaning scores"
    )

    if newsguard_included:
        print("\nNewsGuard coverage:")
        print(
            "Domains with NewsGuard ratings: "
            f"{enriched_df['newsguard_rating'].notna().sum()}"
        )
        print(f"Total domains: {len(enriched_df)}")
        print(
            f"Proportion: {enriched_df['newsguard_rating'].notna().mean():.2} "
            "of domains have NewsGuard ratings"
        )

    print("\nPC1 quality coverage:")
    print(
        "Domains with PC1 quality scores: "
        f"{enriched_df['pc1_quality'].notna().sum()}"
    )
    print(f"Total domains: {len(enriched_df)}")
    print(
        f"Proportion: {enriched_df['pc1_quality'].notna().mean():.2} "
        "of domains have PC1 quality scores"
    )

    # Save enriched data with appropriate filename based on NewsGuard inclusion
    if newsguard_included:
        output_filename = "enriched_web_urls.parquet"
    else:
        output_filename = "enriched_web_urls_newsguard_excluded.parquet"

    output_path = os.path.join(CLEAN_DATA_DIR, output_filename)
    print(f"\nSaving enriched data to: {output_path}")
    enriched_df.to_parquet(output_path, index=False)

    print("Script complete.")
