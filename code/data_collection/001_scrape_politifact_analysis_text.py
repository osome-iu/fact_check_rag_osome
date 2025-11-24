"""
Purpose:
    - Scrape the PolitiFact fact check analysis articles.

Input:
    - None
    - Data read from path set as a constant.

Output:
    - .parquet file containing the below columns:
        - factcheck_analysis_link: link to the politifact article.
        - factcheck_analysis_text: the text scraped from the article.

Author:
    - Matthew DeVerna
"""

import datetime
import os
import time
import random

import pandas as pd

from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Ensure that the current directory is the location of this script for relative paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = "../../data/raw"

# File names
FC_FILE = "2024-10-10_factchecks.parquet"
OUTPUT_FILE = "fc_analysis_text.parquet"


def drop_duplicates(df):
    """Drop duplicate URLs."""

    # Create a list of columns specific to the data
    dedup_cols = list(df.columns)

    # These are generated during the scraping process
    dedup_cols.remove("date_retrieved")
    dedup_cols.remove("page")

    df = df.drop_duplicates(subset=dedup_cols)
    return df.reset_index(drop=True)


# Decorator applies exponentially backoffs to the function, retrying up to six times
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def download_article_text(url):
    """Download the text of a politifact article from the given URL."""
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def download_all_articles(links):
    """Download the text of all politifact articles from the given URL."""
    print("Downloading articles...")
    num_links = len(links)
    for idx, link in enumerate(links, start=1):
        try:
            print(f"\t- ({idx}/{num_links}): {link}")
            yield download_article_text(link)

            # Be kind to PolitiFact servers. Wait between 0.2 and 2.2 seconds.
            time.sleep(0.2 + random.uniform(0, 2))

        except Exception as e:
            print(f"Error downloading {link}: {e}")
            yield None


if __name__ == "__main__":
    fc_df = pd.read_parquet(os.path.join(DATA_DIR, FC_FILE))
    fc_df = drop_duplicates(fc_df)
    fc_analysis_links = fc_df["factcheck_analysis_link"].tolist()
    fc_analysis_text = list(download_all_articles(fc_analysis_links))
    fc_analysis_df = pd.DataFrame(
        {
            "factcheck_analysis_link": fc_analysis_links,
            "factcheck_analysis_text": fc_analysis_text,
        }
    )
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_filename = f"{current_date}_{OUTPUT_FILE}"
    output_path = os.path.join(DATA_DIR, output_filename)
    fc_analysis_df.to_parquet(output_path, index=False)

    print("--- Script Complete ---")
