"""
Purpose:
    Identify the rank of the correct summary within the RAG retrieval results.

Inputs:
    - factchecks dataframe
    - summaries dataframe
    - paths are set as constants

Output:
    - retrieval_ranks.parquet
    - Columns are:
    factcheck_analysis_link (str): the unique fact checking link
    rank (int): the rank position of the correct summary within the retrieved results (0 if not found)

Author: Matthew DeVerna
"""

import chromadb
import os

# Ensure we are in the directory where the script is saved
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import pandas as pd

USER_HOME = os.path.expanduser("~")

CHROMA_DIR = os.path.join(USER_HOME, "chroma_fc_rag")
DB_NAME = "fc_rag_cosine"

DATA_DIR = "../../data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
FACT_CHECKS_FP = os.path.join(
    CLEAN_DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)
SUMMARIES_FP = os.path.join(CLEAN_DATA_DIR, "factcheck_summaries.parquet")


# Load data and connect to the database
factchecks_df = pd.read_parquet(
    FACT_CHECKS_FP,
    columns=["factcheck_analysis_link", "statement", "statement_originator"],
)
summaries_df = pd.read_parquet(
    SUMMARIES_FP, columns=["factcheck_analysis_link", "summary"]
)

client = chromadb.PersistentClient(CHROMA_DIR)
rag = client.get_collection(name=DB_NAME)
num_db_items = rag.count()

combined_df = pd.merge(
    factchecks_df,
    summaries_df,
    on="factcheck_analysis_link",
    how="inner",
)


num_claims = len(combined_df)
records = []

for idx, row in combined_df.iterrows():
    print(f"Processing claim {idx+1}/{num_claims}")
    fc_link = row.factcheck_analysis_link
    statement_originator = row.statement_originator
    statement = row.statement
    summary = row.summary

    # Retrieve items with increasing batch sizes until the summary is found or
    # all items have been searched.
    rag_query = f"{statement_originator} claimed {statement}"
    rank = 0
    batch_size = 10

    while batch_size <= num_db_items:
        current_batch = min(batch_size, num_db_items)
        response = rag.query(query_texts=[rag_query], n_results=current_batch)
        documents = response["documents"][0]

        for doc_rank, doc in enumerate(documents, start=1):
            if summary in doc.split(" (Politifact verdict: ")[0]:
                rank = doc_rank
                break

        if rank > 0 or current_batch == num_db_items:
            break

        batch_size *= 10

    records.append(
        {
            "factcheck_analysis_link": fc_link,
            "rank": rank,
        }
    )

# Save the results
df = pd.DataFrame.from_records(records)
output_fp = os.path.join(CLEAN_DATA_DIR, "retrieval_ranks.parquet")
df.to_parquet(output_fp)
print(f"Saved results to {output_fp}\n\n")

print("Conducting some basic statistics on the retrieval performance...\n")

# Calculate the proportion of claims not found. This is a check and should be 0 if all summaries are found
prop_not_found = (df["rank"] == 0).sum() / len(df)
assert np.isclose(prop_not_found, 0.0), "Warning: Some summaries were not found."

# Calculate the proportion of summaries ranked above 3, 6, or 9 items
num_links = len(df)
thresholds = [3, 6, 9]
proportions = {}
for k in thresholds:
    count_above = (df["rank"] <= k).sum()
    proportions[k] = count_above / num_links

print("\nProportion of claims's returning summaries...:")
for threshold, prop in proportions.items():
    print(f"\t ...above {threshold}: {prop:.2%}")

# Calculate mean, median, and standard deviation of the found ranks
mean_rank = df["rank"].mean()
median_rank = df["rank"].median()
std_rank = df["rank"].std()

print("\nRank Statistics (for claims with summaries found in top 500 items):")
print(f"\t- Mean: {mean_rank:.2f}")
print(f"\t- Median: {median_rank:.2f}")
print(f"\t- Std: {std_rank:.2f}")

print("\nScript complete.")
