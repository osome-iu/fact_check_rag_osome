"""
Purpose:
    Update a vector database with new records.

Inputs:
    See the parse_command_line_flags() function for command-line flag details.
    Fact checking information that will be fed into the database are specified
    with paths to files set as constants.

Output:
    Creates the Chroma collection as defined by the CL flags.

Author: Matthew DeVerna
"""

import chromadb
import os

import pandas as pd

# Ensure that the current directory is the location of this script for relative paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))


DISTANCE_METRIC = "cosine"
USER_HOME = os.path.expanduser("~")

CHROMA_DIR = os.path.join(USER_HOME, "chroma_fc_rag")
os.makedirs(CHROMA_DIR, exist_ok=True)


DATA_DIR = "../../data/cleaned/"
FC_SUMMARIES_FP = os.path.join(DATA_DIR, "factcheck_summaries.parquet")
FACT_CHECKS_FP = os.path.join(DATA_DIR, "2024-10-10_factchecks_cleaned.parquet")


def add_records_to_collection(df, collection):
    """
    Add summary records from dataframe to the specified collection.

    NOTE: Politifact verdicts are appended to the end of the summary before being
    added to the collection.

    Parameters:
    -----------
    - df (pandas.DataFrame): the dataframe of records to add. Must contain the
        following columns:
        - summary: summaries of the fact checking articles.
        - verdict: the verdict of the fact checking article.
    - collection (chromadb.Collection): the collection to add records to.
    """
    required_columns = ["summary", "verdict"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    num_items = collection.count()
    ids = [str(i).zfill(10) for i in range(num_items + 1, len(df) + 1)]
    documents = [
        f"{summary} (Politifact verdict: {verdict})"
        for verdict, summary in zip(df.verdict, df.summary)
    ]
    collection.add(
        ids=ids,
        documents=documents,
    )
    print("Records added to collection.")


if __name__ == "__main__":

    print("Setting up the database...")
    db_name = f"fc_rag_{DISTANCE_METRIC}"
    print(f"\t- Database Name           : {db_name}")
    print(f"\t- Database distance metric: {DISTANCE_METRIC}")
    client = chromadb.PersistentClient(CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=db_name,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )
    print(f"Database {db_name} created.\n\n")

    print("Loading summaries...")
    summaries_df = pd.read_parquet(FC_SUMMARIES_FP)

    print("Loading claims/verdicts...")
    fcs_df = pd.read_parquet(FACT_CHECKS_FP)

    print("Combining verdicts and summaries...")
    summaries_df = summaries_df.merge(
        fcs_df[["factcheck_analysis_link", "verdict"]], on="factcheck_analysis_link"
    )

    print("Adding new records...")
    add_records_to_collection(summaries_df, collection)

    print("--- Script complete ---")
