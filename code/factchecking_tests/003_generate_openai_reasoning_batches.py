"""
Purpose:
    Generate batches of fact-checking tasks for different OpenAI models.

Inputs:
    Files with fact checking statements are specified with paths set as constants.

Output:
    .jsonl files containing the fact-checking body.

Author: Matthew DeVerna
"""

import chromadb
import json
import os

import pandas as pd

from llm_fact.prompts import RAG_PROMPT, NON_RAG_PROMPT
from llm_fact.response_structure import OpenAIFactCheckingResponse

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

RAG_ITEMS = [0, 6]
DATA_DIR = "../../data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
BATCHES_DIR = os.path.join(RAW_DATA_DIR, "fc_batch_files")
os.makedirs(BATCHES_DIR, exist_ok=True)

CHROMA_DIR = os.path.expanduser("~/chroma_fc_rag")
DB_NAME = "fc_rag_cosine"

FACT_CHECKS_FP = os.path.join(
    CLEAN_DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.25_per_year.parquet"
)

MODELS = ["o3-mini-2025-01-31", "o1-2024-12-17"]


def get_unique_output_fp(num_rag_items: int, model: str) -> str:
    """
    Generate a unique output file path based on a given batch file name.
    Iteratively increments a version number to avoid overwriting existing files.

    Parameters
    ----------
    num_rag_items : int
        Number of RAG items to retrieve (0 implies no RAG information)
    model : str
        The OpenAI model being used.

    Returns
    -------
    output_fp : str
        A unique output file path string in the BATCHES_DIR, which includes a version
        suffix (formatted as '__vXX') to avoid overwriting existing files.
    """

    rag_fp_tail = f"items_{num_rag_items}" if num_rag_items != 0 else "no_rag"
    base_file_name = f"fc_batch__model_{model}__{rag_fp_tail}"

    # Increment version number to avoid overwriting existing files
    version = 1
    output_fp = f"{base_file_name}__v{version:02d}.jsonl"
    output_fp = os.path.join(BATCHES_DIR, output_fp)
    while os.path.exists(output_fp):
        version += 1
        output_fp = os.path.join(BATCHES_DIR, f"{base_file_name}__v{version:02d}.jsonl")
    return output_fp


def generate_task_list(model, num_rag_items, fact_check_df, rag):
    """
    Generate a list of tasks for fact-checking given a model and a RAG configuration.

    Parameters
    ----------
    model : str
        The OpenAI model to use.
    num_rag_items : int
        Number of RAG items to retrieve (0 implies no RAG information).
    fact_check_df : pandas.DataFrame
        DataFrame containing fact-checking statements.
    rag : object
        The ChromaDB collection used for performing RAG queries.

    Returns
    -------
    list
        A list of task dictionaries to be processed.
    """
    task_list = []
    for _, row in fact_check_df.iterrows():
        # Extract relevant information
        statement_originator = row.statement_originator
        statement = row.statement
        fc_link = row.factcheck_analysis_link

        task_base = {
            "custom_id": f"url__{fc_link}",
            "method": "POST",
            "url": "/v1/chat/completions",
        }

        # Build input for OpenAI
        input_text = (
            f"STATEMENT ORIGINATOR: {statement_originator}\n" f"CLAIM: {statement}\n\n"
        )

        # Build RAG query
        if num_rag_items != 0:
            rag_query = f"{statement_originator} claimed {statement}"
            response = rag.query(
                query_texts=rag_query, n_results=num_rag_items
            )  # Returns a list of lists
            documents = response["documents"][0]
            input_text += "\n\nFACT-CHECKING INFORMATION:"
            for i, doc in enumerate(documents):
                input_text += f"\n\nSummary {i+1}:\n{doc}"

        prompt = RAG_PROMPT if num_rag_items != 0 else NON_RAG_PROMPT

        schema = OpenAIFactCheckingResponse.model_json_schema()
        schema["additionalProperties"] = False  # Required for batches

        task = task_base.copy()

        # Set role based on model as OpenAI is in a transition period with the API language
        role = "developer" if "o1" in model else "system"
        task["body"] = {
            "model": model,
            "messages": [
                {"role": role, "content": prompt},
                {"role": "user", "content": input_text},
            ],
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "OpenAIFactCheckingResponse",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        task_list.append(task)
    return task_list


def create_batches(fact_check_df, rag):
    """
    Generate and save batches of fact-checking tasks for each model and RAG configuration.

    Parameters
    ----------
    fact_check_df : pandas.DataFrame
        DataFrame containing fact-checking statements.
    rag : object
        The ChromaDB collection to perform RAG queries.

    Returns
    -------
    None
    """
    for model in MODELS:
        for num_rag_items in RAG_ITEMS:
            print(f"Generating batch for {model} with {num_rag_items} RAG items...")
            task_list = generate_task_list(model, num_rag_items, fact_check_df, rag)
            output_fp = get_unique_output_fp(num_rag_items, model)
            with open(output_fp, "w") as f:
                for task in task_list:
                    f.write(json.dumps(task) + "\n")
            print(f"\t Saved here: {output_fp}")


if __name__ == "__main__":
    print("Generating batches of fact-checking tasks for OpenAI models...\n")
    fact_check_df = pd.read_parquet(FACT_CHECKS_FP)
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    rag = chroma_client.get_collection(name=DB_NAME)

    create_batches(fact_check_df, rag)

    print("Finished.")
