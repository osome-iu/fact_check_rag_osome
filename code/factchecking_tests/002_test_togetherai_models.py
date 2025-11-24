"""
Purpose:
    Test various open-sourced models using the Together.ai infrastructure.
    - Ref: https://www.together.ai/

Inputs:
    See the parse_command_line_flags() function for command-line flag details.
    Files with fact checking statements are specified with paths set as constants.

Output:
    .jsonl files containing the results of the fact-checking test.

Author: Matthew DeVerna
"""

import argparse
import chromadb
import json
import os
import sys

import pandas as pd

from together import Together

from llm_fact.prompts import RAG_W_JSON_INST_PROMPT, NON_RAG_W_JSON_INST_PROMPT

from tenacity import retry, stop_after_attempt, wait_random_exponential

api_key = os.environ["TOGETHER_API_KEY"]
together_client = Together(api_key=api_key)

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = "../../data"
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TEST_RESULTS_DIR = os.path.join(RAW_DATA_DIR, "fc_test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
FACT_CHECKS_FP = os.path.join(
    CLEAN_DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)

CHROMA_DIR = os.path.expanduser("~/chroma_fc_rag")
DB_NAME = "fc_rag_cosine"

REASONING_MODELS = ["deepseek-ai/DeepSeek-R1"]

AVAILABLE_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
] + REASONING_MODELS


def parse_command_line_flags():
    """
    Parses command line flags for configuring a vector database.

    Returns:
    ---------
    - args (Namespace): Parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Test Google Gemini fact-checking.")

    # Add arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        required=True,
        help=(f"The model to utilize in the test. Options: {AVAILABLE_MODELS}"),
    )

    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="Include this flag to use RAG (Retrieval-Augmented Generation).",
    )

    parser.add_argument(
        "--num-rag-items",
        type=int,
        choices=[3, 6, 9],
        help="Number of items to return by RAG (must be 3, 6, or 9).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Ensure num_rag_items is specified if with_rag is True
    if args.with_rag and args.num_rag_items is None:
        parser.error("--with-rag flag requires --num-rag-items to be specified.")

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chat(prompt=None, text_content=None, model=None):
    """
    Make an Open AI chat model query.

    Parameters:
    ------------
    - prompt (str): The prompt to use when generating the summary.
    - text_content (str): text content to follow the prompt.
    - model (str): An TogetherAI chat completions model.

    Returns
    ------------
    - response (together): The togetherAI ChatCompletion response object.
    """
    if prompt is None:
        raise ValueError("Prompt must be provided!")
    if text_content is None:
        raise ValueError("Text content must be provided!")
    if model is None:
        raise ValueError("Model must be provided!")

    try:

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{text_content}."},
        ]

        response = together_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        return response

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":

    args = parse_command_line_flags()

    if args.model in REASONING_MODELS and args.with_rag:
        if args.num_rag_items in [3, 9]:
            sys.exit(
                "Skipping DeepSeek R1 models with RAG items 3 or 9 for now. "
                "Exiting..."
            )

    print("Loading the fact-checks...")
    if args.model in REASONING_MODELS:
        print("Whoa nelly, this is an expensive model!")
        print("Let's test a smaller sample...")
        FACT_CHECKS_FP = FACT_CHECKS_FP.replace("0.5", "0.25")
    factchecks = pd.read_parquet(FACT_CHECKS_FP)

    print("Connecting to the vector database...")
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    rag = chroma_client.get_collection(name=DB_NAME)

    print("Building output filename...")
    model_name_short = args.model.split("/")[-1]
    output_filename = f"fc_results__model_{model_name_short}"
    if args.with_rag:
        output_filename += f"__items_{args.num_rag_items}"
    else:
        output_filename += "__no_rag"
    output_filename += ".jsonl"

    output_fp = os.path.join(TEST_RESULTS_DIR, output_filename)
    if os.path.exists(output_fp):
        print("Some claims have already been tested.")
        print("Removing them now...")
        num_claims = len(factchecks)
        completed_links = []
        with open(output_fp, "r") as f:
            for line in f:
                record = json.loads(line)
                completed_links.append(record["factcheck_analysis_link"])
        factchecks = factchecks[
            ~(factchecks["factcheck_analysis_link"].isin(completed_links))
        ]
        num_claims_removed = num_claims - len(factchecks)
        print(f"\t- Removed {num_claims_removed} claims.")

    num_claims = len(factchecks)
    print(f"\n\nNumber of claims to test: {num_claims}\n\n")
    if num_claims == 0:
        sys.exit("No claims to test. Exiting...")

    print("Beginning test with the following input parameters:")
    print(f"\t- Together Model   : {args.model}")
    print(f"\t- With RAG         : {args.with_rag}")
    if args.with_rag:
        print(f"\t- RAG items        : {args.num_rag_items}")
    print("-" * 50)

    with open(output_fp, "a") as f:
        factchecks = factchecks.reset_index(drop=True)
        for idx, row in factchecks.iterrows():
            claim_num = idx + 1
            try:
                proportion = round(claim_num / num_claims, 3)
                print(f"\t- {claim_num}/{num_claims} ({proportion:.1%}) claims")

                # Extract relevant information
                statement_originator = row.statement_originator
                statement = row.statement
                pf_verdict = row.verdict
                fc_link = row.factcheck_analysis_link

                # Build input for OpenAI
                input_text = (
                    f"STATEMENT ORIGINATOR: {statement_originator}\n"
                    f"CLAIM: {statement}\n\n"
                )

                # Build RAG query
                if args.with_rag:
                    rag_query = f"{statement_originator} claimed {statement}"

                    # Returns a list of lists
                    response = rag.query(
                        query_texts=rag_query, n_results=args.num_rag_items
                    )
                    documents = response["documents"][0]

                    input_text += "\n\nFACT-CHECKING INFORMATION:"
                    for idx, doc in enumerate(documents):
                        input_text += f"\n\nSummary {idx+1}:\n{doc}"

                record = {
                    "statement": statement,
                    "politifact_verdict": pf_verdict,
                    "factcheck_analysis_link": fc_link,
                    "statement_originator": statement_originator,
                    "rag_query": rag_query if args.with_rag else None,
                    "rag_results": documents if args.with_rag else None,
                    "togetherai_text_input": input_text,
                }

                # Query Gemini and update record with response details
                prompt = (
                    RAG_W_JSON_INST_PROMPT
                    if args.with_rag
                    else NON_RAG_W_JSON_INST_PROMPT
                )
                response = call_chat(
                    prompt=prompt,
                    text_content=input_text,
                    model=args.model,
                )
                response_dict = response.model_dump()
                record["model_response"] = response_dict

                f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(e, row["factcheck_analysis_link"])

    print("All claims have been tested.")
    print("--- Script Complete ---")
