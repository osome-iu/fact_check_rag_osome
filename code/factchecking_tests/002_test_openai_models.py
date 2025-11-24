"""
Purpose:
    Test OpenAI RAG's fact checking ability.

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

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

api_key = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=api_key)

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from llm_fact.prompts import RAG_PROMPT, NON_RAG_PROMPT
from llm_fact.response_structure import OpenAIFactCheckingResponse

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

EXPENSIVE_MODELS = ["gpt-4.5-preview-2025-02-27"]

SEARCH_MODELS = ["gpt-4o-search-preview", "gpt-4o-mini-search-preview"]

AVAILABLE_MODELS = (
    [
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
    ]
    + EXPENSIVE_MODELS
    + SEARCH_MODELS
)


def parse_command_line_flags():
    """
    Parses command line flags for configuring a vector database.

    Returns:
    ---------
    - args (Namespace): Parsed command-line arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Test OpenAI RAG fact-checking ability based on vector database setttings and LLM model."
    )

    # Add arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        required=True,
        help="The model to utilize in the test.",
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

    parser.add_argument(
        "--web-search",
        action="store_true",
        default=False,
        help=(
            "Include this flag to use web-search (default: False). "
            f"Only available for the following models: {', '.join(SEARCH_MODELS)}"
        ),
    )

    # Parse arguments
    args = parser.parse_args()

    # Ensure num_rag_items is specified if with_rag is True
    if args.with_rag and args.num_rag_items is None:
        parser.error("--with-rag flag requires --num-rag-items to be specified.")

    if args.web_search:
        if args.model not in SEARCH_MODELS:
            parser.error(
                f"Model {args.model} does not support web search. "
                f"Please choose one of: {', '.join(SEARCH_MODELS)}"
            )

    return args


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chat(
    prompt=None, text_content=None, model=None, response_format=None, web_search=False
):
    """
    Make an Open AI chat model query.

    Parameters:
    ------------
    - prompt (str): The prompt to use when generating the summary.
    - text_content (str): text content to follow the prompt.
    - model (str): An OpenAI chat completions model (default = gpt-4o-mini-2024-07-18)
    - response_format (pydantic.BaseModel): The response structure to use when
        generating the summary.
    - web_search (bool): Whether to use the web search feature (default = False).

    Returns
    ------------
    - response (OpenAI): The OpenAI response object.
    """
    if prompt is None:
        raise ValueError("Prompt must be provided!")
    if text_content is None:
        raise ValueError("Text content must be provided!")
    if model is None:
        raise ValueError("Model must be provided!")
    if response_format is None:
        raise ValueError("Response structure must be provided!")
    if not isinstance(web_search, bool):
        raise ValueError("Web search must be a boolean!")

    try:

        if "gpt-4o" in model:
            messages = [
                {"role": "developer", "content": prompt},
                {"role": "user", "content": f"{text_content}."},
            ]

            if not web_search:
                response = openai_client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                )
            else:
                # temperature cannot be passed with web search
                response = openai_client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    web_search_options={"search_context_size": "medium"},
                )

        else:
            merged_content = f"{prompt}\n\n{text_content}."
            messages = [
                {"role": "user", "content": merged_content},
            ]

            # Must call separately, as o1 does not take a temperature parameter
            response = openai_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
            )

        return response

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":

    args = parse_command_line_flags()

    print("Loading the fact-checks...")
    if args.model in EXPENSIVE_MODELS or args.web_search:
        print("Whoa nelly, this is an expensive model!")
        print("Let's test a smaller sample...")
        FACT_CHECKS_FP = FACT_CHECKS_FP.replace("0.5", "0.25")
    factchecks = pd.read_parquet(FACT_CHECKS_FP)

    print("Connecting to the vector database...")
    client = chromadb.PersistentClient(CHROMA_DIR)
    rag = client.get_collection(name=DB_NAME)

    print("Building output filename...")
    output_filename = f"fc_results__model_{args.model}"
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
    print(f"\t- OpenAI Model: {args.model}")
    print(f"\t- With RAG    : {args.with_rag}")
    if args.with_rag:
        print(f"\t- RAG items   : {args.num_rag_items}")
    if args.web_search:
        print(f"\t- Web Search  : {args.web_search}")
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
                open_ai_text = (
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

                    open_ai_text += "\n\nFACT-CHECKING INFORMATION:"
                    for idx, doc in enumerate(documents):
                        open_ai_text += f"\n\nSummary {idx+1}:\n{doc}"

                record = {
                    "statement": statement,
                    "politifact_verdict": pf_verdict,
                    "factcheck_analysis_link": fc_link,
                    "statement_originator": statement_originator,
                    "rag_query": rag_query if args.with_rag else None,
                    "rag_results": documents if args.with_rag else None,
                    "openai_text_input": open_ai_text,
                }

                # Query OpenAI and update record with response details
                prompt = RAG_PROMPT if args.with_rag else NON_RAG_PROMPT
                response = call_chat(
                    prompt=prompt,
                    text_content=open_ai_text,
                    model=args.model,
                    response_format=OpenAIFactCheckingResponse,
                    web_search=args.web_search,
                )
                response_dict = response.model_dump()
                record["model_response"] = response_dict

                f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(e, row["factcheck_analysis_link"])

    print("All claims have been tested.")
    print("--- Script Complete ---")
