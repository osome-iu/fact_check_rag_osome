"""
Purpose:
    Test Gemini's fact checking ability.

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
import time

import pandas as pd

from enum import Enum
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from llm_fact.prompts import (
    RAG_PROMPT,
    NON_RAG_PROMPT,
    NON_RAG_W_JSON_INST_PROMPT,
    RAG_W_JSON_INST_PROMPT,
)
from llm_fact.response_structure import GoogleFactCheckingResponse

from tenacity import retry, stop_after_attempt, wait_random_exponential

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

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

REASONING_MODELS = ["gemini-2.0-flash-thinking-exp-01-21"]

AVAILABLE_MODELS = [
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
] + REASONING_MODELS


def custom_serializer(obj):
    """
    Custom serializer for converting objects to strings.

    This specifically handles Google's Enum objects which are not serializable
    and are returned in the `parsed` field of the response.
    """
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


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

    parser.add_argument(
        "--grounded",
        action="store_true",
        default=False,
        help="Include this flag to use grounded responses (default: False).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Ensure num_rag_items is specified if with_rag is True
    if args.with_rag and args.num_rag_items is None:
        parser.error("--with-rag flag requires --num-rag-items to be specified.")

    return args


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(7))
def call_chat(model: str, prompt: str, response_schema=None, grounded=False):
    """
    Generate a fact-checking response using one of Google's Gemini models with
    support for grounded responses.

    Parameters
    ----------
    - model : str
        The name of the generative model to use.
    - prompt : str
        The input prompt for the generative model.
    - response_schema : list[TypedDict] or None, optional
        The schema defining the structure of the response.
        Cannot be used with the grounded flag.
    - grounded : bool
        If True, use google_search_tool for grounded responses. Cannot be used
        with the response_schema parameter.
        If False, do not use google_search_tool for grounded responses.

    Returns
    -------
    response : dict
        The response from Google.
    """
    for name, val in [("model", model), ("prompt", prompt)]:
        if val is None:
            raise ValueError(f"Missing required parameter: {name}")

    if response_schema is not None and grounded:
        raise ValueError(
            "Cannot use 'response_schema' with grounded responses. Please set one of them."
        )

    if "thinking" in model and response_schema is not None:
        raise ValueError(
            "Cannot use 'thinking' model with grounded responses or response_schema. "
            "Please set one of them."
        )

    if "thinking" in model and not grounded:
        config = GenerateContentConfig(
            response_modalities=["TEXT"],
        )
    elif response_schema is not None:
        config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )
    elif grounded:
        google_search_tool = Tool(google_search=GoogleSearch())
        config = GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    else:
        raise Exception("Unknown configuration.")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response


if __name__ == "__main__":

    args = parse_command_line_flags()

    if args.model in REASONING_MODELS and args.with_rag:
        if args.num_rag_items in [3, 9]:
            sys.exit(
                "Skipping reasoning models with RAG items 3 or 9 for now. " "Exiting..."
            )

    print("Loading the fact-checks...")
    if args.model in REASONING_MODELS or args.grounded:
        print("Whoa nelly, this is an expensive model!")
        print("Let's test a smaller sample...")
        FACT_CHECKS_FP = FACT_CHECKS_FP.replace("0.5", "0.25")
    factchecks = pd.read_parquet(FACT_CHECKS_FP)

    print("Connecting to the vector database...")
    chroma_client = chromadb.PersistentClient(CHROMA_DIR)
    rag = chroma_client.get_collection(name=DB_NAME)

    print("Building output filename...")
    ground_str = "grounded" if args.grounded else "ungrounded"
    output_filename = f"fc_results__model_{args.model}__{ground_str}"
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
    print(f"\t- Gemini Model   : {args.model}")
    print(f"\t- With RAG       : {args.with_rag}")
    if args.with_rag:
        print(f"\t- RAG items      : {args.num_rag_items}")
    print(f"\t- Grounded       : {args.grounded}")
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

                # Build RAG query.
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
                    "gemini_text_input": input_text,
                }

                # Structured responses can't be used in these cases.
                if args.grounded or "thinking" in args.model:
                    prompt = (
                        RAG_W_JSON_INST_PROMPT
                        if args.with_rag
                        else NON_RAG_W_JSON_INST_PROMPT
                    )
                    schema = None
                else:
                    prompt = RAG_PROMPT if args.with_rag else NON_RAG_PROMPT
                    schema = list[GoogleFactCheckingResponse]

                prompt += f"\n\n{input_text}"
                response = call_chat(
                    model=args.model,
                    prompt=prompt,
                    response_schema=schema,
                    grounded=args.grounded,
                )
                response_dict = response.model_dump()
                record["model_response"] = response_dict

                if "exp" in args.model:
                    print("Sleeping for 6 seconds to avoid rate limit...")
                    time.sleep(6)

                f.write(json.dumps(record, default=custom_serializer) + "\n")

            except Exception as e:
                print(e, row["factcheck_analysis_link"])

    print("All claims have been tested.")
    print("--- Script Complete ---")
