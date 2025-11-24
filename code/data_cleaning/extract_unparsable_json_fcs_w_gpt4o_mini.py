"""
Purpose:
    Use ChatGPTs gpt-4o-mini model to extract and return clean, well formatted
    JSON from model responses that were not easily parsable.

Inputs:
    - `bad_llm_fc_results.jsonl`: Paths set below with constants.
    - This is a file that contains the results of fact-checking tests for various
    models. We load the data, extract the messy JSON responses, and then pass
    them to the gpt-4o-mini model to return them in clean JSON.

Output:
    - `bad_llm_fc_results_clean.jsonl`: .jsonl file containing the results of
    the fact-checking test. Will include the factchecking_analysis_link to map
    across files.

Author: Matthew DeVerna
"""

import json
import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

api_key = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=api_key)

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from llm_fact.prompts import MESSY_TO_CLEAN_PROMPT
from llm_fact.response_structure import OpenAIFactCheckingResponse

INTERMEDIATE_DATA_DIR = "../../data/intermediate"
BAD_LLM_FC_RESULTS_PATH = os.path.join(
    INTERMEDIATE_DATA_DIR, "bad_llm_fc_results.jsonl"
)
CLEAN_LLM_FC_RESULTS_PATH = os.path.join(
    INTERMEDIATE_DATA_DIR, "bad_llm_fc_results_extracted.jsonl"
)

MODEL = "gpt-4o-mini"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chat(prompt, text_content, model, response_format):
    """
    Make an OpenAI chat model query using gpt-4o-mini.

    Parameters:
    ------------
    - prompt (str): The prompt to use.
    - text_content (str): Text content to follow the prompt.
    - model (str): The OpenAI chat model.
    - response_format (pydantic.BaseModel): The expected response structure.

    Returns:
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

    messages = [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": f"{text_content}."},
    ]

    try:
        response = openai_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=0,
        )
        return response
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":

    # Load the bad LLM fact-checking results
    with open(BAD_LLM_FC_RESULTS_PATH, "r") as f:
        bad_llm_fc_records = [json.loads(line) for line in f]

    num_records = len(bad_llm_fc_records)
    print(f"Loaded {num_records:,} records from {BAD_LLM_FC_RESULTS_PATH}")

    # Filter out records that have already been processed
    if os.path.exists(CLEAN_LLM_FC_RESULTS_PATH):
        print(
            "Some results have already been processed. Filtering out those records..."
        )
        with open(CLEAN_LLM_FC_RESULTS_PATH, "r") as f:
            processed_records = [json.loads(line) for line in f if line.strip()]
        fc_link_model_name_pairs = {
            (record["factcheck_analysis_link"], record["model_name"])
            for record in processed_records
        }

        original_len = len(bad_llm_fc_records)
        bad_llm_fc_records = [
            record
            for record in bad_llm_fc_records
            if (record["factcheck_analysis_link"], record["model_name"])
            not in fc_link_model_name_pairs
        ]
        difference = original_len - len(bad_llm_fc_records)
        print(f"Filtered out {difference} records already processed.")

    print(f"Processing {len(bad_llm_fc_records):,} records...")

    # Changed file open mode to append ("a") to update with new records.
    with open(CLEAN_LLM_FC_RESULTS_PATH, "a") as f:

        for idx, record in enumerate(bad_llm_fc_records, start=1):
            print(f"Handling record {idx}/{len(bad_llm_fc_records)}")

            # Identify the model family to help extract the messy response
            model_name = record["model_name"]
            fc_link = record["factcheck_analysis_link"]
            num_rag_items = record["num_rag_items"]
            used_web_search = record["used_web_search"]

            # There are only problems with together.ai and gemini model output
            if "llama" in model_name.lower() or "deepseek" in model_name.lower():
                response_obj = record["model_response"]
                messy_response = response_obj["choices"][0]["message"]["content"]
            elif "gemini" in model_name.lower():
                response_obj = record["model_response"]
                messy_response = response_obj["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
            else:
                raise ValueError(f"Can't identify model family for file: {model_name}")

            response_record = {
                "factcheck_analysis_link": fc_link,
                "model_name": model_name,
                "num_rag_items": num_rag_items,
                "used_web_search": used_web_search,
            }
            response = call_chat(
                prompt=MESSY_TO_CLEAN_PROMPT,
                text_content=messy_response,
                model=MODEL,
                response_format=OpenAIFactCheckingResponse,
            )

            # Convert to a dictionary, add fc link, and write to file
            response_dict = response.model_dump()
            response_record["model_response"] = response_dict
            f.write(json.dumps(response_record) + "\n")

    print("Script completed successfully.")
    print(f"Saved cleaned results to {CLEAN_LLM_FC_RESULTS_PATH}")
