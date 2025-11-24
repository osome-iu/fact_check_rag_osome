"""
Purpose:
    - Summarize the PolitiFact fact check analysis articles text with OpenAI GPT3.5.

Input:
    - None
    - Data read from path set as a constant.

Output:
    - .jsonl file containing the raw OpenAI ChatCompletion response data in dictionary form.

Author:
    - Matthew DeVerna
"""

import glob
import json
import os

import pandas as pd

from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up OpenAI client
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=api_key)

# Ensure that the current directory is the location of this script for relative paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = "../../data/raw"

# File name suffixes
FC_TEXT_SUMMARIES_FILE = "fc_analysis_text_summaries.jsonl"
FC_TEXT_FILE = "fc_analysis_text.parquet"
FULL_FC_FILE = "factchecks.parquet"

# This is used to clean the text before passing to OpenAI
VERDICT_MAP = {
    "true": "True",
    "mostly-true": "Mostly true",
    "half-true": "Half true",
    "mostly-false": "Mostly false",
    "false": "False",
    "tom_ruling_pof": "Pants on fire",
    "full-flop": "Full flop",
    "half-flip": "Half flip",
    "no-flip": "No flip",
}

PROMPT = (
    "As an AI assistant, your task is to summarize text from a PolitiFact fact-checking article. "
    "The input text may contain incomplete sentences, HTML tags, and non-textual elements. First, "
    "clean the text by removing any irrelevant content or formatting issues. Then, write a concise, "
    "neutral summary focusing on the article's main conclusion and supporting facts, covering who, "
    "what, when, where, and why.\n\n"
    "The summary should be one paragraph, free of editorializing or subjective interpretation. "
    "Provide only the summary with no additional text or comments. "
    "If no article text is provided, respond with 'No article text provided.' "
    "Follow these instructions strictly to ensure an accurate, unbiased summary."
)


# Decorator applies exponentially backoffs to the function, retrying up to six times
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def summarize(article_text, model="gpt-3.5-turbo", raw=True, prompt=None):
    """
    Summarize article text.

    Parameters:
    ------------
    - article_text (str): raw article text scraped from the web.
    - model (str): An OpenAI chat completions model (default = gpt-3.5-turbo)
    - raw (bool): True (default): returns the OpenAI ChatCompletion object directly.
        False: return the summary text only.
    - prompt (str): The prompt to use when generating the summary.

    Returns
    ------------
    - If raw == True (default): returns the OpenAI ChatCompletion object directly.
    - If raw == False: return the summary text only.
    """
    if prompt is None:
        raise ValueError("Prompt must be provided!")

    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Article text: {article_text}."},
        ]
        response = openai_client.chat.completions.create(
            model=model, messages=messages, temperature=0
        )
        if raw:
            return response
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Error: {e}")
        raise


def load_completed_articles(path):
    """
    Load the completed articles from the output file.
    """
    with open(path, "r") as f:
        return [json.loads(line)["factcheck_analysis_link"] for line in f]


if __name__ == "__main__":
    # These files are prefixed with dates, so we need to glob them
    fc_text_path = glob.glob(os.path.join(DATA_DIR, f"*_{FC_TEXT_FILE}"))[0]
    fcs_path = glob.glob(os.path.join(DATA_DIR, f"*_{FULL_FC_FILE}"))[0]
    fc_df = pd.read_parquet(fcs_path)

    # Build the output path, check if the file exists, and load completed articles to skip
    fc_summaries_path = os.path.join(DATA_DIR, FC_TEXT_SUMMARIES_FILE)
    completed_articles = []
    if os.path.exists(fc_summaries_path):
        print("Loading completed articles to skip...")
        completed_articles.extend(load_completed_articles(fc_summaries_path))
        print(f"{len(completed_articles)} articles already summarized.")

    print("Loading all article text, dropping failed/missing/completed links...")
    fc_text_df = pd.read_parquet(fc_text_path)

    # Masks for completed, failed, and empty text
    completed_mask = fc_text_df["factcheck_analysis_link"].isin(completed_articles)
    failed_mask = fc_text_df["factcheck_analysis_text"].isna()
    empty_mask = fc_text_df["factcheck_analysis_text"] == ""

    # Drop completed articles, failed, and empty text
    fc_text_df = fc_text_df[~completed_mask & ~failed_mask & ~empty_mask]

    # Add and clean up verdicts. E.g., "true" -> "True" and "mostly-true" -> "Mostly true"
    merged = pd.merge(
        fc_text_df,
        fc_df[["verdict", "factcheck_analysis_link"]],
        on="factcheck_analysis_link",
        how="inner",
    )
    fc_text_df = merged[~(merged.duplicated("factcheck_analysis_link"))].reset_index(
        drop=True
    )
    fc_text_df["verdict"] = fc_text_df["verdict"].map(VERDICT_MAP)

    # Open with append so that we don't overwrite existing summaries
    num_items = len(fc_text_df)
    print("Begin summarizing...")
    with open(fc_summaries_path, "a") as f:
        for idx, row in fc_text_df.iterrows():

            text = row["factcheck_analysis_text"]
            print(f"\t- {idx}/{num_items} ({round(idx/num_items,2)}) summaries")

            try:
                # Add verdict to summary if applicable
                text = f"Verdict: {row['verdict']}\n\n{text}"

                summary_response = summarize(
                    article_text=text,
                    model="gpt-3.5-turbo",
                    raw=True,  # True returns the OpenAI ChatCompletion object directly
                    prompt=PROMPT,
                )
                # Convert data into a dictionary, add the factcheck_analysis_link, and write to the file
                response_as_dict = summary_response.model_dump()
                response_as_dict["factcheck_analysis_link"] = row[
                    "factcheck_analysis_link"
                ]
                f.write(json.dumps(response_as_dict) + "\n")

            except Exception as e:
                print(e, row["factcheck_analysis_link"])

    print("--- Script Complete ---")
