"""
Purpose:
    Delete all OpenAI batch files.

Inputs:
    Environment variable OPENAI_API_KEY for API authentication.

Output:
    None.
    Batch files with the purpose 'batch' are deleted from OpenAI.

Author: Matthew DeVerna
"""

import os
import sys

from openai import OpenAI


def confirm_deletion():
    """Prompt the user until a valid confirmation is provided."""
    while True:
        confirm = input(
            "WARNING: ALL OpenAI batch files will be deleted! Are you sure you want to proceed? "
            "Please type 'Yes' or 'y' to proceed or 'No' or 'n' to abort: "
        )
        if confirm.lower() in ["yes", "y"]:
            return True
        elif confirm.lower() in ["no", "n"]:
            return False
        else:
            print(
                "Invalid input. "
                "Please type 'Yes' or 'y' to proceed or 'No' or 'n' to abort."
            )


def delete_batch_files():
    """
    Prompt for confirmation and delete all OpenAI batch files if confirmed.
    """
    if not confirm_deletion():
        print("Deletion aborted.")
        sys.exit()

    api_key = os.environ["OPENAI_API_KEY"]
    openai_client = OpenAI(api_key=api_key)
    files = openai_client.files.list()

    if files.data == []:
        sys.exit("No files found. Exiting.")

    for file in files:

        if str(file.purpose) in ["batch", "batch_output"]:
            print(f"Deleting file {file.id} with purpose {file.purpose}")
            openai_client.files.delete(file.id)
        else:
            print(f"Skipping file {file.id} with purpose {file.purpose}")


if __name__ == "__main__":
    delete_batch_files()
