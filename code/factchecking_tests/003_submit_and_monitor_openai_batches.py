"""
Purpose:
    Submit fact-checking batch tests, monitor them until they are finished, then save them.

Inputs:
    Batch files with fact checking tasks are loaded with paths set as constants.

Output:
    .jsonl files containing the fact-checking body.

Author: Matthew DeVerna
"""

import os
import time

from openai import OpenAI

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = "../../data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
BATCHES_DIR = os.path.join(RAW_DATA_DIR, "fc_batch_files")
RESULTS_DIR = os.path.join(RAW_DATA_DIR, "fc_test_results")

# How many seconds to wait between checking if a batch job has finished.
WAIT_SECS = 60

api_key = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=api_key)


def process_and_monitor_batch(file: str) -> None:
    """
    Process and monitor an OpenAI batch file for a given model.

    Parameters
    ----------
    file : str
        The name of the batch file containing fact-checking tasks.

    Returns
    -------
    None
        Submits the batch job, monitors its status, and saves results or errors.
    """
    print(f"Working on batch file: {file}")

    # `file` format: fc_batch__model_o3-mini-2025-01-31__no_rag__v01.jsonl
    output_fname = file.replace("fc_batch", "fc_results")
    output_fp = os.path.join(RESULTS_DIR, output_fname)
    if os.path.exists(output_fp):
        print(f"\t Output file already exists: {output_fp}")
        print("\t Skipping batch job submission.\n\n")
        return

    # Define the task name
    task_model = file.split("__")[1].replace("model_", "")
    task_rag = file.split("__")[2]
    task_name = f"task__{task_model}__{task_rag}"

    # Create an openai batch file, then use the file id to create a batch job
    batch_fp = os.path.join(BATCHES_DIR, file)
    with open(batch_fp, "rb") as f:
        batch_file = openai_client.files.create(file=f, purpose="batch")
    batch_input_file_id = batch_file.id
    batch_job = openai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",  # Only optional at this time
        metadata={"description": f"task_name: {task_name}"},
    )
    print("Job submitted successfully!")

    job_id = batch_job.id
    print("Monitoring batch job...")
    print(f"\t Task name: {task_name}")
    print(f"\t Batch ID : {job_id}")
    print("-" * 50)

    while True:
        time.sleep(WAIT_SECS)
        batch_job = openai_client.batches.retrieve(job_id)
        status = batch_job.status

        if status == "completed":
            print(f"\t *** Task {task_name} completed. ***")
            if batch_job.output_file_id:
                result = openai_client.files.content(batch_job.output_file_id).content
                with open(output_fp, "wb") as f:
                    f.write(result)
                print(f"\t Results saved successfully at {output_fp}")
            elif batch_job.error_file_id:
                error_details = openai_client.files.content(
                    batch_job.error_file_id
                ).content
                error_output_path = output_fp.replace(".jsonl", "__ERRORS.json")
                with open(error_output_path, "wb") as f:
                    f.write(error_details)
                print(
                    f"\t Batch completed with errors. Details saved at {error_output_path}"
                )
            else:
                print("\t Batch completed without generating output or error files.")
            break

        elif status in ["validating", "in_progress", "finalizing"]:
            print(f"\t Task <{task_name}> status: <{status}>. Continuing to monitor...")

        elif status in ["failed", "expired", "cancelled"]:
            print(f"\t Task <{task_name}> status: <{status}>. Exiting monitoring.")
            break

        else:
            print(f"\t Task <{task_name}> encountered unknown status: {status}")


if __name__ == "__main__":

    print("Begin processing and monitoring batch jobs...")
    print("#" * 50, "\n\n")

    files = sorted(os.listdir(BATCHES_DIR))
    for file in files:
        process_and_monitor_batch(file)
    print("All batch jobs completed.")
