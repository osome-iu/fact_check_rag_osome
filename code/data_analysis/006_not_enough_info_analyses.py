"""
Purpose:
    Analyze the use of the "not enough information" (NEI) label in multiple ways:
    - Evaluate NEI as a binary classification task, including precision, recall, F1, and accuracy.
    - Generate a confusion matrix for NEI predictions (true/false positives/negatives).
    - Calculate the overall and verdict-specific proportions of claims labeled as NEI.
    - Summarize NEI usage and model performance across different retrieval settings and web search usage.

Notes:
    - This script analyzes the NEI label using confusion matrix quadrants:
        - True Positive (TP): Model correctly predicts NEI when supporting evidence is absent.
        - True Negative (TN): Model correctly predicts not NEI when supporting evidence is present.
        - False Positive (FP): Model predicts NEI when supporting evidence is present (overuse).
        - False Negative (FN): Model fails to predict NEI when supporting evidence is absent (underuse).
    - Results are stratified by model, retrieval settings, and web search usage.

Inputs:
    See the constants below for the paths to the input files.

Outputs:
    nei_performance_metrics.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - precision: Precision for NEI as a binary classification.
        - recall: Recall for NEI as a binary classification.
        - f1_score: F1 score for NEI as a binary classification.
        - accuracy: Accuracy for NEI as a binary classification.
    nei_confusion_summary.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - true_positive: Count of claims where the model correctly predicted NEI when it should have.
        - false_positive: Count of claims where the model predicted NEI when it should not have (overuse).
        - true_negative: Count of claims where the model correctly predicted not NEI when it should not have.
        - false_negative: Count of claims where the model failed to predict NEI when it should have (underuse).
    nei_given_props.parquet
        - model: Name of the model evaluated.
        - num_rag_items: Number of RAG items used for retrieval.
        - used_web_search: Whether web search was used (bool).
        - prop_nei_overall: Proportion of all claims labeled as NEI.
        - prop_nei_{verdict}: Proportion of claims with a specific verdict labeled as NEI (e.g., prop_nei_true, prop_nei_false, etc.).

Author: Matthew DeVerna
"""

import os
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# Ensure we are in the dir where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === CONSTANTS ===
DATA_DIR = "../../data/cleaned"
RETRIEVAL_PATH = os.path.join(DATA_DIR, "retrieval_ranks.parquet")
FACTCHECKS_PATH = os.path.join(
    DATA_DIR, "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
)
RESULTS_PATH = os.path.join(DATA_DIR, "cleaned_factchecking_results_combined.parquet")
NEI_PERFORMANCE_METRICS_PATH = os.path.join(DATA_DIR, "nei_performance_metrics.parquet")
NEI_CONFUSION_SUMMARY_PATH = os.path.join(DATA_DIR, "nei_confusion_summary.parquet")
NEI_GIVEN_PROPS_PATH = os.path.join(DATA_DIR, "nei_given_props.parquet")


# === FUNCTIONS ===
def analyze_nei_behavior(temp_df, rank_cutoff="in_top_9"):
    """
    Returns results of a binary classification analysis and a confusiong matrix for
    the "Not enough information" (NEI) label.

    Parameters
    ----------
    temp_df : pd.DataFrame
        DataFrame with model predictions and retrieval coverage indicators.
    rank_cutoff : str, default="in_top_9"
        Column name (e.g., 'in_top_3', 'in_top_6', 'in_top_9') used to determine if
        the correct document was retrieved.

    Returns
    -------
    metrics : dict
        Dictionary with binary classification metrics (precision, recall, f1, accuracy).

    confusion_summary : dict
        Dictionary with counts for true positives, false positives, true negatives,
        and false negatives, to support overuse/underuse analysis.
    """
    assert rank_cutoff in {"in_top_3", "in_top_6", "in_top_9"}, "Invalid rank_cutoff"

    # Ground truth: should the model say NEI?
    # If correct summary not in top-k, model *should* say NEI
    y_should_be_nei = (~temp_df[rank_cutoff]).astype(int)

    # Model prediction: did the model say NEI?
    y_predicted_nei = (temp_df["label"] == "Not enough information").astype(int)

    # Binary classification metrics
    precision = precision_score(y_should_be_nei, y_predicted_nei)
    recall = recall_score(y_should_be_nei, y_predicted_nei)
    f1 = f1_score(y_should_be_nei, y_predicted_nei)
    accuracy = accuracy_score(y_should_be_nei, y_predicted_nei)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_should_be_nei, y_predicted_nei).ravel()

    confusion_summary = {
        "true_positive": int(tp),  # Correctly said NEI when it should have
        "false_positive": int(fp),  # Said NEI when it shouldn't have (overuse)
        "true_negative": int(tn),  # Correctly gave verdict when it should have
        "false_negative": int(fn),  # Failed to say NEI when it should have (underuse)
    }

    return metrics, confusion_summary


# === MAIN EXECUTION ===
if __name__ == "__main__":
    retrieval_df = pd.read_parquet(RETRIEVAL_PATH)
    fact_checks_df = pd.read_parquet(
        FACTCHECKS_PATH,
        columns=["factcheck_analysis_link", "verdict"],
    )
    results_df = pd.read_parquet(
        RESULTS_PATH,
        columns=[
            "factcheck_analysis_link",
            "model",
            "label",
            "num_rag_items",
            "used_web_search",
            "prompt_tokens",
        ],
    )

    fact_checks_df = pd.merge(
        results_df,
        fact_checks_df,
        on="factcheck_analysis_link",
        how="left",
    )

    retrieval_df["in_top_3"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 3 else False
    )
    retrieval_df["in_top_6"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 6 else False
    )
    retrieval_df["in_top_9"] = retrieval_df["rank"].apply(
        lambda x: True if x <= 9 else False
    )

    fcs_w_recall_hits = pd.merge(
        fact_checks_df,
        retrieval_df[["factcheck_analysis_link", "in_top_3", "in_top_6", "in_top_9"]],
        on="factcheck_analysis_link",
        how="left",
    )

    nei_given_props = []
    performance_records = []
    confusion_records = []

    for group, temp_df in fcs_w_recall_hits.groupby(
        ["model", "num_rag_items", "used_web_search"]
    ):
        model_name = group[0]
        num_rag_items = group[1]
        used_web_search = group[2]
        if num_rag_items == 0:
            continue
        base_record = {
            "model": model_name,
            "num_rag_items": num_rag_items,
            "used_web_search": used_web_search,
        }
        nei_given_record = base_record.copy()

        # Select rows where NEI was the model's response
        nei_df = temp_df[temp_df["label"] == "Not enough information"]

        # Calculate the proportion of claims that were labeled "Not enough information"
        nei_response_prop = nei_df.shape[0] / temp_df.shape[0]

        # Proportion of claims *by verdict* that were labeled "Not enough information"
        label_by_verdict_props = (
            temp_df.groupby(["verdict"])["label"]
            .value_counts(normalize=True)
            .reset_index(name="prop")
        )
        # Convert to a dictionary to include in the nei_given_record
        nei_by_verdict_props = label_by_verdict_props[
            label_by_verdict_props["label"] == "Not enough information"
        ].copy()
        nei_by_verdict_props.drop(columns=["label"], inplace=True)
        nei_by_verdict_props_dict = {
            f"prop_nei_{row.verdict.lower().replace(' ', '_')}": row.prop
            for row in nei_by_verdict_props.itertuples()
        }
        nei_given_record.update(
            {
                "prop_nei_overall": nei_response_prop,
                **nei_by_verdict_props_dict,
            }
        )
        nei_given_props.append(nei_given_record)

        # Calculate NEI behavior metrics
        metrics, confusion_summary = analyze_nei_behavior(
            temp_df, rank_cutoff=f"in_top_{(int(num_rag_items))}"
        )

        # Create and update performance and confusion records
        performance_record = base_record.copy()
        confusion_record = base_record.copy()
        performance_record.update(metrics)
        confusion_record.update(confusion_summary)

        # Append the records to the lists
        performance_records.append(performance_record)
        confusion_records.append(confusion_record)

    # Build DataFrames from the records
    performance_df = pd.DataFrame(performance_records).sort_values(
        ["model", "num_rag_items"]
    )
    confusion_df = (
        pd.DataFrame(confusion_records)
        .sort_values(["model", "num_rag_items"])
        .reset_index(drop=True)
    )
    nei_given_df = (
        pd.DataFrame(nei_given_props)
        .sort_values(["model", "num_rag_items"])
        .reset_index(drop=True)
    )

    # Save the results to parquet files
    performance_df.to_parquet(NEI_PERFORMANCE_METRICS_PATH, index=False)
    confusion_df.to_parquet(NEI_CONFUSION_SUMMARY_PATH, index=False)
    nei_given_df.to_parquet(NEI_GIVEN_PROPS_PATH, index=False)
