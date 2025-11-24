# data_cleaning/

Scripts for cleaning and processing data for analysis.

## Contents

- `001_clean_summaries.py`: Clean and validate GPT-3.5 summaries of PolitiFact articles
- `002_sample_claims.py`: Sample claims for manual annotation
- `002_subsample_claims.py`: Create subsamples of claims for testing
- `004_clean_fc_results_pt1.py`: First pass cleaning of fact-checking results
- `004_clean_fc_results_pt2_bad_json.py`: Second pass cleaning for malformed JSON responses
- `005_extract_web_sources_from_fc_results.py`: Extract web source URLs from fact-checking results
- `combined_clean_fc_results.py`: Combine all cleaned fact-checking results into final dataset
- `count_bad_llm_models.py`: Count malformed responses by model
- `extract_unparsable_json_fcs_w_gpt4o_mini.py`: Use GPT-4o-mini to extract valid JSON from malformed responses
- `sample_bad_llm_responses_exact_match.py`: Sample malformed responses for manual validation
