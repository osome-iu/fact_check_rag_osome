"""
Purpose:
    Sample claims to test on. We take 50% of the claims from each year in the dataset.

Inputs:
    Reads the scraped fact checked dataset from the path set as a constant.

Output:
    The sampled fact checks as a .parquet file.

Author: Matthew DeVerna
"""

import os
import sys
import pandas as pd

# SAMPLE_PROPORTIONS = [0.05, 0.5]
SAMPLE_PROPORTION = 0.5

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = "../../data/cleaned/"
FACTCHECKS_FNAME = "2024-10-10_factchecks_cleaned.parquet"
FACTCHECKS_FPATH = os.path.join(DATA_DIR, FACTCHECKS_FNAME)
EXCLUDED_VERDICTS = ["Full flop", "Half flip", "No flip"]

# Load full data set and exclude bad links
factchecks = pd.read_parquet(FACTCHECKS_FPATH)

# Convert fact check dates to datetimes and extract the year
factchecks["factcheck_date"] = pd.to_datetime(factchecks["factcheck_date"])
factchecks["fc_year"] = factchecks["factcheck_date"].dt.year

# Remove claims not related to veracity from the dataset
excluded_verdicts_mask = factchecks["verdict"].isin(EXCLUDED_VERDICTS)
factchecks = factchecks[~excluded_verdicts_mask]

# Create the output file path
new_fname = FACTCHECKS_FNAME.replace(
    ".parquet", f"_sampled_{SAMPLE_PROPORTION}_per_year.parquet"
)
output_fpath = os.path.join(DATA_DIR, new_fname)
if os.path.exists(output_fpath):
    print(f"File already exists: {output_fpath}. Exiting...")
    sys.exit()

# Sample a proportion of each year's fact checks, fixing the random state for reproducibility
sampled_factchecks = (
    factchecks.groupby("fc_year", group_keys=False)
    .apply(
        lambda x: x.sample(frac=SAMPLE_PROPORTION, random_state=1), include_groups=False
    )
    .reset_index(drop=True)
)

print(f"Sampled {len(sampled_factchecks)} fact checks")

sampled_factchecks.to_parquet(output_fpath)
print(f"Saved: {output_fpath}")

print("--- Script Complete ---")
