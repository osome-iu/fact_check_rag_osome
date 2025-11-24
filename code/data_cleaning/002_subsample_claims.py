"""
Purpose:
    Select a subsample of claims from the already sampled claims.

Inputs:
    Reads the already sampled fact checked dataset from the path set as a constant.

Output:
    The sampled fact checks as a .parquet file.

Author: Matthew DeVerna
"""

import os
import sys
import pandas as pd

SAMPLE_PROPORTION = 0.5

# Ensure we are in the correct directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = "../../data/cleaned/"
FACTCHECKS_FNAME = "2024-10-10_factchecks_cleaned_sampled_0.5_per_year.parquet"
FACTCHECKS_FPATH = os.path.join(DATA_DIR, FACTCHECKS_FNAME)

# Load the already sampled data set (which is a 50% sample from each year)
factchecks = pd.read_parquet(FACTCHECKS_FPATH)

# Convert fact check dates to datetimes and extract the year
factchecks["factcheck_date"] = pd.to_datetime(factchecks["factcheck_date"])
factchecks["fc_year"] = factchecks["factcheck_date"].dt.year

# Create the output file path
new_fname = FACTCHECKS_FNAME.replace("_0.5_", "_0.25_")
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
