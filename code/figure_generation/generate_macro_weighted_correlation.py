"""
Generate a scatter plot showing the correlation between macro and weighted F1 scores.

This script creates a visualization comparing macro F1 scores (x-axis) vs weighted F1 scores
(y-axis) across all model configurations, with points colored by model and styled by web search usage.
Includes a diagonal reference line and correlation coefficient annotation.

Input:
    - multi_class_all_summary_0.25.parquet: Performance metrics for all models (~6k claims)

Output:
    - macro_weighted_correlation.pdf: Scatter plot saved to figures/ directory

Author: Matthew DeVerna
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import spearmanr

# Ensure we are in the correct directory
os.chdir(Path(__file__).parent)

# Define data paths
classification_report_path = Path(
    "../../data/cleaned/classification_reports/multi_class_all_summary_0.25.parquet"
)
output_path = Path("../../figures/si_macro_weighted_correlation.pdf")

# Ensure figures directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load performance data
multi_summary_df = pd.read_parquet(classification_report_path)

# Drop results for this model because we have a more recent version
drop_this_model = "gpt-4o-2024-08-06"
multi_summary_df = multi_summary_df[
    multi_summary_df.model != drop_this_model
].reset_index(drop=True)

# Calculate Spearman's correlation
rho, p_value = spearmanr(
    multi_summary_df["f1-score-macro-avg"],
    multi_summary_df["f1-score-weighted-avg"]
)

# Format p-value
if p_value < 0.001:
    p_str = r"$p < 0.001$"
elif p_value < 0.01:
    p_str = r"$p < 0.01$"
elif p_value < 0.05:
    p_str = r"$p < 0.05$"
else:
    p_str = fr"$p = {p_value:.2f}$"

# Create the scatter plot
fig, ax = plt.subplots(figsize=(4, 4))

sns.scatterplot(
    data=multi_summary_df,
    x="f1-score-macro-avg",
    y="f1-score-weighted-avg",
    ax=ax,
)

# Add diagonal reference line (y = x)
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)

# Configure axes
ax.set_xlabel("macro F1")
ax.set_ylabel("weighted F1")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(fr"$\rho = {rho:.3f}$, {p_str}")

# Style the plot
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3)

# Set equal aspect ratio to make the diagonal line at 45 degrees
ax.set_aspect("equal", adjustable="box")

# Define output paths
pdf_path = output_path
png_path = Path(str(output_path).replace(".pdf", ".png"))

# Save the figure in both formats
plt.tight_layout()
plt.savefig(pdf_path, bbox_inches="tight", dpi=900)
plt.savefig(png_path, bbox_inches="tight", dpi=900)

print("Macro vs weighted F1 correlation plot saved to:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
