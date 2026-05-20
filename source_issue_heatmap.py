import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv(
    "semantic_review_intelligence_output.csv"
)

# -----------------------------------
# CROSS TAB
# -----------------------------------

heatmap_data = pd.crosstab(
    df["source"],
    df["issue_type"]
)

# -----------------------------------
# PLOT
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.imshow(
    heatmap_data,
    aspect='auto'
)

plt.colorbar()

plt.xticks(
    range(len(heatmap_data.columns)),
    heatmap_data.columns,
    rotation=30
)

plt.yticks(
    range(len(heatmap_data.index)),
    heatmap_data.index
)

plt.title(
    "Source vs Issue Type Heatmap"
)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/source_issue_heatmap.png"
)

plt.close()

print(
    "Saved source_issue_heatmap.png"
)