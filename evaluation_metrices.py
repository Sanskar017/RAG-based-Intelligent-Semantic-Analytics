import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------------
# LOAD DATASET
# -----------------------------------

FILE_PATH = r"D:\o2.csv"

df = pd.read_csv(FILE_PATH)

print("\nDataset Loaded Successfully")
print(df.head())

# -----------------------------------
# CREATE FIGURE FOLDER
# -----------------------------------

import os

OUTPUT_DIR = "evaluation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# 1. SENTIMENT DISTRIBUTION
# -----------------------------------

print("\nGenerating Sentiment Distribution...")

sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%'
)

plt.title("Sentiment Distribution")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_distribution.png"
    )
)

plt.close()

# -----------------------------------
# 2. ISSUE TYPE DISTRIBUTION
# -----------------------------------

print("Generating Issue Type Distribution...")

issue_counts = df["issue_type"].value_counts()

plt.figure(figsize=(8, 5))

issue_counts.plot(kind="bar")

plt.title("Issue Type Distribution")

plt.xlabel("Issue Type")

plt.ylabel("Count")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "issue_distribution.png"
    )
)

plt.close()

# -----------------------------------
# 3. SEVERITY DISTRIBUTION
# -----------------------------------

print("Generating Severity Distribution...")

severity_counts = df["severity"].value_counts()

plt.figure(figsize=(6, 5))

severity_counts.plot(kind="bar")

plt.title("Severity Distribution")

plt.xlabel("Severity")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "severity_distribution.png"
    )
)

plt.close()

# -----------------------------------
# 4. SOURCE DISTRIBUTION
# -----------------------------------

print("Generating Source Distribution...")

source_counts = df["source"].value_counts()

plt.figure(figsize=(6, 5))

source_counts.plot(kind="bar")

plt.title("Source-wise Review Distribution")

plt.xlabel("Source")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "source_distribution.png"
    )
)

plt.close()

# -----------------------------------
# 5. TOP KEYWORDS
# -----------------------------------

print("Generating Keyword Frequency...")

all_keywords = []

for row in df["keywords"].dropna():

    kws = row.split(",")

    kws = [k.strip() for k in kws]

    all_keywords.extend(kws)

keyword_counts = Counter(all_keywords)

top_keywords = keyword_counts.most_common(10)

keywords = [k[0] for k in top_keywords]

counts = [k[1] for k in top_keywords]

plt.figure(figsize=(8, 5))

plt.bar(keywords, counts)

plt.title("Top Keywords")

plt.xlabel("Keywords")

plt.ylabel("Frequency")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top_keywords.png"
    )
)

plt.close()

# -----------------------------------
# 6. SOURCE VS SENTIMENT
# -----------------------------------

print("Generating Source vs Sentiment Analysis...")

cross_tab = pd.crosstab(
    df["source"],
    df["sentiment"]
)

cross_tab.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Source vs Sentiment")

plt.xlabel("Source")

plt.ylabel("Count")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "source_vs_sentiment.png"
    )
)

plt.close()

# -----------------------------------
# FINAL SUMMARY
# -----------------------------------

print("\n===================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("===================================")

print(f"\nTotal Reviews Processed: {len(df)}")

print("\nSentiment Counts:")
print(sentiment_counts)

print("\nIssue Type Counts:")
print(issue_counts)

print("\nSeverity Counts:")
print(severity_counts)

print("\nTop Keywords:")
print(top_keywords)

print(
    f"\nAll graphs saved inside: {OUTPUT_DIR}"
)