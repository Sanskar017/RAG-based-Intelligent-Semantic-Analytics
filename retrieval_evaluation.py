import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv(
    "semantic_review_intelligence_output.csv"
)

# -----------------------------------
# MODEL
# -----------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# SAMPLE REVIEWS
# -----------------------------------

reviews = df["review_text"].dropna().tolist()

reviews = reviews[:100]

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

embeddings = model.encode(reviews)

# -----------------------------------
# SIMILARITY MATRIX
# -----------------------------------

similarity_matrix = cosine_similarity(
    embeddings
)

# -----------------------------------
# RETRIEVAL QUALITY
# -----------------------------------

top1_scores = []
top3_scores = []

for i in range(len(similarity_matrix)):

    sims = similarity_matrix[i]

    sims = np.sort(sims)[::-1]

    # remove self similarity
    sims = sims[1:]

    top1_scores.append(sims[0])

    top3_scores.append(
        np.mean(sims[:3])
    )

# -----------------------------------
# METRICS
# -----------------------------------

avg_top1 = np.mean(top1_scores)

avg_top3 = np.mean(top3_scores)

retrieval_consistency = np.std(
    top1_scores
)

print("\n===== RETRIEVAL METRICS =====")

print(
    "Average Top-1 Similarity:",
    avg_top1
)

print(
    "Average Top-3 Similarity:",
    avg_top3
)

print(
    "Retrieval Consistency:",
    retrieval_consistency
)

# -----------------------------------
# TOP-1 DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    top1_scores,
    bins=20
)

plt.title(
    "Top-1 Retrieval Similarity Distribution"
)

plt.xlabel("Cosine Similarity")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/top1_similarity_distribution.png"
)

plt.close()

# -----------------------------------
# TOP-3 DISTRIBUTION
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    top3_scores,
    bins=20
)

plt.title(
    "Top-3 Retrieval Similarity Distribution"
)

plt.xlabel("Average Similarity")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/top3_similarity_distribution.png"
)

plt.close()

# -----------------------------------
# LINE PLOT
# -----------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    top1_scores[:50],
    marker='o'
)

plt.title(
    "Retrieval Similarity Across Samples"
)

plt.xlabel("Sample Index")

plt.ylabel("Top-1 Similarity")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/retrieval_similarity_lineplot.png"
)

plt.close()

# -----------------------------------
# HEATMAP
# -----------------------------------

subset = similarity_matrix[:20, :20]

plt.figure(figsize=(8, 6))

plt.imshow(
    subset,
    aspect='auto'
)

plt.colorbar()

plt.title(
    "Semantic Similarity Heatmap"
)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/semantic_similarity_heatmap.png"
)

plt.close()

print(
    "\nSaved retrieval evaluation plots"
)