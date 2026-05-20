import time
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

# -----------------------------------
# MODELS
# -----------------------------------

models = {

    "MiniLM":
    "all-MiniLM-L6-v2",

    "MPNet":
    "all-mpnet-base-v2",

    "DistilRoBERTa":
    "all-distilroberta-v1",

    "BGE-Small":
    "BAAI/bge-small-en-v1.5",

    "GTE-Base":
    "thenlper/gte-base",

    "MultiQA":
    "multi-qa-MiniLM-L6-cos-v1"
}

# -----------------------------------
# SAMPLE TEXTS
# -----------------------------------

sample_texts = [

    "battery drains quickly",

    "excellent product quality",

    "random crashes happen",

    "very confusing interface",

    "beautiful stylish design"

] * 20

# -----------------------------------
# BENCHMARK
# -----------------------------------

results = []

for name, model_name in models.items():

    print(f"\nTesting {name}")

    model = SentenceTransformer(
        model_name
    )

    start = time.time()

    embeddings = model.encode(
        sample_texts
    )

    end = time.time()

    inference_time = end - start

    embedding_dim = len(
        embeddings[0]
    )

    throughput = (
        len(sample_texts) /
        inference_time
    )

    results.append({

        "Model": name,

        "Inference_Time":
        inference_time,

        "Embedding_Dimension":
        embedding_dim,

        "Throughput":
        throughput
    })

# -----------------------------------
# DATAFRAME
# -----------------------------------

df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")

print(df)

# -----------------------------------
# SAVE TABLE
# -----------------------------------

df.to_csv(
    "evaluation_outputs/model_comparison_table.csv",
    index=False
)

# -----------------------------------
# INFERENCE TIME
# -----------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    df["Model"],
    df["Inference_Time"]
)

plt.title(
    "Model Inference Time Comparison"
)

plt.ylabel("Seconds")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/model_inference_time.png"
)

plt.close()

# -----------------------------------
# THROUGHPUT
# -----------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    df["Model"],
    df["Throughput"]
)

plt.title(
    "Model Throughput Comparison"
)

plt.ylabel("Texts Per Second")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/model_throughput.png"
)

plt.close()

# -----------------------------------
# EMBEDDING DIMENSIONS
# -----------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    df["Model"],
    df["Embedding_Dimension"]
)

plt.title(
    "Embedding Dimension Comparison"
)

plt.ylabel("Dimensions")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "evaluation_outputs/model_embedding_dimensions.png"
)

plt.close()

print(
    "\nSaved model comparison plots"
)