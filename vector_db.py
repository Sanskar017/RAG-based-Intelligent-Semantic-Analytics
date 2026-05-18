from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def build_index(chunks):

    if len(chunks) == 0:
        raise ValueError(
            "No chunks found. Check dataset loading."
        )

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts)

    embeddings = np.array(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index, texts