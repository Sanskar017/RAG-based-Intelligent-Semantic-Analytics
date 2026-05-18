from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def similarity(a, b):
    v1 = model.encode([a])[0]
    v2 = model.encode([b])[0]
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

def deduplicate(records, threshold=0.85):
    unique = []

    for r in records:
        is_dup = False

        for u in unique:
            if similarity(str(r), str(u)) > threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(r)

    return unique