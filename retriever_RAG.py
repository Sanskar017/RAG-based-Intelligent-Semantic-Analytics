from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def retrieve_context(query,
                     index,
                     texts,
                     k=3):

    q_vec = model.encode([query])

    distances, indices = index.search(q_vec, k)

    retrieved = []

    for i in indices[0]:
        retrieved.append(texts[i])

    return retrieved