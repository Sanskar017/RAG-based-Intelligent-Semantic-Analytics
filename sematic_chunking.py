def chunk_documents(documents):

    chunks = []

    for doc in documents:

        sentences = doc["text"].split(".")

        for i, sentence in enumerate(sentences):

            sentence = sentence.strip()

            if len(sentence.split()) < 5:
                continue

            chunks.append({
                "chunk_id": f"{doc['id']}_{i}",
                "text": sentence,
                "source": doc["source"]
            })

    return chunks