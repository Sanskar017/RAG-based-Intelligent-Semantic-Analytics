import pandas as pd

from loader import load_data
from sematic_chunking import chunk_documents
from vector_db import build_index
from retriever_RAG import retrieve_context
from analyzer import analyze_review


def run_pipeline():

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    print("Loading data...\n")

    docs = load_data()

    print(f"\nLoaded {len(docs)} documents")

    # -----------------------------------
    # CHECK EMPTY DATA
    # -----------------------------------

    if len(docs) == 0:

        print("\n❌ No documents loaded")

        return

    # -----------------------------------
    # CHUNKING
    # -----------------------------------

    print("\nChunking documents...\n")

    chunks = chunk_documents(docs)

    print(f"Generated {len(chunks)} chunks")

    # -----------------------------------
    # CHECK EMPTY CHUNKS
    # -----------------------------------

    if len(chunks) == 0:

        print("\n❌ No chunks generated")

        return

    # -----------------------------------
    # BUILD VECTOR DATABASE
    # -----------------------------------

    print("\nBuilding vector database...\n")

    index, texts = build_index(chunks)

    # -----------------------------------
    # ANALYSIS LOOP
    # -----------------------------------

    results = []

    print("\nRunning Semantic RAG Analysis...\n")

    for i, chunk in enumerate(chunks):

        print(
            f"Processing chunk {i+1}/{len(chunks)}"
        )

        # -----------------------------------
        # RAG RETRIEVAL
        # -----------------------------------

        context = retrieve_context(
            chunk["text"],
            index,
            texts,
            k=3
        )

        context_text = " ".join(context)

        # keep context smaller for stability
        context_text = context_text[:500]

        # -----------------------------------
        # ANALYSIS
        # -----------------------------------

        analysis = analyze_review(
            chunk["text"],
            context_text
        )

        # -----------------------------------
        # ADD SOURCE INFO
        # -----------------------------------

        analysis["source"] = chunk["source"]

        analysis["chunk_id"] = chunk["chunk_id"]

        # -----------------------------------
        # STORE RESULT
        # -----------------------------------

        results.append(analysis)

        # -----------------------------------
        # DEBUG PRINT
        # -----------------------------------

        print(analysis)
        print()

    # -----------------------------------
    # SAVE OUTPUT
    # -----------------------------------

    print("\nSaving final dataset...\n")

    df = pd.DataFrame(results)

    df.to_csv(
        "semantic_review_intelligence_output.csv",
        index=False
    )

    # -----------------------------------
    # FINAL
    # -----------------------------------

    print(
        "\n✅ Pipeline completed successfully"
    )

    print(
        "\nSaved file:"
    )

    print(
        "semantic_review_intelligence_output3.csv"
    )


if __name__ == "__main__":

    run_pipeline()