import os
import pandas as pd
import fitz
from docx import Document


def load_data():

    documents = []

    # -----------------------------------
    # CSV PATH
    # -----------------------------------

    csv_path = r"C:\Users\SANSKAR\Downloads\nlp_dataset (1)\nlp_dataset (1)\nlp_dataset\raw\reviews.csv"

    if os.path.exists(csv_path):

        print("Loading CSV...")

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():

            review = str(
                row.get("reviewText", "")
            )

            summary = str(
                row.get("summary", "")
            )

            combined = (
                summary + ". " + review
            )

            documents.append({
                "id": f"csv_{i}",
                "text": combined,
                "source": "csv"
            })

    else:
        print("CSV path not found")

    # -----------------------------------
    # TXT FOLDER
    # -----------------------------------

    txt_path = r"C:\Users\SANSKAR\Downloads\nlp_dataset (1)\nlp_dataset (1)\nlp_dataset\raw\complaints"

    if os.path.exists(txt_path):

        print("Loading TXT files...")

        for file in os.listdir(txt_path):

            if file.endswith(".txt"):

                full_path = os.path.join(
                    txt_path,
                    file
                )

                with open(
                        full_path,
                        "r",
                        encoding="utf-8"
                ) as f:

                    text = f.read()

                documents.append({
                    "id": file,
                    "text": text,
                    "source": "txt"
                })

    else:
        print("TXT folder not found")

    # -----------------------------------
    # DOCX FOLDER
    # -----------------------------------

    docx_path = r"C:\Users\SANSKAR\Downloads\nlp_dataset (1)\nlp_dataset (1)\nlp_dataset\raw\summaries"

    if os.path.exists(docx_path):

        print("Loading DOCX files...")

        for file in os.listdir(docx_path):

            if file.endswith(".docx"):

                full_path = os.path.join(
                    docx_path,
                    file
                )

                doc = Document(full_path)

                text = "\n".join(
                    [p.text for p in doc.paragraphs]
                )

                documents.append({
                    "id": file,
                    "text": text,
                    "source": "docx"
                })

    else:
        print("DOCX folder not found")

    # -----------------------------------
    # PDF FOLDER
    # -----------------------------------

    pdf_path = r"C:\Users\SANSKAR\Downloads\nlp_dataset (1)\nlp_dataset (1)\nlp_dataset\raw\report"

    if os.path.exists(pdf_path):

        print("Loading PDFs...")

        for file in os.listdir(pdf_path):

            if file.endswith(".pdf"):

                full_path = os.path.join(
                    pdf_path,
                    file
                )

                doc = fitz.open(full_path)

                text = ""

                for page in doc:
                    text += page.get_text()

                documents.append({
                    "id": file,
                    "text": text,
                    "source": "pdf"
                })

    else:
        print("PDF folder not found")

    # -----------------------------------
    # FINAL
    # -----------------------------------

    print(
        f"\nTotal loaded documents: {len(documents)}"
    )

    return documents