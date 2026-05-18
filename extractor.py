from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch

torch.set_num_threads(1)

# -----------------------------------
# SENTIMENT MODEL
# -----------------------------------

sentiment_model = pipeline("sentiment-analysis")

# -----------------------------------
# ISSUE CLASSIFICATION MODEL
# -----------------------------------

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# -----------------------------------
# PRODUCT EXTRACTION
# -----------------------------------

# -----------------------------------
# PRODUCT EXTRACTION
# -----------------------------------

def extract_product(text):

    patterns = [
        r'product\s*id[:\s]*([A-Za-z0-9]+)',
        r'asin[:\s]*([A-Za-z0-9]+)',
        r'(\bB[0-9A-Z]{3,}\b)'
    ]

    for p in patterns:

        match = re.search(p, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None

# -----------------------------------
# DATE EXTRACTION
# -----------------------------------

def extract_date(text):

    patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}'
    ]

    for p in patterns:

        match = re.search(p, text)

        if match:
            return match.group(0)

    return None

# -----------------------------------
# RATING EXTRACTION
# -----------------------------------

def extract_rating(text):

    match = re.search(r'\b([1-5])\b', text)

    if match:
        return int(match.group(1))

    return None

# -----------------------------------
# SENTIMENT
# -----------------------------------

def extract_sentiment(text):

    try:

        result = sentiment_model(text[:512])[0]

        label = result["label"]

        if label == "POSITIVE":
            return "positive"

        elif label == "NEGATIVE":
            return "negative"

    except:
        pass

    return "neutral"

# -----------------------------------
# ISSUE CLASSIFICATION
# -----------------------------------

def classify_issue(text, context):

    prompt = f"""
Classify the issue into EXACTLY ONE category.

Categories:
design_issue
performance_issue
usability_issue
durability_issue
unexpected_behavior
no_issue

Context:
{context}

Text:
{text}

Answer ONLY with one category.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    response = response.strip().lower()

    valid = [
        "design_issue",
        "performance_issue",
        "usability_issue",
        "durability_issue",
        "unexpected_behavior",
        "no_issue"
    ]

    for v in valid:
        if v in response:
            return v

    return "unknown_issue"

# -----------------------------------
# MAIN EXTRACTION
# -----------------------------------

def extract_structured(text, context, schema):

    combined = context + "\n" + text

    product = extract_product(combined)

    date = extract_date(combined)

    rating = extract_rating(combined)

    sentiment = extract_sentiment(text)

    issue = classify_issue(text, context)

    return {
        "product": product,
        "issue": issue,
        "sentiment": sentiment,
        "date": date,
        "rating": rating
    }