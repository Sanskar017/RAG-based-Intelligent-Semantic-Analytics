from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------------
# SENTIMENT MODEL
# -----------------------------------

sentiment_model = pipeline(
    "sentiment-analysis"
)

# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# ISSUE CATEGORY DESCRIPTIONS
# -----------------------------------

ISSUE_DESCRIPTIONS = {

    "performance_issue":
        "battery drain slow lag charging heat performance problem",

    "durability_issue":
        "broken crack damage durability weak build quality",

    "usability_issue":
        "hard difficult confusing interface usability issue",

    "design_issue":
        "bad design ugly appearance uncomfortable style",

    "unexpected_behavior":
        "random issue weird sudden unexpected bug",

    "no_issue":
        "good excellent satisfied works perfectly no problem"
}

# -----------------------------------
# PRECOMPUTE ISSUE EMBEDDINGS
# -----------------------------------

issue_labels = list(
    ISSUE_DESCRIPTIONS.keys()
)

issue_embeddings = embedding_model.encode(
    list(ISSUE_DESCRIPTIONS.values())
)

# -----------------------------------
# KEYWORD EXTRACTION
# -----------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5
)

def extract_keywords(text):

    try:

        tfidf = vectorizer.fit_transform([text])

        words = vectorizer.get_feature_names_out()

        return ", ".join(words)

    except:
        return ""

# -----------------------------------
# SEMANTIC ISSUE CLASSIFICATION
# -----------------------------------

def classify_issue(text, context):

    combined = context + " " + text

    review_embedding = embedding_model.encode(
        [combined]
    )

    similarities = cosine_similarity(
        review_embedding,
        issue_embeddings
    )[0]

    best_index = np.argmax(similarities)

    return issue_labels[best_index]

# -----------------------------------
# SEVERITY
# -----------------------------------

def detect_severity(sentiment,
                    issue):

    if sentiment == "negative":

        if issue != "no_issue":
            return "high"

        return "medium"

    return "low"

# -----------------------------------
# MAIN ANALYSIS
# -----------------------------------

def analyze_review(text,
                   context):

    # -----------------------------------
    # SENTIMENT
    # -----------------------------------

    try:

        s = sentiment_model(text[:512])[0]

        sentiment = (
            "positive"
            if s["label"] == "POSITIVE"
            else "negative"
        )

    except:
        sentiment = "neutral"

    # -----------------------------------
    # ISSUE CLASSIFICATION
    # -----------------------------------

    issue = classify_issue(
        text,
        context
    )

    # -----------------------------------
    # KEYWORDS
    # -----------------------------------

    keywords = extract_keywords(text)

    # -----------------------------------
    # SEVERITY
    # -----------------------------------

    severity = detect_severity(
        sentiment,
        issue
    )

    return {
        "review_text": text,
        "sentiment": sentiment,
        "issue_type": issue,
        "keywords": keywords,
        "severity": severity
    }