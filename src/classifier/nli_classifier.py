from transformers import pipeline as hf_pipeline
import os

# Intent labels mapped to natural language descriptions
# This is what the NLI model scores against — phrasing matters
INTENT_DESCRIPTIONS = {
    "playback_issue": "music is not playing, skipping, shuffling, or has audio problems",
    "account_access": "cannot log in, access account, or has password and email issues",
    "billing_subscription": "has billing, payment, charge, refund, or subscription issues",
    "content_missing": "song, album, artist or playlist is missing or unavailable",
    "app_device_issue": "the app is crashing, not loading, or has device compatibility issues",
    "general_feedback": "giving feedback, suggestions, complaints or compliments about Spotify",
    "other": "unrelated or unclear request that does not fit other categories"
}

_classifier = None

def get_classifier():
    """Lazy load — only downloads model on first call"""
    global _classifier
    if _classifier is None:
        print("Loading NLI model (first run will download ~90MB)...")
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU
        )
        print("NLI model loaded.")
    return _classifier


def classify(text: str, threshold: float = 0.15) -> str:
    """
    Zero-shot NLI classification.
    threshold: minimum score for top intent, else returns 'other'
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return "other"

    clf = get_classifier()
    candidate_labels = list(INTENT_DESCRIPTIONS.values())
    label_to_intent = {v: k for k, v in INTENT_DESCRIPTIONS.items()}

    result = clf(
        text,
        candidate_labels=candidate_labels,
        multi_label=False
    )

    top_label = result["labels"][0]
    top_score = result["scores"][0]

    if top_score < threshold:
        return "other"

    return label_to_intent.get(top_label, "other")


def classify_batch(texts: list, batch_size: int = 16) -> list:
    """Classify a list of texts efficiently"""
    results = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_results = [classify(t) for t in batch]
        results.extend(batch_results)
        print(f"Classified {min(i+batch_size, total)}/{total}", end="\r")
    print()
    return results


if __name__ == "__main__":
    tests = [
        "songs keep skipping and shuffle doesn't work",
        "I can't log into my account",
        "I was charged twice this month",
        "where is Taylor Swift's new album",
        "the app keeps crashing on my iPhone",
        "please add a sleep timer feature",
        "ok thanks",
        "someone signed into my account from an unknown device"
    ]

    print("NLI Classifier Self-Test:")
    print("-" * 50)
    for text in tests:
        intent = classify(text)
        print(f"→ {intent:<25} | {text[:60]}")