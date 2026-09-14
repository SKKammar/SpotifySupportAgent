import re

INTENT_KEYWORDS = {
    "playback_issue": [
        "skip", "skipping", "shuffle", "repeat", "play", "playing",
        "playback", "buffer", "buffering", "audio", "sound", "queue",
        "stream", "streaming", "song won't", "not playing", "stops playing",
        "keeps pausing", "pausing", "pause"
    ],
    "account_access": [
        "login", "log in", "log out", "logout", "password", "sign in",
        "signin", "locked", "can't access", "cannot access", "email",
        "username", "account access", "forgot password", "reset password",
        "unauthorized", "hacked", "someone else", "not my device"
    ],
    "billing_subscription": [
        "charge", "charged", "charging", "bill", "billing", "payment",
        "paid", "pay", "refund", "cancel", "cancelled", "cancellation",
        "premium", "subscription", "free trial", "trial", "price",
        "pricing", "student discount", "family plan", "overcharged",
        "double charged", "credit card", "debit"
    ],
    "content_missing": [
        "missing", "not available", "can't find", "cannot find",
        "removed", "taken off", "taken down", "where is", "not on spotify",
        "not showing", "disappeared", "gone", "album", "artist not",
        "song not", "playlist gone", "deleted playlist"
    ],
    "app_device_issue": [
        "crash", "crashing", "crashes", "won't open", "not opening",
        "won't load", "not loading", "error", "bug", "glitch",
        "offline", "download", "sync", "syncing", "app", "update",
        "android", "iphone", "ios", "windows", "mac", "desktop",
        "mobile", "alexa", "car", "driving mode", "connect",
        "connection", "device", "400", "500"
    ],
    "general_feedback": [
        "please add", "feature request", "suggestion", "would love",
        "wish you", "you should", "idea", "feedback", "complain",
        "disappointed", "frustrated", "worst", "terrible", "amazing",
        "love spotify", "great app", "thank you", "thanks spotify"
    ]
}

def classify(text: str) -> str:
    """
    Keyword-based intent classifier.
    Returns the intent with the most keyword matches.
    Falls back to 'other' if no keywords match.
    """
    if not text or not isinstance(text, str):
        return "other"

    text_lower = text.lower()
    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(
            1 for kw in keywords
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        )
        if score > 0:
            scores[intent] = score

    if not scores:
        return "other"

    return max(scores, key=scores.get)


if __name__ == "__main__":
    # Quick self-test
    tests = [
        ("songs keep skipping and shuffle doesn't work", "playback_issue"),
        ("I can't log into my account", "account_access"),
        ("I was charged twice this month", "billing_subscription"),
        ("where is Taylor Swift's new album", "content_missing"),
        ("the app keeps crashing on my iPhone", "app_device_issue"),
        ("please add a sleep timer feature", "general_feedback"),
        ("ok thanks", "other"),
    ]

    print("Keyword Baseline Self-Test:")
    print("-" * 50)
    correct = 0
    for text, expected in tests:
        predicted = classify(text)
        status = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        print(f"{status} Expected: {expected:<25} Got: {predicted}")

    print(f"\nSelf-test accuracy: {correct}/{len(tests)}")