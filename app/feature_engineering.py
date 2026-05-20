import re


def extract_custom_features(text):

    text = text.lower()

    suspicious_words = [
        "urgent",
        "verify",
        "password",
        "bank",
        "suspended",
        "click",
        "login",
        "account",
        "crypto",
        "free",
        "winner",
        "limited",
        "security",
        "confirm"
    ]

    suspicious_count = sum(
        word in text for word in suspicious_words
    )

    url_count = len(
        re.findall(r'http[s]?://', text)
    )

    special_chars = len(
        re.findall(r'[$#@!]', text)
    )

    return {
        "url_count": url_count,
        "suspicious_count": suspicious_count,
        "special_chars": special_chars,
        "length": len(text)
    }
