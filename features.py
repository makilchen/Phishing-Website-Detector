"""
features.py
------------
Extracts lexical / structural features from a URL string.
Used both for building the training dataset and for real-time
prediction in the Flask app, so training and inference always
stay in sync.

All features are computed purely from the URL text itself (no
network calls / WHOIS lookups), so this works fully offline.
"""

import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "password", "webscr", "ebayisapi",
    "suspend", "alert", "urgent", "click", "limited",
]

SHORTENING_SERVICES = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "shorte.st",
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_query_params",
    "has_at_symbol",
    "has_ip_address",
    "has_https_token",
    "has_double_slash_redirect",
    "is_shortened",
    "has_suspicious_word",
    "has_suspicious_tld",
]

SUSPICIOUS_TLDS = ["zip", "review", "country", "kim", "cricket", "science", "work", "party", "gq", "tk"]


def _get_hostname(url: str) -> str:
    parsed = urlparse(url if "//" in url else "//" + url)
    return parsed.hostname or ""


def extract_features(url: str) -> dict:
    """Return an ordered dict of numeric features for a given URL string."""
    url = url.strip()
    hostname = _get_hostname(url)
    path_and_after = url.split("://", 1)[-1]  # everything after scheme, for "//" redirect check
    after_scheme_body = path_and_after[len(hostname):] if hostname else path_and_after

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(url.count(c) for c in ["%", "$", "!", "*", ";", "+"]),
        "num_subdomains": max(hostname.count(".") - 1, 0) if hostname else 0,
        "num_query_params": url.count("=") + url.count("&"),
        "has_at_symbol": int("@" in url),
        "has_ip_address": int(bool(IP_PATTERN.match(hostname))),
        "has_https_token": int("https" in url.lower().split("://")[0] or url.lower().startswith("https")),
        "has_double_slash_redirect": int("//" in after_scheme_body),
        "is_shortened": int(any(s in hostname for s in SHORTENING_SERVICES)),
        "has_suspicious_word": int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
        "has_suspicious_tld": int(hostname.split(".")[-1].lower() in SUSPICIOUS_TLDS) if hostname else 0,
    }
    return features


def features_to_vector(features: dict):
    """Convert a features dict into an ordered list matching FEATURE_NAMES."""
    return [features[name] for name in FEATURE_NAMES]
