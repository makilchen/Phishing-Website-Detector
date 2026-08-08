"""
generate_dataset.py
--------------------
Builds a labeled dataset of legitimate-style and phishing-style URLs.

Note: this environment has no internet access, so we can't download the
UCI/Kaggle phishing dataset directly. Instead we synthetically generate
realistic URLs using templates that mirror the well-documented patterns
seen in real phishing vs legitimate URLs (short trusted domains vs long
hyphenated/IP-based/suspicious-keyword URLs), then extract the same
lexical features used at inference time. A small amount of label noise
is added so the classification task isn't trivially easy (real datasets
always have overlapping/ambiguous cases too).

For an actual submission with a real published dataset, swap this file's
output for the UCI "Phishing Websites Dataset" / Kaggle equivalent --
the rest of the pipeline (features.py, train_model.py, app.py) doesn't
need to change since it only depends on the feature extraction logic.
"""

import random
import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features, FEATURE_NAMES  # noqa: E402

random.seed(42)

LEGIT_BRANDS = [
    "google", "amazon", "wikipedia", "github", "microsoft", "apple",
    "netflix", "spotify", "linkedin", "flipkart", "paypal", "chase",
    "irctc", "nptel", "coursera", "leetcode", "stackoverflow", "reddit",
]
LEGIT_TLDS = ["com", "org", "in", "io", "edu", "net"]
LEGIT_PATHS = ["", "/", "/home", "/products", "/about", "/docs", "/search?q=item", "/user/profile"]

PHISH_BRANDS = LEGIT_BRANDS  # attackers impersonate the same real brands
PHISH_KEYWORDS = ["login", "verify", "secure", "update", "confirm", "account", "signin", "webscr"]
SUSPICIOUS_TLDS = ["zip", "review", "country", "kim", "cricket", "science", "work", "party", "gq", "tk"]


def random_string(n):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=n))


def make_legit_url():
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    path = random.choice(LEGIT_PATHS)
    subdomain = random.choice(["", "www.", "en.", "shop."])
    return f"https://{subdomain}{brand}.{tld}{path}"


def make_phishing_url():
    brand = random.choice(PHISH_BRANDS)
    keyword = random.choice(PHISH_KEYWORDS)
    style = random.choice(["ip", "hyphen_subdomain", "fake_tld", "long_query", "at_symbol"])

    if style == "ip":
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        return f"http://{ip}/{brand}-{keyword}/{random_string(6)}"
    elif style == "hyphen_subdomain":
        return f"http://{brand}-{keyword}-{random_string(4)}.{random_string(6)}.com/{keyword}"
    elif style == "fake_tld":
        tld = random.choice(SUSPICIOUS_TLDS)
        return f"http://{brand}{keyword}.{tld}/{keyword}.php"
    elif style == "long_query":
        return (f"http://{random_string(8)}.com/{brand}/{keyword}"
                f"?user={random_string(5)}&token={random_string(10)}&redirect={random_string(6)}")
    else:  # at_symbol
        return f"http://{brand}.com@{random_string(8)}.{random.choice(SUSPICIOUS_TLDS)}/{keyword}"


def generate_dataset(n_per_class=1200, noise_rate=0.05):
    rows = []
    for _ in range(n_per_class):
        url = make_legit_url()
        label = 0  # legitimate
        if random.random() < noise_rate:
            label = 1
        rows.append((url, label))

    for _ in range(n_per_class):
        url = make_phishing_url()
        label = 1  # phishing
        if random.random() < noise_rate:
            label = 0
        rows.append((url, label))

    random.shuffle(rows)
    return rows


def main():
    rows = generate_dataset()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phishing_dataset.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"] + FEATURE_NAMES + ["label"])
        for url, label in rows:
            feats = extract_features(url)
            writer.writerow([url] + [feats[name] for name in FEATURE_NAMES] + [label])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
