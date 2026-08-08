# Phishing Website Detection using Machine Learning

A mini project that classifies a URL as **Legitimate** or **Phishing** using
a machine learning model trained on lexical/structural URL features, served
through a small Flask web app.

Pairs with the accompanying project report (`Phishing_Website_Detection_Report.docx`).

## Project structure

```
phishing-project/
├── features.py              # shared feature extraction (used by training + app)
├── data/
│   ├── generate_dataset.py  # builds the labeled training dataset
│   └── phishing_dataset.csv # generated dataset (2,400 URLs)
├── train_model.py           # trains & compares 4 algorithms, saves the best one
├── model/
│   ├── model.pkl            # saved best model (created after training)
│   └── results.json         # accuracy/precision/recall comparison table
├── app.py                   # Flask backend + /predict API
├── templates/
│   └── index.html           # web interface
└── requirements.txt
```

## How it works

1. **`features.py`** extracts 16 features from a raw URL string — length,
   number of dots/hyphens, presence of an IP address, `@` symbol, suspicious
   keywords (`login`, `verify`, `secure`...), suspicious TLDs, etc. Everything
   is computed from the URL text itself, so it works fully offline (no WHOIS
   or network lookups).
2. **`data/generate_dataset.py`** builds a labeled dataset of ~2,400 URLs by
   generating legitimate-style URLs (real brand names, clean domains) and
   phishing-style URLs (IP addresses, hyphenated fake subdomains, suspicious
   TLDs, long query strings) with a small amount of label noise so the task
   isn't trivial. *(This environment has no internet access to download the
   real UCI/Kaggle phishing dataset — swap this file's CSV output for that
   dataset if you want to train on real-world data instead; nothing else in
   the pipeline needs to change.)*
3. **`train_model.py`** trains Logistic Regression, Decision Tree, Random
   Forest, and SVM, compares them on accuracy/precision/recall, and saves the
   best-performing one to `model/model.pkl`.
4. **`app.py`** loads the saved model and exposes `/predict`, which takes a
   URL, extracts the same 16 features, and returns a label + confidence.

## Setup & run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python3 data/generate_dataset.py

# 3. Train the model (prints the comparison table)
python3 train_model.py

# 4. Start the web app
python3 app.py
```

Then open **http://localhost:5000** in a browser, paste a URL, and click Scan.

## Notes for the report

The accuracy numbers you get from `train_model.py` (printed in the terminal
and saved to `model/results.json`) may differ slightly from the placeholder
numbers already in the report — update Table 2 in the Word doc with your
actual numbers once you've run training, so the report matches your real
results.

## Possible extensions (mentioned in report's Future Scope)

- Swap the synthetic dataset for the real UCI/Kaggle Phishing Websites Dataset
- Add a browser extension frontend instead of / in addition to the web page
- Add content-based features (page title, form fields) instead of URL-only
