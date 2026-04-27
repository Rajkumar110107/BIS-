import os
import json
import re
import pdfplumber

RAW_PATH = "data/raw"
OUT_PATH = "data/processed/corpus.json"


def extract_codes(text):
    return list(set(re.findall(r"IS\s?\d{3,5}", text)))


def build_corpus():
    corpus = []

    for file in os.listdir(RAW_PATH):
        if not file.endswith(".pdf"):
            continue

        with pdfplumber.open(os.path.join(RAW_PATH, file)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                chunks = text.split("\n\n")

                for chunk in chunks:
                    codes = extract_codes(chunk)
                    if codes:
                        corpus.append({
                            "text": chunk,
                            "codes": codes
                        })

    os.makedirs("data/processed", exist_ok=True)

    with open(OUT_PATH, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Corpus built with {len(corpus)} chunks")


if __name__ == "__main__":
    build_corpus()