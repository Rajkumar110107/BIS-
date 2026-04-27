import pdfplumber
import re
import json
import os

def extract_is_codes(text):
    return re.findall(r'IS\s*\d+(?::\d+)?', text)

def split_sections(text):
    parts = re.split(r'\n\s*(\d+(\.\d+)*\s+[A-Z][^\n]+)\n', text)
    chunks = []
    for i in range(0, len(parts), 3):
        block = parts[i] if i < len(parts) else ""
        if len(block.strip()) > 80:
            chunks.append(block.strip())
    return chunks

def build_corpus(pdf_dir, output_file):
    corpus = []

    for file in os.listdir(pdf_dir):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)

        try:
            with pdfplumber.open(path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"

            sections = split_sections(text)

            for sec in sections:
                codes = extract_is_codes(sec)
                if codes:
                    corpus.append({
                        "text": sec,
                        "codes": list(set(codes))
                    })

        except Exception as e:
            print(f"Error processing {file}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Corpus built with {len(corpus)} chunks")

if __name__ == "__main__":
    build_corpus("data/raw", "data/processed/corpus.json")