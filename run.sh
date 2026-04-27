#!/bin/bash

echo "🚀 Starting BIS RAG System..."

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Step 2: Build corpus (only if not exists)
if [ ! -f "data/processed/corpus.json" ]; then
    echo "📚 Building corpus..."
    python src/ingest.py
else
    echo "✅ Corpus already exists, skipping..."
fi

# Step 3: Run inference
echo "⚡ Running inference..."
python inference.py --input sample.json --output result.json

# Step 4: Show output
echo "📄 Result:"
cat result.json

echo "✅ Done!"