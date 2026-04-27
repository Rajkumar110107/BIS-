import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

class HybridRetriever:
    def __init__(self, corpus_path):
        with open(corpus_path, "r") as f:
            self.data = json.load(f)

        self.texts = [d["text"] for d in self.data]
        self.codes = [d["codes"] for d in self.data]

        # Dense embeddings
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

        # BM25
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=20):
        # Dense search
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)

        dense_results = [self.data[i] for i in I[0]]

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

        bm25_results = [self.data[i] for i in bm25_indices]

        # Combine (simple fusion)
        combined = dense_results + bm25_results

        # Remove duplicates (based on text)
        seen = set()
        unique_results = []
        for item in combined:
            if item["text"] not in seen:
                seen.add(item["text"])
                unique_results.append(item)

        return unique_results[:top_k]