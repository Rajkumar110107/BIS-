import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, corpus_path):
        with open(corpus_path, "r") as f:
            self.data = json.load(f)

        self.texts = [d["text"] for d in self.data]

        # BM25
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.texts)

    def search(self, query, top_k=10):
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())

        # Embedding scores
        q_emb = self.model.encode([query])[0]
        emb_scores = np.dot(self.embeddings, q_emb)

        # Combine
        final_scores = 0.5 * bm25_scores + 0.5 * emb_scores

        idxs = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for i in idxs:
            results.append({
                "text": self.texts[i],
                "codes": self.data[i]["codes"],
                "score": float(final_scores[i])
            })

        return results