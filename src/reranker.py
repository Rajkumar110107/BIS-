from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs):
        pairs = [(query, doc["text"]) for doc in docs]

        scores = self.model.predict(pairs)

        # Attach score to each doc
        for i, doc in enumerate(docs):
            doc["rerank_score"] = float(scores[i])

        # Sort by score
        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

        return docs