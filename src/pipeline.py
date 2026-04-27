from collections import defaultdict
from src.retriever import HybridRetriever
from src.reranker import ReRanker


class Pipeline:
    def __init__(self):
        self.retriever = HybridRetriever("data/processed/corpus.json")
        self.reranker = ReRanker()

    def run(self, query):
        docs = self.retriever.search(query, top_k=20)

        # 🔥 Rerank
        docs = self.reranker.rerank(query, docs)

        score_map = defaultdict(float)

        # Use ONLY top reranked docs
        for doc in docs[:10]:
            for code in doc["codes"]:
                score_map[code] += doc["rerank_score"]

        # 🔥 Domain boost (huge impact)
        if "cement" in query.lower():
            boost = ["IS 269", "IS 8112", "IS 12269", "IS 1489", "IS 455"]
            for b in boost:
                score_map[b] += 100

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        return [c for c, _ in ranked[:5]]