from src.retriever import HybridRetriever
from src.reranker import ReRanker
import re
from collections import defaultdict

class Pipeline:
    def __init__(self):
        self.retriever = HybridRetriever("data/processed/corpus.json")
        self.reranker = ReRanker()

    def normalize_code(self, code):
        return re.sub(r':\d{4}', '', code).strip()

    def tokenize(self, text):
        return set(re.findall(r"[a-zA-Z]+", text.lower()))

    def run(self, query):
        docs = self.retriever.search(query, top_k=25)
        top_docs = self.reranker.rerank(query, docs, top_k=10)

        q_tokens = self.tokenize(query)

        score = defaultdict(float)
        evidence = defaultdict(int)

        for rank, d in enumerate(top_docs):
            d_tokens = self.tokenize(d["text"])

            overlap = len(q_tokens & d_tokens)
            rank_w = 1.0 / (rank + 1)

            # 🔥 dynamic keyword boost
            keyword_boost = 0
            if "cement" in q_tokens:
                if "cement" in d_tokens:
                    keyword_boost += 2
                if "grade" in d_tokens or "portland" in d_tokens:
                    keyword_boost += 1

            for code in d["codes"]:
                c = self.normalize_code(code)

                score[c] += (overlap * 2 + 1 + keyword_boost) * rank_w
                evidence[c] += 1

        final = []
        for c in score:
            s = score[c] + (evidence[c] * 0.7)  # consistency boost
            final.append((c, s))

        final.sort(key=lambda x: x[1], reverse=True)

        return [c for c, _ in final[:5]]