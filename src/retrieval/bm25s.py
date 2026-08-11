# pip install bm25s --break-system-packages
import bm25s
from config import LegalQAConfig

class BM25sRetriever:
    def __init__(self) -> None:
        self.top_k = LegalQAConfig.TOP_K
        
    
    def build_bm25_fast(self, nodes):
        corpus_texts = [n["text"] for n in nodes]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None)  # thay bằng tokenizer tiếng Việt nếu cần
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever

    def bm25_retrieve_fast(self, query, retriever, nodes, top_k=20):
        query_tokens = bm25s.tokenize([query])
        results, scores = retriever.retrieve(query_tokens, k=self.top_k)
        return [{**nodes[idx], "bm25_score": float(score)} for idx, score in zip(results[0], scores[0])]