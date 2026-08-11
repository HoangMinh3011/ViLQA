from dataclasses import dataclass

@dataclass
class LegalQAConfig:
    
    # bm25s retriever
    TOP_K: int = 10
    
    # biencoder retriever
    
    # rerank
    
    # llm
    TEMPERATURE: float = 1.0
    TOP_P: float = 1.0