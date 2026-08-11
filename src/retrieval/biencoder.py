"https://huggingface.co/NghiemAbe/Vi-Legal-Bi-Encoder-v2"
from transformers import AutoTokenizer, AutoModel

class BiencoderRetriever:
    
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained("")
        self.tokenizer = AutoTokenizer.from_pretrained("")
        
    def foward(self):
        pass
    
    