from sentence_transformers import SentenceTransformer
import torch
import os

class SimilarityService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("models", "sentence_transformer")
        
        # 加载预训练模型
        if os.path.exists(self.model_path):
            self.model = SentenceTransformer(self.model_path)
        else:
            # 使用预训练的中文模型
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        self.model.to(self.device)

    def calculate(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度"""
        # 编码文本
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            embeddings1.unsqueeze(0),
            embeddings2.unsqueeze(0)
        )
        
        return similarity.item() 