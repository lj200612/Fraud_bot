from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import logging
from config.settings import MODEL_CONFIG, MODEL_PATHS
from app.utils.helpers import to_device

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = MODEL_PATHS["macbert_classifier"]
        self.config = MODEL_CONFIG["macbert_classifier"]
        
        logger.info(f"正在加载模型，路径: {self.model_path}")
        # 直接从本地目录加载分词器和模型（自动适配类别数）
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("模型加载完成")

    def get_fraud_types(self):
        """返回所有诈骗类型"""
        return list(self.config["id2label"].values())

    def classify(self, text: str) -> dict:
        """对输入文本进行分类"""
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = to_device(inputs, self.device)

        # 进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()

        # 获取所有类别的概率
        all_probabilities = {
            fraud_type: prob.item() 
            for fraud_type, prob in zip(self.get_fraud_types(), predictions[0])
        }

        return {
            "type": self.get_fraud_types()[predicted_class],
            "confidence": confidence,
            "all_probabilities": all_probabilities
        } 