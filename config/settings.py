import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
MODEL_CONFIG = {
    "macbert_classifier": {
        "model_name": "hfl/chinese-macbert-base",
        "num_labels": 8,
        "id2label": {
            "0": "中奖诈骗",
            "1": "冒充公检法",
            "2": "刷单诈骗",
            "3": "投资陷阱",
            "4": "客服诈骗",
            "5": "冒充领导",
            "6": "疫情诈骗",
            "7": "情感诈骗"
        }
    }
}

# 数据库配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# 模型路径配置
MODEL_PATHS = {
    "macbert_classifier": os.path.join("models", "macbert_classifier"),
    "bert_ner": os.path.join("models", "bert_ner"),
    "sentence_transformer": os.path.join("models", "sentence_transformer"),
    "deepspeech": os.path.join("models", "deepspeech"),
    "tacotron2": os.path.join("models", "tacotron2")
}

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true" 