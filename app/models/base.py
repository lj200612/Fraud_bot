import torch
import os
from abc import ABC, abstractmethod
from config.settings import MODEL_PATHS

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = MODEL_PATHS.get(model_name)
        
        if not self.model_path:
            raise ValueError(f"Model path not found for {model_name}")
            
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
    
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def save_model(self):
        """保存模型"""
        pass
    
    def to_device(self, data):
        """将数据移动到指定设备"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device)
    
    def eval(self):
        """设置为评估模式"""
        self.model.eval()
    
    def train(self):
        """设置为训练模式"""
        self.model.train() 