from transformers import BertTokenizer, BertForTokenClassification
import torch
import os
import logging
from config.settings import MODEL_PATHS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = MODEL_PATHS["bert_ner"]
        
        # 定义正确的实体类型
        self.label2id = {
            'O': 0,
            'B-MONEY': 1, 'I-MONEY': 2,
            'B-ACCOUNT': 3, 'I-ACCOUNT': 4,
            'B-LINK': 5, 'I-LINK': 6,
            'B-PHONE': 7, 'I-PHONE': 8,
            'B-NAME': 9, 'I-NAME': 10,
            'B-TIME': 11, 'I-TIME': 12,
            'B-LOC': 13, 'I-LOC': 14
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 加载预训练模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 尝试加载本地微调模型
        try:
            if os.path.exists(self.model_path):
                logger.info(f"从 {self.model_path} 加载NER模型")
                self.model = BertForTokenClassification.from_pretrained(
                    self.model_path
                )
                logger.info("成功加载本地微调模型")
            else:
                logger.warning(f"路径 {self.model_path} 不存在，将使用基础模型")
                self.model = BertForTokenClassification.from_pretrained(
                    "bert-base-chinese",
                    num_labels=len(self.label2id),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
                logger.info("使用基础模型，未进行反诈NER微调")
        except Exception as e:
            logger.error(f"加载模型出错: {str(e)}")
            logger.info("回退到基础模型")
            self.model = BertForTokenClassification.from_pretrained(
                "bert-base-chinese",
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("NER服务初始化完成")

    def get_entity_types(self):
        """返回所有实体类型"""
        return list(self.label2id.keys())

    def extract(self, text: str) -> list:
        """提取文本中的命名实体"""
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            
        # 解码预测结果
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        current_entity = {"text": "", "type": "", "start": 0}
        
        for i, (token, pred) in enumerate(zip(tokens, predictions[0])):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            pred_id = pred.item()
            if pred_id in self.id2label:
                pred_type = self.id2label[pred_id]
            else:
                logger.warning(f"未知的预测ID: {pred_id}")
                continue
            
            if pred_type.startswith("B-"):
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": pred_type[2:],
                    "start": i
                }
            elif pred_type.startswith("I-"):
                if current_entity["text"] and current_entity["type"] == pred_type[2:]:
                    # 只有当前实体类型匹配时才添加
                    current_entity["text"] += token
            else:  # "O" 或其他
                if current_entity["text"]:
                    entities.append(current_entity)
                    current_entity = {"text": "", "type": "", "start": 0}
        
        if current_entity["text"]:
            entities.append(current_entity)
            
        # 后处理：合并子词标记
        merged_entities = []
        for entity in entities:
            # 处理BERT分词的"##"前缀
            if any(part.startswith("##") for part in entity["text"].split()):
                entity["text"] = entity["text"].replace(" ##", "")
            merged_entities.append(entity)
            
        return merged_entities

    def extract_with_confidence(self, text: str, threshold=0.5) -> list:
        """提取文本中的命名实体，并包含置信度"""
        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=2)
            predictions = torch.argmax(probabilities, dim=2)
            confidences = torch.max(probabilities, dim=2).values
            
        # 解码预测结果
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        current_entity = {"text": "", "type": "", "start": 0, "confidence": 0.0}
        current_conf_sum = 0.0
        current_token_count = 0
        
        for i, (token, pred, conf) in enumerate(zip(tokens, predictions[0], confidences[0])):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            pred_id = pred.item()
            conf_val = conf.item()
            
            if pred_id in self.id2label:
                pred_type = self.id2label[pred_id]
            else:
                continue
            
            if pred_type.startswith("B-") and conf_val >= threshold:
                if current_entity["text"]:
                    current_entity["confidence"] = current_conf_sum / current_token_count
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": pred_type[2:],
                    "start": i,
                    "confidence": 0.0
                }
                current_conf_sum = conf_val
                current_token_count = 1
            elif pred_type.startswith("I-") and conf_val >= threshold:
                if current_entity["text"] and current_entity["type"] == pred_type[2:]:
                    current_entity["text"] += token
                    current_conf_sum += conf_val
                    current_token_count += 1
            else:
                if current_entity["text"]:
                    current_entity["confidence"] = current_conf_sum / current_token_count
                    entities.append(current_entity)
                    current_entity = {"text": "", "type": "", "start": 0, "confidence": 0.0}
                    current_conf_sum = 0.0
                    current_token_count = 0
        
        if current_entity["text"]:
            current_entity["confidence"] = current_conf_sum / current_token_count
            entities.append(current_entity)
            
        # 后处理：合并子词标记
        merged_entities = []
        for entity in entities:
            if any(part.startswith("##") for part in entity["text"].split()):
                entity["text"] = entity["text"].replace(" ##", "")
            merged_entities.append(entity)
            
        return merged_entities 