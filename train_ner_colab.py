#!/usr/bin/env python
# coding: utf-8

# # 优化版中文诈骗场景 NER 微调训练脚本 - Colab版
# 本脚本可被复制到Google Colab中的不同单元格运行。
# 每个区域用 #%% 分隔，表示一个Colab单元格。

#%% [markdown]
# ## 1. 环境准备
# 首先安装和导入必要的库

#%%
# 安装必要的库
!pip install transformers datasets seaborn

#%%
# 导入库
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import logging
from collections import Counter
from google.colab import drive
from google.colab import files
import requests
import io
import zipfile
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'使用设备: {device}')

#%% [markdown]
# ## 2. 数据加载
# 在Colab环境中可以选择不同的方式加载数据

#%%
# 检测是否在Colab环境中
IN_COLAB = 'google.colab' in str(get_ipython())
if IN_COLAB:
    logger.info('检测到Google Colab环境')

    # 挂载Google Drive (如果需要)
    MOUNT_DRIVE = False  # 设置为True以挂载Drive
    if MOUNT_DRIVE:
        logger.info('正在挂载Google Drive...')
        drive.mount('/content/drive')
        logger.info('Google Drive挂载成功')

#%%
# 生成示例数据函数
def generate_sample_data():
    """生成示例数据用于演示"""
    texts = [
        "恭喜您中奖了50000元，请点击http://prize.com领取奖金",
        "中奖信息已发放，请联系李经理，电话：13912345678",
        "您是本月幸运用户，奖金将汇入账号6222000000000000",
        "我是警官张三，您涉嫌洗钱，请拨打110配合调查",
        "公安局紧急通知，请于明天上午10点到公安局办理",
        "您身份证涉及案件需核实账户，电话：13888888888",
        "刷单返佣日入500，返现到账户1234567890123456",
        "兼职平台要先支付押金，收款账号：6222333344445555",
        "刷单任务完成后未收到返现，请联系客服",
        "理财平台年化收益18%，客服：李经理，电话：13999999999"
    ]
    
    labels = [
        "O O O B-MONEY I-MONEY O O B-LINK I-LINK I-LINK I-LINK O",
        "O O O O O B-NAME I-NAME O B-PHONE I-PHONE I-PHONE I-PHONE I-PHONE I-PHONE",
        "O O O O O O O O O B-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT",
        "O O B-NAME I-NAME O O O O O B-PHONE I-PHONE I-PHONE O O O",
        "O O O O O O B-TIME I-TIME I-TIME O B-LOC O O",
        "O O O O O O O O O B-PHONE I-PHONE I-PHONE I-PHONE I-PHONE I-PHONE",
        "O O O O B-MONEY O O O B-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT",
        "O O O O O O O O O B-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT I-ACCOUNT",
        "O O O O O O O O O O",
        "O O O O O O B-NAME I-NAME O B-PHONE I-PHONE I-PHONE I-PHONE I-PHONE I-PHONE"
    ]
    
    return pd.DataFrame({"text": texts, "labels": labels})

#%%
# 选择数据加载方式
def load_data_in_colab():
    """在Colab环境中加载数据的多种方式"""
    
    upload_option = input("选择数据加载方式：\n1. 上传CSV文件\n2. 从GitHub获取\n3. 使用样例数据\n请输入序号 (默认3): ") or "3"
    
    if upload_option == '1':
        logger.info('请上传数据文件 (CSV格式)...')
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        logger.info(f'已上传文件: {file_name}')
        df = pd.read_csv(io.BytesIO(uploaded[file_name]), encoding='utf-8')
        
    elif upload_option == '2':
        # 从GitHub获取数据
        repo_url = input("输入GitHub仓库URL (默认使用fraud_bot仓库): ") or "https://github.com/user/fraud_bot"
        branch = input("输入分支名 (默认main): ") or "main"
        file_path = input("输入CSV文件路径 (默认data/ner_training_data.csv): ") or "data/ner_training_data.csv"
        
        raw_url = f"{repo_url.rstrip('/')}/raw/{branch}/{file_path}"
        logger.info(f'正在从GitHub下载数据: {raw_url}')
        
        try:
            df = pd.read_csv(raw_url, encoding='utf-8')
            logger.info('数据下载成功')
        except Exception as e:
            logger.error(f'从GitHub下载失败: {str(e)}')
            logger.info('使用样例数据...')
            df = generate_sample_data()
    else:
        # 使用样例数据
        logger.info('使用样例数据...')
        df = generate_sample_data()
        
    return df

#%%
# 开始加载数据
if IN_COLAB:
    try:
        df = load_data_in_colab()
    except Exception as e:
        logger.error(f'加载数据异常: {str(e)}')
        logger.info('使用样例数据...')
        df = generate_sample_data()
else:
    # 非Colab环境
    try:
        df = pd.read_csv('data/ner_training_data.csv', encoding='utf-8')
        logger.info('从本地路径加载数据')
    except Exception as e:
        logger.error(f'加载数据失败: {str(e)}')
        raise

#%% [markdown]
# ## 3. 数据预处理
# 处理数据并可视化数据分布

#%%
# 检查并处理缺失值
if df['labels'].isnull().any():
    logger.warning(f'发现 {df["labels"].isnull().sum()} 个缺失值，将被替换为空字符串')
    df['labels'] = df['labels'].fillna('')

# 数据预处理
        df['labels'] = df['labels'].apply(lambda x: x.split() if isinstance(x, str) else [])
        
# 移除空标签的行
df = df[df['labels'].apply(len) > 0]

logger.info(f'成功加载数据，共 {len(df)} 条有效样本')

# 显示数据样例
print('\n数据样例：')
print(df.head())

#%%
# 可视化数据分布
all_labels = [label for labels in df['labels'] for label in labels]
label_counts = pd.Series(all_labels).value_counts()
print('\n标签分布：')
print(label_counts)

# 实体类型分布可视化
plt.figure(figsize=(12, 6))
entity_types = [label.split('-')[1] for label in label_counts.index if label != 'O']
entity_counts = {}

for entity in set(entity_types):
    b_count = label_counts.get(f'B-{entity}', 0)
    i_count = label_counts.get(f'I-{entity}', 0)
    entity_counts[entity] = b_count + i_count

entity_df = pd.DataFrame({'实体类型': list(entity_counts.keys()), 
                          '数量': list(entity_counts.values())})
entity_df = entity_df.sort_values('数量', ascending=False)

sns.barplot(x='实体类型', y='数量', data=entity_df)
plt.title('各实体类型数量分布')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 计算类别权重
label_weights = {label: 1.0 / count for label, count in label_counts.items()}
max_weight = max(label_weights.values())
label_weights = {label: weight/max_weight for label, weight in label_weights.items()}

print('\n类别权重：')
for label, weight in label_weights.items():
    print(f'{label}: {weight:.2f}')

#%% [markdown]
# ## 4. 模型训练准备
# 定义标签列表，处理数据集

#%%
# 定义标签列表
label_list = ["O", "B-MONEY", "I-MONEY", "B-ACCOUNT", "I-ACCOUNT", "B-LINK", "I-LINK", 
             "B-TIME", "I-TIME", "B-LOC", "I-LOC", "B-PHONE", "I-PHONE", "B-NAME", "I-NAME"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# 标签转为id
df['label_ids'] = df['labels'].apply(lambda x: [label2id[l] for l in x])

# 划分训练/验证集
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'].apply(lambda x: x[0]))
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}')

#%%
# 分词器
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=128,  # 增加最大长度
        is_split_into_words=False
    )
    
    labels = []
    for i, label in enumerate(examples['label_ids']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 处理数据集
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

logger.info('数据集处理完成')

#%% [markdown]
# ## 5. 模型定义与训练
# 定义模型、训练参数和评估函数，开始训练

#%%
# 定义模型
    model = BertForTokenClassification.from_pretrained(
    'bert-base-chinese', 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
# 训练参数
model_name = "bert_ner_antifraud"
output_dir = f'./{model_name}'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,  # Colab环境下适当减少轮数
    per_device_train_batch_size=16,  # 增大批量
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',  # 使用F1作为最佳模型指标
    greater_is_better=True,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    label_smoothing_factor=0.1,  # 添加标签平滑
    fp16=True,  # 使用混合精度训练加速
    report_to="none"  # 避免报告到wandb等
    )
    
#%%
# 定义评估指标
def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 只计算非-100的标签
    true_labels = []
    true_preds = []
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:
                true_labels.append(l)
                true_preds.append(p)
    
    # 计算多个指标
    metrics = {
        'accuracy': accuracy_score(true_labels, true_preds),
        'f1': f1_score(true_labels, true_preds, average='macro'),
        'precision': precision_score(true_labels, true_preds, average='macro'),
        'recall': recall_score(true_labels, true_preds, average='macro')
    }
    
    # 添加每个类别的F1分数
    for label in label_list:
        label_id = label2id[label]
        metrics[f'f1_{label}'] = f1_score(
            [1 if l == label_id else 0 for l in true_labels],
            [1 if p == label_id else 0 for p in true_preds],
            average='binary'
        )
    
    return metrics

#%%
# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

logger.info('开始训练...')
trainer.train()

# 保存模型和分词器
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f'模型已保存到 {output_dir}')

#%% [markdown]
# ## 6. 评估与测试
# 评估模型性能，可视化评估结果

#%%
# 进行评估
eval_results = trainer.evaluate()
print("\n评估结果:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
        
# 绘制评估指标可视化
metrics_df = pd.DataFrame({
    '指标': list(eval_results.keys()),
    '得分': list(eval_results.values())
})
metrics_df = metrics_df[~metrics_df['指标'].str.contains('f1_')]  # 过滤掉各类别的F1
metrics_df = metrics_df[metrics_df['指标'].str.contains('eval_')]  # 只保留eval开头的指标
metrics_df['指标'] = metrics_df['指标'].str.replace('eval_', '')

plt.figure(figsize=(10, 6))
sns.barplot(x='指标', y='得分', data=metrics_df)
plt.title('评估结果')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#%%
# 测试函数
def predict_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    pred_labels = [id2label[p.item()] for p in predictions[0]]
    
    # 提取实体
    entities = []
    current_entity = {"text": "", "type": "", "start": 0}
    
    for i, (token, label) in enumerate(zip(tokens, pred_labels)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label.startswith("B-"):
            if current_entity["text"]:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "type": label[2:],
                "start": i
            }
        elif label.startswith("I-"):
            if current_entity["text"]:
                current_entity["text"] += token
        else:
            if current_entity["text"]:
                entities.append(current_entity)
                current_entity = {"text": "", "type": "", "start": 0}
    
    if current_entity["text"]:
        entities.append(current_entity)
        
    # 后处理：处理BERT分词导致的##前缀
    processed_entities = []
    for entity in entities:
        entity["text"] = re.sub(r'\s*##', '', entity["text"])
        processed_entities.append(entity)
        
    return processed_entities

#%%
    # 测试样例
    test_texts = [
        "恭喜您中奖了50000元，请点击http://prize.com领取奖金",
        "中奖信息已发放，请联系李经理，电话：13912345678",
        "您是本月幸运用户，奖金将汇入账号6222000000000000",
        "我是警官张三，您涉嫌洗钱，请拨打110配合调查",
        "公安局紧急通知，请于明天上午10点到公安局办理",
    "您身份证涉及案件需核实账户，电话：13888888888",
    "刷单返佣日入500，返现到账户1234567890123456"
]

for text in test_texts:
    entities = predict_entities(text)
    print(f'\n测试文本: {text}')
    print('识别出的实体:')
    for entity in entities:
        print(f"类型: {entity['type']}, 文本: {entity['text']}")

#%% [markdown]
# ## 7. 模型导出
# 导出训练好的模型，提供模型使用示例代码

#%%
# 如果在Colab环境中，导出模型到Drive或下载
if IN_COLAB:
    # 导出方式1: 打包下载
    def package_model():
        # 创建一个zip文件
        logger.info("正在打包模型...")
        model_zip = f"{model_name}.zip"
        with zipfile.ZipFile(model_zip, 'w') as zipf:
            # 添加模型文件夹下的所有文件
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, os.path.dirname(output_dir)))
        
        # 下载zip文件
        logger.info(f"打包完成，正在下载 {model_zip}...")
        files.download(model_zip)
    
    # 导出方式2: 保存到Google Drive
    def save_to_drive():
        if not os.path.exists('/content/drive'):
            logger.warning("Google Drive未挂载，请先挂载Drive")
            return
        
        drive_path = '/content/drive/MyDrive/models/fraud_bot'
        os.makedirs(drive_path, exist_ok=True)
        
        # 复制模型文件到Drive
        logger.info(f"正在将模型保存到Google Drive: {drive_path}")
        !cp -r {output_dir}/* {drive_path}/
        logger.info(f"模型已保存到Google Drive: {drive_path}")
    
    print("\n选择模型导出方式:")
    print("1. 打包下载")
    print("2. 保存到Google Drive (需要已挂载)")
    export_option = input("请输入选项 (默认1): ") or "1"
    
    if export_option == "1":
        package_model()
    elif export_option == "2":
        save_to_drive()

#%%
# 模型推理示例代码
print("\n以下是模型推理示例代码:")
print("""
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
import re

# 加载模型和分词器
model_path = "bert_ner_antifraud"  # 模型保存路径
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_entities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = model.config.id2label
    pred_labels = [id2label[p.item()] for p in predictions[0]]
        
        # 提取实体
        entities = []
        current_entity = {"text": "", "type": "", "start": 0}
        
    for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            if label.startswith("B-"):
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {
                "text": token,
                    "type": label[2:],
                    "start": i
                }
        elif label.startswith("I-"):
            if current_entity["text"]:
                current_entity["text"] += token
        else:
                if current_entity["text"]:
                    entities.append(current_entity)
                    current_entity = {"text": "", "type": "", "start": 0}
        
        if current_entity["text"]:
            entities.append(current_entity)
        
    # 后处理
    processed_entities = []
        for entity in entities:
        entity["text"] = re.sub(r'\s*##', '', entity["text"])
        processed_entities.append(entity)
        
    return processed_entities

# 使用示例
text = "恭喜您中奖了50000元，请点击http://prize.com领取奖金"
entities = predict_entities(text)
for entity in entities:
    print(f"类型: {entity['type']}, 文本: {entity['text']}")
""") 
