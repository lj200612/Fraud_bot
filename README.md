# 智能反诈骗客服系统

这是一个基于人工智能的反诈骗智能客服系统，集成了多种先进技术来提供实时的反诈骗咨询服务。

## 主要功能

- 文本分类：使用BERT模型对用户输入的诈骗类型进行分类
- 命名实体识别：识别文本中的关键实体信息
- 语义相似度匹配：使用Sentence-BERT进行相似度计算
- 语音识别：使用DeepSpeech进行语音转文本
- 语音合成：使用Tacotron2进行文本转语音
- 知识图谱：使用Neo4j存储和查询反诈骗知识

## 项目结构

```
fraud_bot/
├── app/
│   ├── api/            # API接口
│   ├── models/         # 模型相关代码
│   ├── services/       # 业务逻辑
│   └── utils/          # 工具函数
├── config/             # 配置文件
├── data/              # 数据文件
├── models/            # 预训练模型
└── tests/             # 测试文件
```

## 安装说明

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
复制 `.env.example` 到 `.env` 并填写必要的配置信息

## 使用说明

1. 启动服务：
```bash
uvicorn app.main:app --reload
```

2. 访问API文档：
打开浏览器访问 `http://localhost:8000/docs` 