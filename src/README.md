## CatNLP

👋👋本项目聚焦于NLP技术，包括不限于命名实体识别，实体关系抽取，文本匹配，实体链接等技术

### 索引

### 命名实体识别（NER）

#### BiLSTM

**运行**

```
cd src
python train.py --task=NER --train_config=data/config/ner/bilstm.yaml --log_config=data/config/ner/logging.yaml
```

#### BERT

**运行**

```
python train.py --task=NER --train_config=data/config/ner/bert.yaml --log_config=data/config/ner/logging.yaml
```
