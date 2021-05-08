## CatNLP

👋项目聚焦于NLP技术，包括不限于命名实体识别，实体关系抽取，文本相似度，实体链接等技术

### 索引

### 命名实体识别（NER）

#### BiLSTM

**运行**

```
cd src
python train.py --task=NER --train_config=data/config/ner/bilstm_crf.yaml --log_config=data/config/ner/logging.yaml
```

**训练配置**

```
type: BiLSTM
name: BiLSTM_CRF
seed: 42
train:
  cuda: True
  batch: 24
  epoch: 30
  lr: 0.01
  optim: "Adam"
  delimiter: "\t"
  tag_format: "bio"
  embed_format: "word2vec"
  input: "data/dataset/ner/zh/cluener/bio"
  output: "data/output/ner/zh/cluener"
  embedding: "data/embed/character.vec.txt"
model:
  word_dim: 100
  hidden_dim: 150
  num_layer: 3
  dropout: 0.4
```

#### BERT

**运行**

```
python train.py --task=NER --train_config=data/config/ner/bert.yaml --log_config=data/config/ner/logging.yaml
```

**训练配置**

```
type: BERT
name: BERT_Softmax
delimiter: "\t"
input: "data/dataset/ner/zh/cluener/bio"
output: "data/output/ner/zh/cluener/bert"
max_length: 128
pad_to_max_length: True
model_path: data/pretrained/bert_base_chinese
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
learning_rate: 5.0e-5
weight_decay: 0.0
num_train_epochs: 3
gradient_accumulation_steps: 1
lr_scheduler_type: linear
num_warmup_steps: 0
seed: 100
label_all_tokens: True
do_lower_case: True
task_name: ner
return_entity_level_metrics: True
```