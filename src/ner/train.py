#!/usr/bin/python3
# -*- coding:utf-8 -*-

import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .util.progressbar import ProgressBar
from .util.vocab import Vocab
from .util.embed import get_word2vec
from .util.data import NerDataset, NerDataLoader
from .util.merge import merge_tag_list


class NerTrain:
    def __init__(self, config):
        self.train_config = config["train"]
        self.model_config = config["model"]

    def train(self):
        self.setup_seed(self.train_config["seed"])
        if self.train_config["cuda"] and \
                torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("在GPU上训练模型")
        else:
            device = torch.device("cpu")
            logging.info("在CPU上训练模型")

        train_file = args.train
        dev_file = args.dev
        test_file = args.test
        # 加载词表
        delimiter = args.delimiter
        vocab = Vocab(pad="<pad>", unk="<unk>", nlabel=nlabel)
        if args.do_train:
            vocab.build_vocab(train_file, dev_file, test_file,
                            delimiter=delimiter, count=0)
            vocab.save_vocab(args.output_word, args.output_label)
        else:
            vocab.load_vocab(args.output_word, args.output_label)

        config['word_size'] = vocab.get_word_size()
        config['labels_size'] = []
        for i in range(nlabel):
            config['labels_size'].append(vocab.get_label_size(nlabel=i))

        # 数据处理
        train_data = NerDataset(train_file, vocab, delimiter=delimiter, nlabel=nlabel)
        dev_data = NerDataset(dev_file, vocab, delimiter=delimiter, nlabel=nlabel)
        test_data = NerDataset(test_file, vocab, delimiter=delimiter, nlabel=nlabel)

        train_loader = NerDataLoader(train_data, args.batch_size, shuffle=True, nlabel=nlabel)
        dev_loader = NerDataLoader(dev_data, 1, shuffle=False, nlabel=nlabel)
        test_loader = NerDataLoader(test_data, 1, shuffle=False, nlabel=nlabel)

        # 构建word2vec
        word2vec = get_word2vec(args.embedding, vocab, config['word_dim'])
        config['word2vec'] = word2vec

        # 构建模型
        logging.info(config)
        model_name = config['name']
        if model_name == "BiLSTM":
            from model import BiLSTM
            model = BiLSTM(config)
        elif model_name == "BiLSTM-CRF":
            from model import BiLSTM_CRF
            model = BiLSTM_CRF(config)
        elif model_name == "Transformer-CRF":
            from model import Transformer_CRF
            model = Transformer_CRF(config)
        else:
            raise RuntimeError("没有对应的模型")

        if args.checkpoint and \
                os.path.exists(args.output_model):
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_state_dict(torch.load(args.output_model,
                                            map_location=device_name))

        model = model.to(device)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        writer = SummaryWriter(args.output_summary)
        if args.do_train:
            train(args.epoch)
        test()
        writer.close()

    def dev(self):
        pass

    def test(self):
        pass

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True