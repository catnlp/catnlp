#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .util.progressbar import ProgressBar
from .util.vocab import Vocab
from .util.embed import get_embed
from .util.data import NerDataset, NerDataLoader
from .util.score import get_f1


class NerTrain:
    def __init__(self, config):
        logging.info(config)
        train_cfg = config["train"]
        model_cfg = config["model"]
        self.setup_seed(train_cfg["seed"])
        if train_cfg["cuda"] and \
                torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("在GPU上训练模型")
        else:
            self.device = torch.device("cpu")
            logging.info("在CPU上训练模型")
        input_dir = Path(train_cfg["input"])
        train_file = input_dir / "train.txt"
        dev_file = input_dir / "dev.txt"
        test_file = input_dir / "test.txt"
        # 加载词表
        delimiter = train_cfg["delimiter"]
        vocab = Vocab(pad="<pad>", unk="<unk>")
        vocab.build_vocab(train_file, dev_file,
                          delimiter=delimiter, count=0)
        output_dir = Path(train_cfg["output"])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        word_file = output_dir / "word.txt"
        label_file = output_dir / "label.txt"
        vocab.save_vocab(word_file, label_file)

        model_cfg['word_size'] = vocab.get_word_size()
        model_cfg['label_size'] = vocab.get_label_size()

        # 数据处理
        train_data = NerDataset(train_file, vocab, delimiter=delimiter)
        dev_data = NerDataset(dev_file, vocab, delimiter=delimiter)

        self.train_loader = NerDataLoader(train_data, train_cfg["batch"], shuffle=True, drop_last=True)
        self.dev_loader = NerDataLoader(dev_data, train_cfg["batch"], shuffle=False, drop_last=False)

        # 构建word2vec
        model_cfg["embed"] = \
            get_embed(train_cfg["embedding"], vocab, model_cfg["word_dim"])

        # 构建模型
        logging.info(model_cfg)
        model_name = config["name"].lower()
        if model_name == "bilstm":
            from .model import BiLSTM
            model = BiLSTM(model_cfg)
        elif model_name == "bilstm_crf":
            from .model import BiLSTM_CRF
            model = BiLSTM_CRF(model_cfg)
        else:
            raise RuntimeError(f"没有对应的模型: {config['name']}")

        self.model = model.to(self.device)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        summary_dir = output_dir / "summary/"
        self.writer = SummaryWriter(summary_dir)
        self.vocab = vocab
        self.train_cfg = train_cfg
        self.output_model = output_dir / f"{model_name}.pt"

    def train(self):
        logging.info("开始训练")
        optim_name = self.train_cfg["optim"].lower()
        lr = self.train_cfg["lr"]
        if optim_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optim_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise RuntimeError("当前优化器不支持")
        scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=5)

        best_f1 = 0.0
        for epoch in range(self.train_cfg["epoch"]):
            self.model.train()
            bar = ProgressBar(n_total=len(self.train_loader), desc='Training')
            for step, batch in enumerate(self.train_loader):
                word_batch, label_batch = batch
                word_batch = word_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.model.calculate_loss(word_batch, label_batch)
                loss.backward()
                optimizer.step()
                bar(step=step, info={'loss': loss.item()})
                # if step % 5 == 4:
                #     f1, dloss = dev(dev_loader)
                #     print("f1: ", f1)
                #     print("loss: ", dloss)
                #     model.train()

            train_f1, train_loss = self.dev(self.train_loader)
            dev_f1, dev_loss = self.dev(self.dev_loader)
            print()
            logging.info("Epoch: {} 验证集F1: {}".format(epoch + 1, dev_f1))
            self.writer.add_scalars("f1",
                            {
                                "train": round(100 * train_f1, 2),
                                "dev": round(100 * dev_f1, 2)
                            },
                            epoch + 1)
            self.writer.add_scalars('loss',
                            {
                                "train": round(
                                    train_loss, 2),
                                "dev": round(dev_loss,
                                                2)
                            },
                            epoch + 1)

            if dev_f1 >= best_f1:
                best_f1 = dev_f1
                torch.save(self.model.state_dict(), self.output_model)

            scheduler.step(100 * dev_f1)

        logging.info(f"训练完成，best f1: {best_f1}")

    def dev(self, loader):
        self.model.eval()
        gold_lists, pred_lists = self.generate_result(loader)
        f1 = get_f1(gold_lists, pred_lists, self.train_cfg["tag_format"])
        loss = self.get_loss(loader)
        return f1, loss

    def generate_result(self, loader):
        gold_list = list()
        pred_list = list()
        for batch in loader:
            word_batch, gold_ids_batch = batch
            word_batch = word_batch.to(self.device)
            pred_ids_batch, len_list_batch = self.model(word_batch)
            gold_lists_batch, pred_lists_batch = self.recover_id_to_tag(
                gold_ids_batch.tolist(),
                pred_ids_batch,
                len_list_batch
            )
            gold_list.extend(gold_lists_batch)
            pred_list.extend(pred_lists_batch)
        return gold_list, pred_list

    def get_loss(self, loader):
        loss = 0.0
        for batch in loader:
            word_batch, label_batch = batch
            word_batch = word_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            loss_batch = self.model.calculate_loss(word_batch, label_batch)
            loss += loss_batch.item()
        return loss

    def recover_id_to_tag(self, gold_ids_list, pred_ids_list, len_list):
        gold_tag_lists = list()
        pred_tag_lists = list()

        for gold_id_list, pred_id_list, seq_len in \
                zip(gold_ids_list, pred_ids_list, len_list):
            tmp_gold_list = list()
            tmp_pred_list = list()
            for i in range(seq_len):
                tmp_gold_list.append(self.vocab.get_label(gold_id_list[i]))
                tmp_pred_list.append(self.vocab.get_label(pred_id_list[i]))
            gold_tag_lists.append(tmp_gold_list)
            pred_tag_lists.append(tmp_pred_list)

        return gold_tag_lists, pred_tag_lists

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
