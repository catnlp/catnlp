#!/usr/bin/python3
# -*- coding:utf-8 -*-

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
from .util.merge import merge_tag_lists


class NerTrain:
    def __init__(self, config):
        train_cfg = config["train"]
        model_cfg = config["model"]
        self.setup_seed(train_cfg["seed"])
        if self.train_config["cuda"] and \
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
        vocab.build_vocab(train_file, dev_file, test_file,
                          delimiter=delimiter, count=0)
        output_dir = Path(train_cfg["output"])
        word_file = output_dir / "word.txt"
        label_file = output_dir / "label.txt"
        vocab.save_vocab(word_file, label_file)

        model_cfg['word_size'] = vocab.get_word_size()
        model_cfg['labels_size'] = vocab.get_label_size()

        # 数据处理
        train_data = NerDataset(train_file, vocab, delimiter=delimiter)
        dev_data = NerDataset(dev_file, vocab, delimiter=delimiter)
        test_data = NerDataset(test_file, vocab, delimiter=delimiter)

        self.train_loader = NerDataLoader(train_data, train_cfg["batch"], shuffle=True)
        self.dev_loader = NerDataLoader(dev_data, 1, shuffle=False)
        self.test_loader = NerDataLoader(test_data, 1, shuffle=False)

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
        results, _, golds_list, preds_list = self.generate_result(loader)
        f1 = self.get_f1(golds_list, preds_list)
        loss = self.get_loss(loader)
        return f1, loss


    def generate_result(self, loader):
        results = list()
        pred_results = list()
        golds_list = list()
        preds_list = list()
        for batch in loader:
            word_batch, gold_ids_batch = batch
            word_batch = word_batch.to(self.device)
            pred_ids_list, len_list = self.model(word_batch)
            word_list, gold_list, pred_list, pred_result = self.recover_id_to_tag(
                word_batch.tolist(),
                gold_ids_batch,
                pred_ids_list,
                len_list
            )
            golds_list.append(gold_list)
            preds_list.append(pred_list)
            for word, gold, pred in zip(word_list, gold_list, pred_list):
                results.append(f"{word}\t{gold}\t{pred}")
            results.append("")

            for pred in zip(word_list, *pred_result):
                pred_results.append(pred)
            pred_results.append(list())
        return results, pred_results, golds_list, preds_list


    def get_loss(self, loader):
        loss = 0.0
        for batch in loader:
            word_batch, label_batch = batch
            word_batch = word_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            loss_batch = self.model.calculate_loss(word_batch, label_batch)
            loss += loss_batch.item()
        return loss


    def recover_id_to_tag(self, word_id_list, gold_ids_list, pred_ids_list, len_list, nlabel=1):
        word_tag_list = list()
        gold_tags_list = [[] for _ in range(nlabel)]
        pred_tags_list = [[] for _ in range(nlabel)]
        for word_id, seq_len in \
                zip(word_id_list, len_list):
            for idx in range(seq_len):
                word_tag_list.append(self.vocab.get_word(word_id[idx]))

        for i, (gold_id_list, pred_id_list) in \
                enumerate(zip(gold_ids_list, pred_ids_list)):
            for gold_id, pred_id, seq_len in \
                    zip(gold_id_list.tolist(), pred_id_list.tolist(), len_list):
                for j in range(seq_len):
                    gold_tags_list[i].append(self, self.vocab.get_label(gold_id[j]))
                    pred_tags_list[i].append(self, self.vocab.get_label(pred_id[j]))

        # todo 多列情况
        # gold_tag_list = merge_tag_list(list(reversed(gold_tags_list)))
        # pred_tag_list = merge_tag_list(list(reversed(pred_tags_list)))
        gold_tag_list = gold_tags_list[0]
        pred_tag_list = pred_tags_list[0]

        return word_tag_list, gold_tag_list, pred_tag_list, pred_tags_list

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
