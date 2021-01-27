#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class NerDataLoader(DataLoader):
    """
    数据加载器
    """
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        super(NerDataLoader, self).__init__(dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            collate_fn=self._collate_fn,
                                            drop_last=drop_last)

    def _collate_fn(self, data):
        word = pad_sequence([x[0] for x in data], batch_first=True,
                            padding_value=0)
        label = pad_sequence([x[-1] for x in data], batch_first=True,
                             padding_value=0)
        return word, label


class NerDataset(Dataset):
    """
    数据集类
    """
    def __init__(self, data_file, vocab, delimiter="\t"):
        """
        初始化数据集类
        Args:
            data_file(str): 数据集文件路径
            vocab(Vocab): 词典类
            delimiter(str): 分隔符
        Returns: 无
        """
        self._data = list()
        self._load_data_file(data_file, vocab, delimiter)

    def _load_data_file(self, data_file, vocab, delimiter):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            vocab(Vocab): 词典类
            delimiter(str): 分隔符
        Returns: 无
        """
        word_id_list = list()
        label_id_list = list()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.rstrip()
                if line:
                    word, label = line.split(delimiter)
                    word_id = vocab.get_word_id(word)
                    word_id_list.append(word_id)
                    label_id = vocab.get_label_id(label)
                    label_id_list.append(label_id)
                else:
                    if word_id_list:
                        data = [torch.tensor(word_id_list), torch.tensor(label_id_list)]
                        self._data.append(data)
                        word_id_list = list()
                        label_id_list = list()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
