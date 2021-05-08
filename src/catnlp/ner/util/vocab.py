# -*- coding: utf-8 -*-


class Vocab:
    """
    词典类
    """
    def __init__(self, pad="<pad>", unk="<unk>"):
        # word - id
        self._word2id = dict()
        self._id2word = dict()
        self._word_size = 0
        # label - id
        self._label2id = dict()
        self._id2label = dict()
        self._label_size = 0
        self._word2count = dict()
        self._pad = pad
        self._unk = unk

    def build_vocab(self, train, dev, delimiter="\t", count=0):
        """
        构建词典
        Args:
            train(str): 训练集
            dev(str): 验证集
            delimiter(str): 分隔符
            count(int): 阈值
        Returns: 无
        """
        self._word2id[self._pad] = self._word_size
        self._id2word[self._word_size] = self._pad
        self._word_size += 1
        self._word2id[self._unk] = self._word_size
        self._id2word[self._word_size] = self._unk
        self._word_size += 1

        self._read_file(train, delimiter, count)
        self._read_file(dev, delimiter, count)

    def _read_file(self, data_file, delimiter, count):
        """
        读取数据集
        Args:
            data_file(str): 文件路径
            delimiter(str): 分隔符
            count(int): 阈值
        :return: 无
        """
        if not data_file:
            return

        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.rstrip()
                if not line:
                    continue
                word, label = line.split(delimiter)

                # 判断标签是否在标签词典里
                if label not in self._label2id:
                    self._label2id[label] = self._label_size
                    self._id2label[self._label_size] = label
                    self._label_size += 1

                # 判断单词是否在单词统计词典里
                if word not in self._word2count:
                    self._word2count[word] = 1
                else:
                    self._word2count[word] += 1

                # 判断是否把单词添加到单词词典
                if word not in self._word2id \
                        and self._word2count[word] > count:
                    self._word2id[word] = self._word_size
                    self._id2word[self._word_size] = word
                    self._word_size += 1

    def save_vocab(self, word_file, label_file):
        """
        保存词典文件
        Args:
            word_file(str): 单词文件路径
            label_file: 标签文件路径
        :Returns: 无
        """
        with open(word_file, 'w', encoding='utf-8') as wf, \
                open(label_file, 'w', encoding='utf-8') as lf:
            for i in range(self._word_size):
                wf.write(self._id2word[i] + "\n")

            for i in range(self._label_size):
                lf.write(self._id2label[i] + "\n")

    def load_vocab(self, word_file, label_file):
        """
        加载词典文件
        Args:
            word_file(str): 单词文件路径
            label_file(str): 标签文件路径
        Returns: 无
        """
        if not word_file or not label_file:
            return

        with open(word_file, 'r', encoding='utf-8') as wf, \
                open(label_file, 'r', encoding='utf-8') as lf:
            for line in wf:
                word = line.rstrip()
                if not word:
                    continue
                self._word2id[word] = self._word_size
                self._id2word[self._word_size] = word
                self._word_size += 1

            for line in lf:
                label = line.rstrip()
                if not label:
                    continue
                self._label2id[label] = self._label_size
                self._id2label[self._label_size] = label
                self._label_size += 1

    def get_word2id(self):
        """
        获得单词索引映射词典
        Returns: 单词索引词典
        """
        return self._word2id

    def get_label2id(self):
        """
        获得标签索引映射词典
        Returns: 标签索引词典
        """
        return self._label2id

    def get_word(self, idx):
        """
        获得单词
        Args:
            idx: 索引
        Returns: 单词
        """
        return self._id2word.get(idx, self._unk)
    
    def get_word_id(self, word):
        """
        获得单词索引
        Args:
            word: 单词
        Returns: 单词索引
        """
        return self._word2id.get(word, 1)
    
    def get_label(self, idx):
        """
        获得标签
        Args:
            idx: 索引
        Returns: 标签
        """
        return self._id2label.get(idx)
    
    def get_label_id(self, label):
        """
        获得标签索引
        Args:
            label: 标签
        Returns: 标签索引
        """
        return self._label2id.get(label)
    
    def get_word_size(self):
        """
        获得单词词典长度
        Returns: 单词词典长度
        """
        return self._word_size
    
    def get_label_size(self):
        """
        获得标签词典长度
        Return: 标签词典长度
        """
        return self._label_size
