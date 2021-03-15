#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
from torch import tensor
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


class NerBertDataLoader(DataLoader):
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
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        word = pad_sequence([x[0] for x in data], batch_first=True,
                            padding_value=0)
        all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*data))
        max_len = max(all_lens).item()
        all_input_ids = all_input_ids[:, :max_len]
        all_attention_mask = all_attention_mask[:, :max_len]
        all_token_type_ids = all_token_type_ids[:, :max_len]
        all_labels = all_labels[:,:max_len]
        return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens


class NerBertDataset(Dataset):
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
        datas = self._load_conll_file(data_file, delimiter)
        self._data = self._to_feature(datas, vocab)

    def _load_conll_file(self, data_file, delimiter='\t'):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            vocab(Vocab): 词典类
            delimiter(str): 分隔符
        Returns: 无
        """
        datas = list()
        word_list = list()
        label_list = list()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.rstrip()
                if line:
                    word, label = line.split(delimiter)
                    word_list.append(word)
                    label_list.append(label)
                else:
                    if word_list:
                        # data = [torch.tensor(word_id_list), torch.tensor(label_id_list)]
                        datas.append([word_list, label_list])
                        word_list = list()
                        label_list = list()
        return datas
    
    def _to_features(datas, vocab, max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        features = []
        for (ex_index, data) in enumerate(datas):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(datas))
            tokens = tokenizer.tokenize(datas[0])
            label_ids = [vocab.get_label_id[x] for x in datas[1]]
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            O_id = vocab.get_label_id('O')
            tokens += [sep_token]
            label_ids += [O_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [O_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [O_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(label_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = torch.tensor(input_mask, dtype=torch.long)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                        segment_ids=segment_ids, label_ids=label_ids))
        return features

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
