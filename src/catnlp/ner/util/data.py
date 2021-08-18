# -*- coding: utf-8 -*-
import re
import copy
import json
import logging

import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class NerLstmDataLoader(DataLoader):
    """
    数据加载器
    """
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        super(NerLstmDataLoader, self).__init__(dataset,
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


class NerLstmDataset(Dataset):
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
        datas = self._load_bio_file(data_file, delimiter)
        self._data = self._to_features(datas, vocab)
    
    def _load_bio_file(self, data_file, delimiter='\t'):
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
                        datas.append([word_list, label_list])
                        word_list = list()
                        label_list = list()
        return datas

    def _to_features(self, datas, vocab):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            vocab(Vocab): 词典类
        Returns: 无
        """
        features = list()
        for (ex_index, data) in enumerate(datas):
            if ex_index < 3:
                print(data)
            word_ids = [vocab.get_word_id(x) for x in data[0]]
            label_ids = [vocab.get_label_id(x) for x in data[1]]
            word_ids = torch.tensor(word_ids, dtype=torch.long)
            label_ids = torch.tensor(label_ids, dtype=torch.long)
            features.append([word_ids, label_ids])
        return features

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class NerBertDataLoader(DataLoader):
    """
    数据加载器
    """
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        super(NerBertDataLoader, self).__init__(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                collate_fn=self._collate_fn,
                                                drop_last=drop_last)

    def _collate_fn(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features])
        all_input_mask = torch.tensor([f.input_mask for f in features])
        all_segment_ids = torch.tensor([f.segment_ids for f in features])
        all_label_ids = torch.tensor([f.label_ids for f in features])
        all_label_mask = torch.tensor([f.label_mask for f in features])
        all_input_len = torch.tensor([f.input_len for f in features])
        all_masks = torch.tensor([f.masks for f in features])
        all_seg_ids = torch.tensor([f.seg_ids for f in features])
        all_start_ids = torch.tensor([f.start_ids for f in features])
        all_end_ids = torch.tensor([f.end_ids for f in features])
        return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_mask, all_input_len, all_masks, all_seg_ids, all_start_ids, all_end_ids


class NerBertDataset(Dataset):
    """
    数据集类
    """
    def __init__(self, data_file, tokenizer, max_seq_length, file_format="bio", delimiter="\t", do_lower=False):
        """
        初始化数据集类
        Args:
            data_file(str): 数据集文件路径
            vocab(Vocab): 词典类
            delimiter(str): 分隔符
        Returns: 无
        """
        self.tokenizer = tokenizer
        self._do_lower = do_lower
        datas = self._load_file(data_file, file_format, delimiter)
        self._data = self._to_features(datas, file_format=file_format, max_seq_length=max_seq_length)
    
    def _load_file(self, data_file, file_format, delimiter):
        if file_format == "json":
            return self._load_json_file(data_file)
        elif file_format == "split":
            return self._load_split_file(data_file)
        elif file_format == "biaffine":
            return self._load_biaffine_file(data_file)
        elif file_format == "bies":
            return self._load_bies_file(data_file, delimiter)
        elif file_format == "span":
            return self._load_span_file(data_file)
        else:
            return self._load_conll_file(data_file, delimiter)


    def _load_conll_file(self, data_file, delimiter):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            delimiter(str): 分隔符
        Returns: 无
        """
        datas = list()
        word_list = list()
        tag_list = list()
        label_set = set()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.rstrip()
                if line:
                    word, tag = line.split(delimiter)
                    word_list.append(word)
                    tag_list.append(tag)
                    label_set.add(tag)
                else:
                    if word_list:
                        datas.append([word_list, tag_list])
                        word_list = list()
                        tag_list = list()
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.contents = list()
        self.offset_lists = list()
        return datas
    
    def _load_bies_file(self, data_file, delimiter):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            delimiter(str): 分隔符
        Returns: 无
        """
        datas = list()
        word_list = list()
        tag_list = list()
        label_set = {"O"}
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = line.rstrip()
                if line:
                    word, tag = line.split(delimiter)
                    word_list.append(word)
                    tag_list.append(tag)
                    if len(tag) > 1:
                        tag_name = tag[2:]
                        label_set.add(f"B-{tag_name}")
                        label_set.add(f"I-{tag_name}")
                        label_set.add(f"E-{tag_name}")
                        label_set.add(f"S-{tag_name}")
                else:
                    if word_list:
                        datas.append([word_list, tag_list])
                        word_list = list()
                        tag_list = list()
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.contents = list()
        self.offset_lists = list()
        return datas
    
    def _load_json_file(self, data_file):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            delimiter(str): 分隔符
        Returns: 无
        """
        datas = list()
        label_set = set(["O"])
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = json.loads(line)
                if not line:
                    continue
                text = line["text"]
                entities = line["labels"]
                tag_list = ["O"] * len(text)
                for entity in entities:
                    start, end, tag = entity
                    label_set.add(f"B-{tag}")
                    label_set.add(f"I-{tag}")
                    tag_list[start] = f"B-{tag}"
                    for i in range(start+1, end):
                        tag_list[i] = f"I-{tag}"
                datas.append([text, tag_list])
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.contents = list()
        self.offset_lists = list()
        return datas
    
    def _load_split_file(self, data_file):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
        Returns: 无
        """
        datas = list()
        label_set = set(["O"])
        contents = list()
        offset_lists = list()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = json.loads(line)
                if not line:
                    continue
                content = line["text"]
                offsets = line["offsets"]
                contents.append(content)
                offset_lists.append(offsets)
                sents = line["sents"]
                entity_lists = line["label_lists"]
                for sent, entity_list in zip(sents, entity_lists):
                    tag_list = ["O"] * len(sent)
                    for entity in entity_list:
                        start, end, tag = entity
                        label_set.add(f"B-{tag}")
                        label_set.add(f"I-{tag}")
                        tag_list[start] = f"B-{tag}"
                        for i in range(start+1, end):
                            tag_list[i] = f"I-{tag}"
                    datas.append([sent, tag_list])
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.contents = contents
        self.offset_lists = offset_lists
        return datas
    
    def _load_biaffine_file(self, data_file):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
            delimiter(str): 分隔符
        Returns: 无
        """
        datas = list()
        label_set = set()
        seg_set = set()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = json.loads(line)
                if not line:
                    continue
                text = line["text"]
                entities = line.get("labels", list())
                if not entities:
                    entities = line.get("ner", list())
                for entity in entities:
                    label_set.add(entity[2])
                segs = line.get("seg", list())
                for seg in segs:
                    seg_set.add(seg[2])
                datas.append([text, entities, segs])
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.seg_list = ["[PAD]"] + sorted(list(seg_set))
        self.seg_to_id = {seg: idx for idx, seg in enumerate(self.seg_list)}
        self.contents = list()
        self.offset_lists = list()
        return datas
    
    def _load_span_file(self, data_file):
        """
        加载数据集文件
        Args:
            data_file(str): 数据集文件路径
        Returns: 无
        """
        datas = list()
        label_set = set()
        seg_set = set()
        with open(data_file, 'r', encoding='utf-8') as rf:
            for line in rf:
                line = json.loads(line)
                if not line:
                    continue
                text = line["text"]
                entities = line.get("labels", list())
                if not entities:
                    entities = line.get("ner", list())
                for entity in entities:
                    label_set.add(entity[2])
                segs = line.get("seg", list())
                for seg in segs:
                    seg_set.add(seg[2])
                datas.append([text, entities, segs])
        self.label_list = ["[PAD]"] + sorted(list(label_set))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        self.seg_list = ["[PAD]"] + sorted(list(seg_set))
        self.seg_to_id = {seg: idx for idx, seg in enumerate(self.seg_list)}
        self.contents = list()
        self.offset_lists = list()
        return datas

    def get_label_list(self):
        return self.label_list

    def get_label_to_id(self):
        return self.label_to_id

    def get_seg_list(self):
        return self.seg_list

    def get_seg_to_id(self):
        return self.seg_to_id

    def get_contents(self):
        return self.contents

    def get_offset_lists(self):
        return self.offset_lists
    
    def tokenize(self, text, labels, words, format="bio"):
        if format == "bio":
            return self.tokenize_bio(text, labels)
        elif format == "bies":
            return self.tokenize_bies(text, labels)
        elif format == "biaffine":
            return self.tokenize_biaffine(text, labels, words)
        elif format == "span":
            return self.tokenize_span(text, labels, words)
        else:
            raise ValueError

    def tokenize_bio(self, words, labels):
        _tokens = list()
        _labels = list()
        _masks = list()
        _word_lens = list()
        for word, label in zip(words, labels):
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                raise ValueError
            _word_lens.append(len(tmp_tokens))
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
                    if label.startswith("B-"):
                        label = "I" + label[1:]
                _labels.append(label)
        return _tokens, _labels, _masks, _word_lens
    
    def tokenize_bies(self, words, labels):
        _tokens = list()
        _labels = list()
        _masks = list()
        _word_lens = list()
        for word, label in zip(words, labels):
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                raise ValueError
            _word_lens.append(len(tmp_tokens))
            if len(tmp_tokens) == 1:
                _tokens.append(tmp_tokens[0])
                _labels.append(label)
                _masks.append(1)
                continue
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
                if re.search(r"^(B|E)-", label):
                    tmp_label = "I" + label[1:]
                else:
                    tmp_label = label
                if idx == len(tmp_tokens) - 1:
                    tmp_label = "E" + label[1:]
                _labels.append(tmp_label)
        return _tokens, _labels, _masks, _word_lens
    
    def tokenize_biaffine(self, words, labels, segs):
        _tokens = list()
        _labels = list()
        _segs = list()
        _masks = list()
        _word_lens = list()
        for idx, word in enumerate(words):
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                tmp_tokens = ['[UNK]']
                # raise ValueError
            _word_lens.append(len(tmp_tokens))
            if len(tmp_tokens) == 1:
                _tokens.append(tmp_tokens[0])
                _masks.append(1)
                continue
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
        _labels = labels
        _segs = segs
        return _tokens, _labels, _segs, _masks, _word_lens
    
    def tokenize_span(self, words, labels, segs):
        _tokens = list()
        _labels = list()
        _segs = list()
        _masks = list()
        _word_lens = list()
        for idx, word in enumerate(words):
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                tmp_tokens = ['[UNK]']
                # raise ValueError
            _word_lens.append(len(tmp_tokens))
            if len(tmp_tokens) == 1:
                _tokens.append(tmp_tokens[0])
                _masks.append(1)
                continue
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
        _labels = labels
        _segs = segs
        return _tokens, _labels, _segs, _masks, _word_lens

    def _to_features(self, datas, file_format="general", max_seq_length=-1,
                     cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=0,
                     sep_token="[SEP]",pad_on_left=False,pad_token="[PAD]",pad_token_segment_id=0,
                     sequence_a_segment_id=0,mask_padding_with_zero=True,):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        features = list()
        for (ex_index, data) in enumerate(datas):
            tokens, labels, segs, masks, word_lens = self.tokenize(data[0], data[1], data[2], file_format)
            if file_format not in ["biaffine", "span"]:
                label_ids = [self.label_to_id[x] for x in labels]
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                masks = masks[: (max_seq_length - special_tokens_count)]
                if file_format not in ["biaffine", "span"]:
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
            pad_id = self.label_to_id.get(pad_token)
            tokens += [sep_token]
            masks += [0]
            if file_format not in ["biaffine", "span"]:
                label_ids += [pad_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            masks = [0] + masks
            if file_format not in ["biaffine", "span"]:
                label_ids = [pad_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len = len(input_ids)
            
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_id] * padding_length
            masks += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            if file_format not in ["biaffine", "span"]:
                label_ids += [pad_id] * padding_length
            
            label_mask = [1] * max_seq_length
            seg_ids = [0] * max_seq_length
            start_ids = [0] * max_seq_length
            end_ids = [0] * max_seq_length
            if file_format in ["span"]:
                label_ids = [0] * max_seq_length
            
            if file_format == "biaffine":
                # label_mask还要再看看
                label_mask = [[0 for _ in range(max_seq_length)]]
                for i in range(1, input_len-1):
                    label_mask.append([0 for _ in range(i)] + [1 for _ in range(input_len-i-1)] + [0 for _ in range(max_seq_length-input_len+1)])
                for i in range(input_len-1, max_seq_length):
                    label_mask.append([0 for _ in range(max_seq_length)])
                assert len(label_mask) == len(label_mask[0])
                
                label_ids = list()
                for i in range(max_seq_length):
                    label_ids.append([0 for _ in range(max_seq_length)])
                for entity in data[1]:
                    try:
                        start, end, tag = entity
                    except Exception:
                        start, end, tag, _ = entity
                    # 默认第一个字符为[CLS]
                    if end > max_seq_length - 1:
                        print("big")
                        continue
                    label_ids[start+1][end] = self.label_to_id[tag]
                
                seg_ids = label_ids
                if self.seg_list:
                    seg_ids = list()
                    for i in range(max_seq_length):
                        seg_ids.append([0 for _ in range(max_seq_length)])
                    for seg in data[2]:
                        try:
                            start, end, tag = seg
                        except Exception:
                            start, end, tag, _ = seg
                        # 默认第一个字符为[CLS]
                        if end > max_seq_length - 1:
                            print("big")
                            continue
                        seg_ids[start+1][end] = self.seg_to_id[tag]
            elif file_format == "span":
                for entity in data[1]:
                    try:
                        start, end, tag = entity
                    except Exception:
                        start, end, tag, _ = entity
                    if start+1 < max_seq_length:
                        start_ids[start+1] = self.label_to_id[tag]
                    if end < max_seq_length:
                        end_ids[end] = self.label_to_id[tag]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(label_mask) == max_seq_length
            assert len(seg_ids) == max_seq_length
            assert len(masks) == max_seq_length
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length
            if ex_index < 1:
                print("*** Example ***")
                print("tokens: ", tokens)
                print("input_ids: ", input_ids)
                print("input_mask: ", input_mask)
                print("input_len: ", input_len)
                print("segment_ids: ", segment_ids)
                print("label_mask: ", label_mask)
                print("masks: ", masks)
                if file_format == "biaffine":
                    print("label_ids: ")
                    for i in range(len(label_ids)):
                        print(label_ids[i])
                elif file_format == "span":
                    print("start_ids: ", start_ids)
                    print("end_ids: ", end_ids)
                else:
                    print("label_ids: ", label_ids)
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, label_mask=label_mask, input_len=input_len, masks=masks, seg_ids=seg_ids, start_ids=start_ids, end_ids=end_ids))
        return features

    def save_label(self, label_file):
        with open(label_file, "w", encoding="utf-8") as lf:
            for label in self.label_list:
                lf.write(f"{label}\n")
    
    def save_seg(self, seg_file):
        with open(seg_file, "w", encoding="utf-8") as lf:
            for seg in self.seg_list:
                lf.write(f"{seg}\n")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask, input_len, masks, seg_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.input_len = input_len
        self.masks = masks
        self.seg_ids = seg_ids
        self.start_ids = start_ids
        self.end_ids = end_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
