# -*- coding: utf-8 -*-

import json
from collections import defaultdict

import numpy as np

from .util.clean import clean_text
from .util.split import cut, recover
from ..tool import visual


class NerFormat:
    def convert(self, source, target, is_clean, format):
        if format == "json2bio":
            self._json2bio(source, target, is_clean)
        elif format == "bio2json":
            self._bio2json(source, target, is_clean)
        elif format == "json2bies":
            self._json2bies(source, target, is_clean)
        elif format == "bies2json":
            self._bies2json(source, target, is_clean)
        elif format == "json2clue":
            self._json2clue(source, target, is_clean)
        elif format == "clue2json":
            self._clue2json(source, target, is_clean)
        elif format == "json2split":
            self._json2split(source, target, is_clean)
        elif format == "split2json":
            self._split2json(source, target, is_clean)
        elif format == "json2many":
            self._json2many(source, target, is_clean)
        else:
            raise RuntimeError(f"无效格式：{format}")
        print(f"{format}格式转换成功")

    def _json2bio(self, source, target, is_clean=False):
        """
        json格式转bio格式
        Args:
            source(str): json格式的文件路径
            target(str): bio格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                entities = line['labels']
                tag_list = ['O'] * len(text)
                for entity in entities:
                    start, end, tag = entity
                    tag_list[start] = f"B-{tag}"
                    for i in range(start + 1, end):
                        tag_list[i] = f"I-{tag}"
                for word, tag in zip(text, tag_list):
                    tf.write(f"{word}\t{tag}\n")
                tf.write("\n")


    def _bio2json(self, source, target, is_clean=False):
        """
        bio格式转json格式
        Args:
            source(str): bio格式的文件路径
            target(str): json格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            word_list = []
            entities = []
            pre_o = True
            idx = 0
            for line in sf:
                line = line.rstrip()
                if line:
                    if line.find("-DOCSTART-") != -1:
                        continue
                    tokens = line.split("\t")
                    word = tokens[0]
                    tag = tokens[-1]
                    word_list.append(word)
                    if tag != "O":
                        prefix, tname = tag.split("-")
                        if prefix == "B":
                            entities.append([idx, idx + 1, tname])
                        elif prefix == "I":
                            if pre_o or \
                                    (entities and entities[-1][-1] != tname):
                                entities.append([idx, idx + 1, tname])
                            else:
                                entities[-1][1] += 1
                        else:
                            print(f"error line: {line}")
                            exit(1)
                        pre_o = False
                    else:
                        pre_o = True
                    idx += 1
                elif entities:
                    tf.write(json.dumps({
                        "words": word_list,
                        "labels": entities
                    }, ensure_ascii=False) + "\n")
                    word_list = []
                    entities = []
                    pre_o = True
                    idx = 0


    def _json2bies(self, source, target, is_clean=False):
        """
        json格式转bies格式
        Args:
            source(str): json格式的文件路径
            target(str): bies格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                entities = line['labels']
                tag_list = ['O'] * len(text)
                for entity in entities:
                    start, end, tag = entity
                    if start + 1 == end:
                        tag_list[start] = f"S-{tag}"
                        continue
                    else:
                        tag_list[start] = f"B-{tag}"
                    for i in range(start + 1, end - 1):
                        tag_list[i] = f"I-{tag}"
                    tag_list[end - 1] = f"E-{tag}"
                for word, tag in zip(text, tag_list):
                    tf.write(f"{word}\t{tag}\n")
                tf.write("\n")


    def _bies2json(self, source, target, is_clean=False):
        """
        bies格式转json格式
        Args:
            source(str): bies格式的文件路径
            target(str): json格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            word_list = []
            entities = []
            pre_o = True
            idx = 0
            for line in sf:
                line = line.rstrip()
                if line:
                    word, tag = line.split("\t")
                    word_list.append(word)
                    if tag != "O":
                        prefix, tname = tag.split("-")
                        if prefix in ["B", "S"]:
                            entities.append([idx, idx + 1, tname])
                        elif prefix in ["I", "E"]:
                            if pre_o or \
                                    (entities and entities[-1][-1] != tname):
                                entities.append([idx, idx + 1, tname])
                            else:
                                entities[-1][1] += 1
                        else:
                            print(f"error line: {line}")
                            exit(1)
                        pre_o = False
                    else:
                        pre_o = True
                    idx += 1
                elif entities:
                    text = "".join(word_list)
                    if is_clean:
                        text = clean_text(text)
                    tf.write(json.dumps({
                        "text": text,
                        "labels": entities
                    }, ensure_ascii=False) + "\n")
                    word_list = []
                    entities = []
                    pre_o = True
                    idx = 0


    def _clue2json(self, source, target, is_clean=False):
        """
        clue格式转json格式
        Args:
            source(str): clue格式的文件路径
            target(str): json格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                entities = line['label']
                entity_list = []
                for tag in entities:
                    for item in entities[tag]:
                        pos_list = entities[tag][item]
                        for pos in pos_list:
                            pos[-1] += 1
                            pos.append(tag)
                            entity_list.append(pos)
                tf.write(json.dumps({
                    "text": text,
                    "labels": entity_list
                }, ensure_ascii=False) + "\n")


    def _json2clue(self, source, target, is_clean=False):
        """
        json格式转clue格式
        Args:
            source(str): json格式的文件路径
            target(str): clue格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                entities = line['label']
                entity_dict = {}
                for entity in entities:
                    start, end, tag = entity
                    word = text[start: end + 1]
                    if tag not in entity_dict:
                        entity_dict[tag] = {
                            word: [[start, end]]
                        }
                    else:
                        if word not in entity_dict[tag]:
                            entity_dict[tag][word] = [[start, end]]
                        else:
                            entity_dict[tag][word].append([start, end])

                tf.write(json.dumps({
                    'text': text,
                    'label': entity_dict
                }, ensure_ascii=False) + '\n')
    
    def _split2json(self, source, target, is_clean=False):
        """
        split格式转json格式
        Args:
            source(str): split格式的文件路径
            target(str): json格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                sent_list = line['sents']
                tag_lists = line['labels']
                offset_list = line['offsets']
                if is_clean:
                    text = clean_text(text)
                
                entity_list = recover(tag_lists, offset_list)

                tf.write(json.dumps({
                    "text": text,
                    "labels": entity_list
                }, ensure_ascii=False) + "\n")

    def _json2split(self, source, target, is_clean=False, max_len=30, overlap_len=10):
        """
        json格式转split格式
        Args:
            source(str): json格式的文件路径
            target(str): split格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                tag_list = ["O"] * len(text)
                entities = line['labels']
                for entity in entities:
                    start, end, tag = entity
                    tag_list[start] = f"B-{tag}"
                    for i in range(start+1, end):
                        tag_list[i] = f"I-{tag}"
                
                sent_list, entity_lists, offset_list = cut(text, tag_list, max_len, overlap_len)

                tf.write(json.dumps({
                    'text': text,
                    'sents': sent_list,
                    'label_lists': entity_lists,
                    'offsets': offset_list
                }, ensure_ascii=False) + '\n')
    
    def _json2many(self, source, target, is_clean=False):
        """
        json格式转many格式
        Args:
            source(str): json格式的文件路径
            target(str): many格式的文件路径
        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as sf, \
                open(target, 'w', encoding='utf-8') as tf:
            for line in sf:
                line = json.loads(line)
                if not line:
                    continue
                text = line['text']
                if is_clean:
                    text = clean_text(text)
                entities = line['labels']

                for entity in entities:
                    start, end, tag = entity
                    tmp_text = text[start: end]
                    tmp_end = end - start
                    tf.write(json.dumps({
                        "text": tmp_text,
                        "labels": [[0, tmp_end, tag]]
                    }, ensure_ascii=False) + "\n")
                
                tf.write(json.dumps(line, ensure_ascii=False) + "\n")          


class JsonFormat:
    def __init__(self, data_file) -> None:
        self._data = self.load(data_file)

    def load(self, data_file):
        datas = list()
        with open(data_file, "r", encoding="utf-8") as df:
            for line in df:
                line = json.loads(line)
                if not line:
                    continue
                datas.append(line)
        return datas

    def get_text(self):
        text_list = list()
        for data in self._data:
            text = data.get("text")
            if text:
                text_list.append(text)
        return text_list

    def get_text_len(self):
        text_list = self.get_text()
        return [len(text) for text in text_list]
    
    def get_label_dict(self):
        label_dict = defaultdict(int)
        for data in self._data:
            labels = data.get("labels")
            for label in labels:
                _, _, tag = label
                label_dict[tag] += 1
        return label_dict

    def size(self):
        return len(self._data)
    
    def statistics(self):
        len_list = self.get_text_len()
        len_array = np.array(sorted(len_list))
        length = len_array.size
        len_mean = np.mean(len_array)
        len_std = np.std(len_array)
        len_min = len_array[0]
        len_50 = len_array[int(length * 0.5)]
        len_70 = len_array[int(length * 0.7)]
        len_90 = len_array[int(length * 0.9)]
        len_max = len_array[-1]
        print(f"count:\t{length}")
        print(f"mean:\t{round(len_mean, 2)}")
        print(f"std:\t{round(len_std, 2)}")
        print(f"min:\t{len_min}")
        print(f"50%:\t{round(len_50, 2)}")
        print(f"70%:\t{round(len_70, 2)}")
        print(f"90%:\t{round(len_90, 2)}")
        print(f"max:\t{len_max}")
    
    def draw_histogram(self, num_bins=100, density=False):
        len_list = self.get_text_len()
        len_list = np.array(len_list)
        visual.draw_histogram(len_list, num_bins=num_bins, density=density)

    def draw_hbar(self):
        label_dict = self.get_label_dict()
        label_list = label_dict.items()
        label_list = sorted(label_list, key=lambda i: i[1], reverse=True)
        labels = [i[0] for i in label_list]
        datas = [i[1] for i in label_list]
        visual.draw_hbar(labels, datas)


class ConllFormat:
    def __init__(self, data_file, delimiter) -> None:
        self._data = self.load(data_file, delimiter)

    def load(self, data_file, delimiter):
        datas = list()
        words = list()
        labels = list()
        with open(data_file, "r", encoding="utf-8") as df:
            for line in df:
                line = line.rstrip()
                if not line:
                    if words:
                        datas.append(("".join(words), labels))
                        words = list()
                        labels = list()
                else:
                    word, label = line.split(delimiter)
                    words.append(word)
                    labels.append(label)
        return datas

    def get_text(self):
        text_list = list()
        for data in self._data:
            text = data[0]
            if text:
                text_list.append(text)
        return text_list

    def get_text_len(self):
        text_list = self.get_text()
        return [len(text) for text in text_list]
    
    def get_label_dict(self):
        label_dict = defaultdict(int)
        for data in self._data:
            labels = data[1]
            for label in labels:
                label_dict[label] += 1
        return label_dict

    def size(self):
        return len(self._data)
    
    def statistics(self):
        len_list = self.get_text_len()
        len_array = np.array(sorted(len_list))
        length = len_array.size
        len_mean = np.mean(len_array)
        len_std = np.std(len_array)
        len_min = len_array[0]
        len_50 = len_array[int(length * 0.5)]
        len_70 = len_array[int(length * 0.7)]
        len_90 = len_array[int(length * 0.9)]
        len_max = len_array[-1]
        print(f"count:\t{length}")
        print(f"mean:\t{round(len_mean, 2)}")
        print(f"std:\t{round(len_std, 2)}")
        print(f"min:\t{len_min}")
        print(f"50%:\t{round(len_50, 2)}")
        print(f"70%:\t{round(len_70, 2)}")
        print(f"90%:\t{round(len_90, 2)}")
        print(f"max:\t{len_max}")
    
    def draw_histogram(self, num_bins=100, density=False):
        len_list = self.get_text_len()
        len_list = np.array(len_list)
        visual.draw_histogram(len_list, num_bins=num_bins, density=density)

    def draw_hbar(self):
        label_dict = self.get_label_dict()
        label_list = label_dict.items()
        label_list = sorted(label_list, key=lambda i: i[1], reverse=True)
        labels = [i[0] for i in label_list]
        datas = [i[1] for i in label_list]
        visual.draw_hbar(labels, datas)
