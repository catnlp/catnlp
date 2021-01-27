#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json

from .util.clean import clean_text


class NerFormat:
    def convert(self, source, target, is_clean, format):
        if format == "json2bio":
            self._json2bio(source, target, is_clean)
        elif format == "bio2json":
            self._bio2json(source, target, is_clean)
        elif format == "json2bioes":
            self._json2bioes(source, target, is_clean)
        elif format == "bioes2json":
            self._bioes2json(source, target, is_clean)
        elif format == "json2clue":
            self._json2clue(source, target, is_clean)
        elif format == "clue2json":
            self._clue2json(source, target, is_clean)
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
                    word, tag = line.split("\t")
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


    def _json2bioes(self, source, target, is_clean=False):
        """
        json格式转bioes格式
        Args:
            source(str): json格式的文件路径
            target(str): bioes格式的文件路径
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


    def _bioes2json(self, source, target, is_clean=False):
        """
        bioes格式转json格式
        Args:
            source(str): bioes格式的文件路径
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
