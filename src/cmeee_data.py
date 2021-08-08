from genericpath import exists
import json
import re
import os
import random
from pathlib import Path
from sklearn.model_selection import KFold


def tojson(source, target):
    with open(source, "r", encoding="utf-8") as sf, \
            open(target, "w", encoding="utf-8") as tf:
        punc_set = set()
        punc_pattern = r"[,?!，。？！；;]"
        count = 0
        for line in sf:
            line = json.loads(line)
            if not line:
                continue
            text = line["text"]
            entities = line["label"]
            entity_list = list()
            for entity in entities:
                start, end, tag = entity
                end += 1
                word = text[start: end]
                puncs = re.findall(punc_pattern, word)
                if puncs:
                    if re.search(r"[。；;]", word):
                        print(word)
                    # print(word, tag)
                    if re.search(r"[，。,;；]", word):
                        count += 1
                    for punc in puncs:
                        punc_set.add(punc)

                entity_list.append([start, end, tag, word])
            tf.write(json.dumps({
                "text": text,
                "ner": entity_list
            }, ensure_ascii=False) + "\n")
        print(punc_set)
        print(count)


def kfold_file(source, target_dir, k):
    with open(source, "r", encoding="utf-8") as sf:
        line_list = list()
        for line in sf:
            line = json.loads(line)
            if not line:
                continue
            line_list.append(line)
        random.shuffle(line_list)
        kf = KFold(n_splits=k)
        for idx, (train_idxs, dev_idxs) in enumerate(kf.split(line_list)):
            cur_dir = target_dir / f"{idx}"
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir, exist_ok=True)
            train_file = cur_dir / f"train.json"
            dev_file = cur_dir / f"dev.json"
            with open(train_file, "w", encoding="utf-8") as tf, \
                    open(dev_file, "w", encoding="utf-8") as df:
                for i in train_idxs:
                    line = json.dumps(line_list[i], ensure_ascii=False) + "\n"
                    tf.write(line)
                for i in dev_idxs:
                    line = json.dumps(line_list[i], ensure_ascii=False) + "\n"
                    df.write(line)


def split_file(source, train_all, train, dev):
    with open(source, "r", encoding="utf-8") as sf, \
            open(train_all, "w", encoding="utf-8") as taf, \
            open(train, "w", encoding="utf-8") as tf, \
            open(dev, "w", encoding="utf-8") as df:
        line_list = list()
        for line in sf:
            line = json.loads(line)
            if not line:
                continue
            line_list.append(line)
        random.shuffle(line_list)
        line_list_len = len(line_list)
        split_len = int(0.8 * line_list_len)
        for i in range(line_list_len):
            line = json.dumps(line_list[i], ensure_ascii=False) + "\n"
            taf.write(line)
            if i < split_len:
                tf.write(line)
            else:
                df.write(line)


def cut_file(source, target):
    with open(source, "r", encoding="utf-8") as sf, \
        open(target, "w", encoding="utf-8") as tf:
        for line in sf:
            line = json.loads(line)
            if not line:
                continue
            text = line["text"]
            entities = line["ner"]
            sent_list, entity_lists, _ = cut(text, entities, max_len=200, overlap_len=50)
            for sent, entities in zip(sent_list, entity_lists):
                tf.write(json.dumps({
                    "text": sent,
                    "ner": entities
                }, ensure_ascii=False) + "\n")


def cut(text, entities=None, max_len=256, overlap_len=50):
    tags = get_tags(len(text), entities)
    sents = get_sents(text, tags)
    # sents = re.split(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', text)
    sents_len = len(sents)
    offset_list = list()
    i = 0
    end_idx = 0
    sent_list = list()
    entity_lists = list()
    while i < sents_len:
        sent = sents[i]
        sent_len = len(sent)
        end_idx += sent_len
        if not sent or re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', sent):
            i += 1
            continue
        # 搜索前缀
        pre_list = list()
        pre_list_len = 0
        j = i - 1
        while j >= 0:
            sent_j = sents[j]
            sent_j_len = len(sent_j)
            if pre_list_len + sent_j_len < overlap_len:
                pre_list.append(sent_j)
                pre_list_len += sent_j_len
            else:
                break
            j -= 1
        pre_idx = 0
        pre_list = pre_list[::-1]
        for tmp_sent in pre_list:
            if not tmp_sent or re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_sent):
                pre_idx += 1
                pre_list_len -= len(tmp_sent)
            else:
                break
        pre_list = pre_list[pre_idx:]

        # 搜索后缀
        post_list = list()
        post_list_len = 0
        j = i + 1
        while j < sents_len:
            sent_j = sents[j]
            sent_j_len = len(sent_j)
            if pre_list_len + sent_len + post_list_len + sent_j_len < max_len:
                post_list.append(sent_j)
                post_list_len += sent_j_len
            else:
                break
            j += 1
        
        # 拼接
        sent = "".join(pre_list + [sent] + post_list)
        sent_list.append(sent)
        end_idx += post_list_len
        start_idx = end_idx - len(sent)
        offset_list.append(start_idx)
        if tags:
            entity_list = get_entity_list(entities, start_idx, end_idx)
            entity_lists.append(entity_list)
        i = j

    if valid(text, sent_list, offset_list):
        return sent_list, entity_lists, offset_list
    else:
        raise ValueError


def get_tags(text_len, entities):
    tags = [True] * text_len
    if entities:
        for entity in entities:
            start, end, _, _ = entity
            for i in range(start, end):
                tags[i] = False
    return tags


def get_sents(text, tags):
    sents = list()
    start = 0
    text_len = len(text)
    for idx, (word, tag) in enumerate(zip(text, tags)):
        if not tag:
            continue
        if re.search(r"[。？?；;！!]", word):  # catnlp 去掉逗号
            sent = text[start: idx]
            if sent:
                sents.append(sent)
            sents.append(word)
            start = idx + 1
        elif word == ".":
            if idx > 1 and re.search(r"\d", text[idx-1]):
                continue
            if idx < text_len - 1 and re.search(r"\d", text[idx+1]):
                continue
            sent = text[start: idx]
            if sent:
                sents.append(sent)
            sents.append(word)
            start = idx + 1
    if start < text_len:
        sents.append(text[start: text_len])
    return sents


def get_entity_list(entities, start, end):
    entity_list = list()
    for entity in entities:
        s, e, t, w = entity
        if start <= s and e <= end and s <= e:
            entity_list.append([s-start, e-start, t, w])
    return entity_list


def valid(text, sent_list, offset_list):
    for offset, sent in zip(offset_list, sent_list):
        sent_len = len(sent)
        start = offset
        end = offset + sent_len
        if text[start: end] != sent:
            print("---")
            print(text[start: end])
            print(sent)
            return False
    return True


if __name__ == "__main__":
    source_path = Path("resources/data/dataset/ner/zh/ccks/cmeee/raw")
    json_path = Path("resources/data/dataset/ner/zh/ccks/cmeee/json")
    # split_path = Path("resources/data/dataset/ner/zh/ccks/cmeee/split")
    kfold_path = Path("resources/data/dataset/ner/zh/ccks/cmeee/kfold")
    cut_path = Path("resources/data/dataset/ner/zh/ccks/cmeee/cut")
    for path in [json_path, kfold_path, cut_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    print("tojson")
    datas = ["train", "test"]
    for data in datas:
        source_file = source_path / f"{data}.json"
        target_file = json_path / f"{data}.json"
        tojson(source_file, target_file)
    
    print("kfold_file")
    k = 5
    source_file = json_path / "train.json"
    kfold_file(source_file, kfold_path, k=k)
    
    # print("split_file")
    # source_file = json_path / "train.json"
    # train_all_file = split_path / "train_all.json"
    # train_file = split_path / "train.json"
    # dev_file = split_path / "dev.json"
    # split_file(source_file, train_all_file, train_file, dev_file)

    print("cut_file")
    for i in range(k):
        datas = ["train", "dev"]
        for data in datas:
            source_file = kfold_path / f"{i}/{data}.json"
            cur_path = cut_path / f"{i}"
            if not os.path.exists(cur_path):
                os.makedirs(cur_path, exist_ok=True)
            target_file = cur_path / f"{data}.json"
            cut_file(source_file, target_file)
