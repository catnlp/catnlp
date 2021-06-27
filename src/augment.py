# -*- coding:utf-8 -*-

import re
from pathlib import Path

from nlpcda import Ner


def aug(source_path, tags, file_names, ignore_tag_list=None, augment_size=3, seed=0):
    ner = Ner(ner_dir_name=source_path,
            ignore_tag_list=ignore_tag_list,
            data_augument_tag_list=tags,
            augument_size=augment_size, seed=seed)
    for file_name in file_names:
        file_name = source_path / f"{file_name}.txt"
        data_sentence_arrs, data_label_arrs = ner.augment(file_name=file_name)
        # 3条增强后的句子、标签 数据，len(data_sentence_arrs)==3
        # 你可以写文件输出函数，用于写出，作为后续训练等
        print(data_sentence_arrs[0][:30], data_label_arrs[0][:30])


if __name__ == "__main__":
    source_path = Path("resources/data/dataset/ner/zh/ccks/address/0621/augment")
    tags = ["poi", "subpoi"]
    file_names = ["dev"]
    ignore_tag_list = ["O"]
    aug(source_path, tags, file_names, ignore_tag_list)
