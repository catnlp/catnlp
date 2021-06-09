# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path


def get_dict(source, target):
    with open(source, "r", encoding="utf-8") as sf, \
            open(target, "w", encoding="utf-8") as tf:
        datas = json.loads(sf.read())
        word_set = set()
        for data in datas:
            name = data["name"]
            name = name.strip()
            word_set.add(name)
        word_list = sorted(list(word_set), key=lambda i: len(i), reverse=True)
        for word in word_list:
            tf.write(f"{word}\n")


if __name__ == "__main__":
    print(os.getcwd())
    source_path = Path("../../resources/data/dict/address/json")
    target_path = Path("../../resources/data/dict/address/txt")
    os.makedirs(target_path, exist_ok=True)
    datas = {
        "areas": "district",
        "cities": "city",
        "provinces": "prov",
        "streets": "town",
        "villages": "community"
    }
    for data in datas:
        print(f"{data} to {datas[data]}")
        source_file = source_path / f"{data}.json"
        target_file = target_path / f"{datas[data]}.txt"
        get_dict(source_file, target_file)
