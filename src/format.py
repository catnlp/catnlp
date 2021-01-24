#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from ner.format import NerFormat


if __name__ == "__main__":
    # NER Format
    ner_format = NerFormat()
    # 1 json2bio
    json_path = ""
    bio_path = ""

    # 2 bio2json
    bio_path = ""
    json_path = ""

    # 3 json2bioes
    json_path = ""
    bioes_path = ""

    # 4 bioes2json
    bioes_path = ""
    json_path = ""

    # 5 clue2json
    clue_path = Path("data/dataset/ner/zh/cluener/clue")
    json_path = Path("data/dataset/ner/zh/cluener/json")
    for path in [json_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    datasets = ["train", "dev"]
    for dataset in datasets:
        clue_file = clue_path / f"{dataset}.json"
        json_file = json_path / f"{dataset}.json"
        ner_format.clue2json(clue_file, json_file)

    # 6 json2clue
    json_path = ""
    clue_path = ""
