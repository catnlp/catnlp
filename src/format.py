#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from catnlp.ner.format import NerFormat


if __name__ == "__main__":
    # NER Format: json2bio,bio2json,json2bioes,bioes2json,json2clue,clue2json
    ner_format = NerFormat()
    source_path = Path("catnlp/data/dataset/ner/zh/cluener/json")
    target_path = Path("catnlp/data/dataset/ner/zh/cluener/bio")
    for path in [target_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    datasets = ["train", "dev"]
    for dataset in datasets:
        source_file = source_path / f"{dataset}.json"
        target_file = target_path / f"{dataset}.txt"
        ner_format.convert(source_file, target_file, 
                           is_clean=False, format="json2bio")
