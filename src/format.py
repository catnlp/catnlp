# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

from catnlp.ner.format import NerFormat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换模型格式")
    parser.add_argument(
        "--input_dir", default="resources/data/dataset/ner/zh/cluener/json", type=str, help="输入文件夹"
    )
    parser.add_argument(
        "--input_type", default="json", type=str, help="输入文件类型"
    )
    parser.add_argument(
        "--output_dir", default="resources/data/dataset/ner/zh/cluener/split", type=str, help="输出文件夹"
    )
    parser.add_argument(
        "--output_type", default="json", type=str, help="输出文件类型"
    )
    parser.add_argument(
        "--convert", default="json2split", type=str,
        choices=["json2bio", "bio2json", "json2bioes", "bioes2json", "json2clue", "clue2json", "json2split", "split2json"], 
        help="转化类型"
    )
    args = parser.parse_args()
    # NER Format: json2bio,bio2json,json2bioes,bioes2json,json2clue,clue2json,json2split,split2json
    ner_format = NerFormat()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    for path in [output_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    datasets = ["train", "dev"]
    for dataset in datasets:
        input_file = input_path / f"{dataset}.{args.input_type}"
        output_file = output_path / f"{dataset}.{args.output_type}"
        ner_format.convert(input_file, output_file, 
                           is_clean=False, format=args.convert)
