#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import logging
import logging.config

from common.load_file import load_config_file
from ner.train import NerTrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--type", type=str,
                        default="NER", help="训练NER模型")
    parser.add_argument("--train_config", type=str,
                        default="", help="训练配置")
    parser.add_argument("--log_config", type=str,
                        default="", help="日志配置")
    args = parser.parse_args()

    try:
        train_config = load_config_file(args.train_config)
        log_config = load_config_file(args.log_config)
        logging.config.dictConfig(log_config)
    except Exception:
        raise RuntimeError("加载配置文件失败")

    type_lower = args.type.lower()
    if type_lower == "ner":
        NerTrain(train_config)
    else:
        raise RuntimeError(f"{args.type}未开发")
