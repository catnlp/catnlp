# -*- coding:utf-8 -*-

import argparse
import logging
import logging.config

from catnlp.common.load_file import load_config_file
from catnlp.ner.train import NerTrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--train_config", type=str,
                        default="data/config/ner/bert.yaml", help="训练配置")
    parser.add_argument("--log_config", type=str,
                        default="data/config/ner/logging.yaml", help="日志配置")
    args = parser.parse_args()

    try:
        train_config = load_config_file(args.train_config)
        log_config = load_config_file(args.log_config)
        logging.config.dictConfig(log_config)
    except Exception:
        raise RuntimeError("加载配置文件失败")

    task = args.task.lower()
    if task == "ner":
        ner_train = NerTrain(train_config)
    else:
        raise RuntimeError(f"{args.task}未开发")
