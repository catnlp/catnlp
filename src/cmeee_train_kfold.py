# -*- coding:utf-8 -*-

import argparse
import copy
import logging
import logging.config

from catnlp.common.load_file import load_config_file
from catnlp.ner.train import NerTrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--train_config", type=str,
                        default="resources/config/ner/CMeEE/train_biaffine_kfold.yaml", help="训练配置")
    parser.add_argument("--log_config", type=str,
                        default="resources/config/ner/logging.yaml", help="日志配置")
    args = parser.parse_args()

    try:
        train_config = load_config_file(args.train_config)
        log_config = load_config_file(args.log_config)
        # logging.config.dictConfig(log_config)
    except Exception:
        raise RuntimeError("加载配置文件失败")

    task = args.task.lower()
    if task == "ner":
        k = train_config["k"]
        for i in range(k):
            tmp_train_config = copy.deepcopy(train_config)
            tmp_train_config["input"] += f"/{i}"
            tmp_train_config["output"] += f"/{i}"
            tmp_train_config["summary"] += f"/{i}"
            ner_train = NerTrain(tmp_train_config)
    else:
        raise RuntimeError(f"{args.task}未开发")
