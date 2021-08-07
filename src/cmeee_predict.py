# -*- coding:utf-8 -*-

import argparse
import logging
import logging.config
import json

from tqdm import tqdm

from catnlp.common.load_file import load_config_file
from catnlp.ner.predict import NerPredict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--input_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/cmeee/0807/test.json", help="测试文件")
    parser.add_argument("--output_file", type=str,
                    default="resources/data/dataset/ner/zh/ccks/cmeee/0807/test_pred.json", help="结果文件")
    parser.add_argument("--predict_config", type=str,
                        default="resources/config/ner/CMeEE/predict_biaffine.yaml", help="预测配置")
    parser.add_argument("--log_config", type=str,
                        default="resources/config/ner/logging.yaml", help="日志配置")
    args = parser.parse_args()

    try:
        predict_config = load_config_file(args.predict_config)
        log_config = load_config_file(args.log_config)
        # logging.config.dictConfig(log_config)
    except Exception:
        raise RuntimeError("加载配置文件失败")

    task = args.task.lower()
    if task == "ner":
        ner_services = list()
        for type in predict_config:
            ner_services.append(NerPredict(predict_config[type], type=type))
        with open(args.input_file, "r", encoding="utf-8") as sf, \
                open(args.output_file, "w", encoding="utf-8") as tf:
            lines = sf.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                if not line:
                    continue
                text = line["text"]
                entity_list = list()
                for ner_service in ner_services:
                    entity_list += ner_service.predict(text)
                tf.write(json.dumps({
                    "text": text,
                    "ner": entity_list
                }, ensure_ascii=False) + "\n")
    else:
        raise RuntimeError(f"{args.task}未开发")
