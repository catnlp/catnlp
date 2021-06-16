# -*- coding:utf-8 -*-

import argparse
import logging
import logging.config

from tqdm import tqdm

from catnlp.common.load_file import load_config_file
from catnlp.ner.predict import NerPredict


def merge_entities(entity_list):
    sorted_entity_list = sorted(entity_list, key=lambda i: i[1]-i[0], reverse=True)
    new_entity_list = list()
    is_appear_list = [False] * len(entity_list)
    for idx, entity in enumerate(sorted_entity_list):
        if is_appear_list[idx]:
            continue
        new_entity_list.append(entity)
        for idy, tmp_entity in enumerate(sorted_entity_list[idx:]):
            if entity[0] < tmp_entity[1] and \
                    tmp_entity[0] < entity[1]:
                is_appear_list[idx+idy] = True
    return new_entity_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--input_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/address/json/test.txt", help="测试文件")
    parser.add_argument("--output_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/address/json/试一下_addr_parsing_runid.txt", help="结果文件")
    parser.add_argument("--predict_config", type=str,
                        default="resources/config/ner/predict/bert.yaml", help="预测配置")
    parser.add_argument("--log_config", type=str,
                        default="resources/config/ner/logging.yaml", help="日志配置")
    args = parser.parse_args()

    try:
        predict_config = load_config_file(args.predict_config)
        log_config = load_config_file(args.log_config)
        logging.config.dictConfig(log_config)
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
                line = line.rstrip()
                if not line:
                    continue
                idx, text = line.split("\u0001")
                entity_list = list()
                for ner_service in ner_services:
                    entity_list += ner_service.predict(text)
                entity_list = merge_entities(entity_list)
                tag_list = ["O"] * len(text)
                for entity in entity_list:
                    start, end, tag = entity
                    if end - start == 1:
                        if tag in ["assist", "intersection"]:
                            tag_list[start] = f"S-{tag}"
                    else:
                        tag_list[start] = f"B-{tag}"
                        for i in range(start+1, end-1):
                            tag_list[i] = f"I-{tag}"
                        tag_list[end-1] = f"E-{tag}"
                tf.write(f"{idx}\u0001{text}\u0001{' '.join(tag_list)}\n")
    else:
        raise RuntimeError(f"{args.task}未开发")
