# -*- coding:utf-8 -*-

import argparse
import copy
import logging
import logging.config
import json

from tqdm import tqdm

from catnlp.common.load_file import load_config_file
from catnlp.ner.predict import NerPredict


def merge_sym_rest(entity_list):
    sym_list = [entity for entity in entity_list if entity[2] == "sym"]
    rest_list = [entity for entity in entity_list if entity[2] != "sym"]
    sym_list = merge_entities(sym_list)
    rest_list = merge_entities(rest_list)
    return sym_list + rest_list


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


def merge_entity_list(entity_list):
    new_entity_list = list()
    entity_list_len = len(entity_list)
    flag_list = [True] * entity_list_len
    for idx in range(entity_list_len):
        start1, end1, tag1, _ = entity_list[idx]
        for idy in range(idx+1, entity_list_len):
            start2, end2, tag2, _ = entity_list[idy]
            if (start2<=start1 and end1<=end2):
                if tag2 != "sym":
                    flag_list[idx] = False
            if (start1<=start2 and end2<=end1):
                if tag1 != "sym":
                    flag_list[idy] = False
    for idx, flag in enumerate(flag_list):
        if not flag:
            continue
        new_entity_list.append(entity_list[idx])
    return new_entity_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument("--task", type=str,
                        default="NER", help="任务")
    parser.add_argument("--input_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/cmeee/0808/test.json", help="测试文件")
    parser.add_argument("--output_dir", type=str,
                    default="resources/data/dataset/ner/zh/ccks/cmeee/0814", help="结果文件")
    parser.add_argument("--predict_config", type=str,
                        default="resources/config/ner/CMeEE/predict_multi_biaffine_kfold.yaml", help="预测配置")
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
        k = predict_config["k"]
        for i in range(k):
            ner_services = list()
            print(predict_config)
            for type in predict_config:
                if type in ["model", "dict", "re", "cmeee"]:
                    service_config = copy.deepcopy(predict_config[type])
                    service_config["model_path"] += f"/{i}"
                    ner_services.append(NerPredict(service_config, type=type))
            print(ner_services)
            output_file = args.output_dir + f"/test_pred_{i}.json"
            with open(args.input_file, "r", encoding="utf-8") as sf, \
                    open(output_file, "w", encoding="utf-8") as tf:
                lines = sf.readlines()
                for line in tqdm(lines):
                    line = json.loads(line)
                    if not line:
                        continue
                    text = line["text"]
                    entity_list = list()
                    for ner_service in ner_services:
                        entity_list += ner_service.predict(text)
                    # entity_list = merge_sym_rest(entity_list)
                    # entity_list = merge_entity_list(entity_list)
                    tf.write(json.dumps({
                        "text": text,
                        "ner": entity_list
                    }, ensure_ascii=False) + "\n")
    else:
        raise RuntimeError(f"{args.task}未开发")
