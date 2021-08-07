# -*- coding: utf-8 -*-
import argparse
import json
from collections import defaultdict
from prettytable import PrettyTable


def get_f1(gold_lists, pred_lists):
    gold_type_dict = defaultdict(set)
    pred_type_dict = defaultdict(set)
    for i, (gold_list, pred_list) in enumerate(zip(gold_lists, pred_lists)):
        for entity in gold_list:
            try:
                start, end, tag = entity
            except Exception as e:
                start, end, tag, _ = entity
            gold_type_dict["total"].add((i, start, end, tag))
            gold_type_dict[tag].add((i, start, end, tag))
        for entity in pred_list:
            try:
                start, end, tag = entity
            except Exception as e:
                start, end, tag, _ = entity
            pred_type_dict["total"].add((i, start, end, tag))
            pred_type_dict[tag].add((i, start, end, tag))
    result_dict = {}
    for tag in gold_type_dict:
        result_dict[tag] = f1_score(gold_type_dict[tag], pred_type_dict[tag])
    table = pretty_print(result_dict)
    return result_dict["total"]["F1"], table


def pretty_print(result_dict):
    table = PrettyTable()
    table.field_names = ["Label", "P", "R", "F1", "Equal"]
    for tag in sorted(result_dict.keys()):
        if tag != "total":
            score = result_dict[tag]
            table.add_row([tag, score["P"], score["R"], score["F1"], score["Equal"]])
    score = result_dict["total"]
    table.add_row(["total", score["P"], score["R"], score["F1"], score["Equal"]])
    return table


def f1_score(gold_set, pred_set):
    gold_num = len(gold_set)
    pred_num = len(pred_set)
    equal_num = len(gold_set & pred_set)
    if pred_num <= 0:
        p = 0
    else:
        p = float(equal_num) / pred_num
    if gold_num <= 0:
        r = 0
    else:
        r = float(equal_num) / gold_num
    
    if not p or not r:
        f1 = 0
    else:
        f1 = float(2 * p * r) / (p + r)
    return {"P": p, "R": r, "F1": f1, "Equal": equal_num}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估结果")
    parser.add_argument("--gold_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/cmeee/0807/test.json", help="测试文件")
    parser.add_argument("--pred_file", type=str,
                    default="resources/data/dataset/ner/zh/ccks/cmeee/0807/test_pred.json", help="结果文件")
    parser.add_argument("--score_file", type=str,
                        default="resources/data/dataset/ner/zh/ccks/cmeee/0807/score.text", help="预测配置")
    args = parser.parse_args()

    with open(args.gold_file, "r", encoding="utf-8") as gf, \
            open(args.pred_file, "r", encoding="utf-8") as pf, \
            open(args.score_file, "w", encoding="utf-8") as sf:
        gold_lists = list()
        pred_lists = list()
        for gold_line, pred_line in zip(gf, pf):
            gold_line = json.loads(gold_line)
            pred_line = json.loads(pred_line)
            if not gold_line or not pred_line:
                continue
            gold_entities = gold_line["ner"]
            pred_entities = pred_line["ner"]
            gold_lists.append(gold_entities)
            pred_lists.append(pred_entities)
        _, table = get_f1(gold_lists, pred_lists)
        print(table)
        sf.write(str(table))
