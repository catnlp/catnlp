# -*- coding: utf-8 -*-
from collections import defaultdict

from prettytable import PrettyTable

from .merge import get_interval

    
def get_f1(gold_lists, pred_lists, format):
    if format != "bies":
        format = "bio"
    gold_type_dict = defaultdict(set)
    pred_type_dict = defaultdict(set)
    for i, (gold_list, pred_list) in enumerate(zip(gold_lists, pred_lists)):
        gold_entity_list = get_interval(gold_list, format=format)
        for entity in gold_entity_list:
            start = entity["start"]
            end = entity["end"]
            tag = entity["tag"]
            gold_type_dict["total"].add((i, start, end, tag))
            gold_type_dict[tag].add((i, start, end, tag))
        pred_entity_list = get_interval(pred_list, format=format)
        for entity in pred_entity_list:
            start = entity["start"]
            end = entity["end"]
            tag = entity["tag"]
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
