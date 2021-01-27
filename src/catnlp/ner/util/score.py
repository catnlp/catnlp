#!/usr/bin/python3
# -*- coding: utf-8 -*-

from .merge import get_interval

    
def get_f1(gold_lists, pred_lists, format):
    gold_set = set()
    pred_set = set()
    for i, (gold_list, pred_list) in enumerate(zip(gold_lists, pred_lists)):
        gold_entity_list = get_interval(gold_list, format)
        for entity in gold_entity_list:
            gold_set.add((i, entity["start"], entity["end"], entity["tag"]))
        pred_entity_list = get_interval(pred_list, format)
        for entity in pred_entity_list:
            pred_set.add((i, entity["start"], entity["end"], entity["tag"]))
    return f1_score(gold_set, pred_set)

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
    return f1
