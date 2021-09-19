
import os
import json
import numpy as np
from catnlp.tool.draw import plot_confusion_matrix

tags = ["dis", "sym", "pro", "equ", "dru", "ite", "bod", "dep", "mic", "oth"]
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
matrix = np.zeros((len(tags), len(tags)), dtype=int)


def confusion_matrix(gold, pred):
    with open(gold, "r", encoding="utf-8") as gf, \
            open(pred, "r", encoding="utf-8") as pf:
        for gl, pl in zip(gf, pf):
            gl = json.loads(gl)
            pl = json.loads(pl)
            if not gl or not pl:
                continue
            g_entities = gl["ner"]
            p_entities = pl["ner"]
            g_set = set()
            for entity in g_entities:
                g_set.add(tuple(entity))
            p_set = set()
            for entity in p_entities:
                p_set.add(tuple(entity))
            pos2tag = dict()
            for entity in g_set:
                start, end, tag, _ = entity
                pos2tag[(start, end)] = tag
            
            for entity in p_set:
                start, end, tag, _ = entity
                row = tag2idx[tag]
                g_tag = pos2tag.get((start, end), None)
                if not g_tag:
                    column = tag2idx["oth"]
                else:
                    column = tag2idx[g_tag]
                matrix[row][column] += 1
            
            pos2tag = dict()
            for entity in p_set:
                start, end, tag, _ = entity
                pos2tag[(start, end)] = tag
            
            for entity in g_set:
                start, end, tag, _ = entity
                column = tag2idx[tag]
                p_tag = pos2tag.get((start, end), None)
                if not p_tag:
                    row = tag2idx["oth"]
                    matrix[row][column] += 1
            
        return matrix


if __name__ == "__main__":
    gold_file = "resources/data/dataset/ner/zh/ccks/cmeee/0808/test_pred_0.json"
    pred_file = "resources/data/dataset/ner/zh/ccks/cmeee/0814/test_pred_0.json"
    mat = confusion_matrix(gold_file, pred_file)
    # plot_confusion_matrix(mat, classes=tags, normalize=True)
    plot_confusion_matrix(mat, classes=tags, normalize=False)
