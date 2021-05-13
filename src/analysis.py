# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from catnlp.ner.format import JsonFormat


def analysis_ner(source):
    ner_data = JsonFormat(source)
    ner_data.statistics()
    ner_data.draw_histogram(num_bins=100, density=False)
    ner_data.draw_hbar()
    plt.show()


if __name__ == "__main__":
    source_file = "resources/data/dataset/ner/zh/cluener/json/train.json"
    analysis_ner(source_file)
