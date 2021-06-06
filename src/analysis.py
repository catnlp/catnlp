# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from catnlp.ner.format import JsonFormat, ConllFormat


def analysis_ner(source, file_format, delimiter):
    if file_format == "conll":
        ner_data = ConllFormat(source, delimiter)
    else:
        ner_data = JsonFormat(source)
    ner_data.statistics()
    ner_data.draw_histogram(num_bins=100, density=False)
    ner_data.draw_hbar()
    plt.show()


if __name__ == "__main__":
    delimiter = "\t"
    file_format = "conll"
    source_file = "resources/data/dataset/ner/zh/ccks/address/bies/train.txt"
    analysis_ner(source_file, file_format, delimiter)
