# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def draw_histogram(datas, num_bins=50, density=True, xlabel="Len", ylabel="Num", title="Histogram of Length"):
     fig, ax = plt.subplots()

     # the histogram of the data
     n, bins, patches = ax.hist(datas, num_bins, density=density)
     ax.set_xlabel(xlabel)
     ax.set_ylabel(ylabel)
     ax.set_title(title)

     # Tweak spacing to prevent clipping of ylabel
     fig.tight_layout()


def draw_hbar(labels, datas):
     plt.rcdefaults()
     fig, ax = plt.subplots()

     # Example data
     y_pos = np.arange(len(labels))

     ax.barh(y_pos, datas, align='center')
     ax.set_yticks(y_pos)
     ax.set_yticklabels(labels)
     ax.invert_yaxis()  # labels read top-to-bottom
     ax.set_xlabel('Num')
     ax.set_title('Num of Label')
