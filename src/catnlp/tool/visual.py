# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def draw_histogram(datas, num_bins=50, density=True, xlabel="Len", ylabel="Num", title="Histogram of Length"):
     def get_ymax(rects):
          max_y = 0
          for rect in rects:
               val_y = rect.get_height()
               if val_y > max_y:
                    max_y = val_y
          return max_y
     fig, ax = plt.subplots()

     # the histogram of the data
     n, bins, patches = ax.hist(datas, num_bins, density=density)
     ax.set_xlabel(xlabel)
     ax.set_ylabel(ylabel)
     ax.set_title(title)
     x_val = datas.mean()
     ymax = get_ymax(patches)
     ax.vlines(x=x_val, ymin=0, ymax=ymax, color='r', linestyle='-')
     ax.text(x_val, ymax/2, round(x_val, 1), color='r')

     # Tweak spacing to prevent clipping of ylabel
     fig.tight_layout()


def draw_hbar(labels, datas):
     def auto_text(ax, rects):
          for rect in rects:
               ax.text(rect.get_width()+1, rect.get_y()+2*rect.get_height()/3, rect.get_width(), color='r')
     plt.rcdefaults()
     fig, ax = plt.subplots()

     # Example data
     y_pos = np.arange(len(labels))

     rects = ax.barh(y_pos, datas, align='center')
     ax.set_yticks(y_pos)
     ax.set_yticklabels(labels)
     ax.invert_yaxis()  # labels read top-to-bottom
     ax.set_xlabel('Num')
     ax.set_title('Num of Label')
     auto_text(ax, rects)


def draw_scatter(x, y, s, c):
     fig, ax = plt.subplots()
     ax.scatter(x, y, s, c)

if __name__ == "__main__":
     # Fixing random state for reproducibility
     np.random.seed(19680801)

     x, y, s, c = np.random.rand(4, 30)
     c[:10] = [1] * 10
     c[10:20] = [2] * 10
     c[20:] = [3] * 10
     s *= 10**2.
     draw_scatter(x, y, s, c)

     plt.show()