# Data Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from Modules.Model_Spec import defaults

con = sqlite3.connect(defaults.SQLite_DB_PATH)
dataset = pd.read_sql_query("SELECT sequence_string FROM Entries", con)

Sequence = dataset['sequence_string']


def AnalyticsAndGraph(sequence):
    count_aminos = {}
    length_seqs = []
    for i, seq in enumerate(sequence):
        length_seqs.append(len(seq))
        for a in seq:
            if a in count_aminos:
                count_aminos[a] += 1
            else:
                count_aminos[a] = 0

    unique_aminos = list(count_aminos.keys())

    print('Unique aminos ({}):\n{}'.format(len(unique_aminos), unique_aminos))
    x = [i for i in range(len(unique_aminos))]
    plt.bar(x, count_aminos.values())
    plt.xticks(x, unique_aminos)
    print(list(count_aminos.values())[-5:])
    plt.show()

    print('Average length:', np.mean(length_seqs))
    print('Deviation:', np.std(length_seqs))
    print('Min length:', np.min(length_seqs))
    print('Max length:', np.max(length_seqs))

    print('Average length:', np.mean(length_seqs))
    print('Deviation:', np.std(length_seqs))
    print('Min length:', np.min(length_seqs))
    print('Max length:', np.max(length_seqs))

# def histogram():
#
#     sorted_seqs = np.array(length_seqs)
#     sorted_seqs.sort()
#     print('10 shortest:\n{}\n10 longest:\n{}'.format(sorted_seqs[:10], sorted_seqs[-10:]))
#
#     print("Number of Sequences: ", SequenceSize)
#     print('Number sequences less than 30 AA:', len(sorted_seqs[sorted_seqs < 30]))
#     print('Number sequences more than 500 AA:', len(sorted_seqs[sorted_seqs > 500]))
#     print('Number sequences more than 1000 AA:', len(sorted_seqs[sorted_seqs > 1000]))
#
#     # density={}
#
#     # for i in range(np.max(length_seqs)):
#     #     lower=len(sorted_seqs[sorted_seqs<i])
#     #     upper=len(sorted_seqs[sorted_seqs>i+2])
#
#     #     print ("Lower  ",lower,"    ","Upper   ",upper,"       ","Sequence Length",np.max(length_seqs))
#
#     #     calc=(SequenceSize)-abs(lower)-abs(upper)
#     #     calc=abs(calc)
#
#     #     density[i]=calc
#     #     # print(i, " / ", np.max(length_seqs))
#
#     # lists = sorted(density.items()) # sorted by key, return a list of tuples
#     # print(density)
#     # x, y = zip(*lists) # unpack a list of pairs into two tuples
#
#     # plt.plot(x, y)
#     # plt.xlabel('Number of Sequencees')
#     # plt.ylabel('The Length of the Sequence')
#     # plt.title('Sequence Size Distribution')
#
#     # plt.show()
#     Optimized_length_seq = []
#
#     for item in length_seqs:
#         if (item < 1000):
#             Optimized_length_seq.append(item)
#
#     N_points = 10000
#     n_bins = 200
#     legend = ['distribution']
#
#     fig, axs = plt.subplots(1, 1,
#                             figsize=(10, 7),
#                             tight_layout=True)
#
#     # Remove axes splines
#     for s in ['top', 'bottom', 'left', 'right']:
#         axs.spines[s].set_visible(False)
#
#         # Remove x, y ticks
#     axs.xaxis.set_ticks_position('none')
#     axs.yaxis.set_ticks_position('none')
#
#     # Add padding between axes and labels
#     axs.xaxis.set_tick_params(pad=5)
#     axs.yaxis.set_tick_params(pad=10)
#
#     # Add x, y gridlines
#     axs.grid(b=True, color='grey',
#              linestyle='-.', linewidth=0.5,
#              alpha=0.6)
#
#     # Add Text watermark
#     fig.text(0.9, 0.15, 'Proten Function Prediction',
#              fontsize=12,
#              color='red',
#              ha='right',
#              va='bottom',
#              alpha=0.7)
#
#     # Creating histogram
#     N, bins, patches = axs.hist(Optimized_length_seq, bins=n_bins)
#
#     # Setting color
#     fracs = ((N ** (1 / 5)) / N.max())
#     norm = colors.Normalize(fracs.min(), fracs.max())
#
#     for thisfrac, thispatch in zip(fracs, patches):
#         color = plt.cm.viridis(norm(thisfrac))
#         thispatch.set_facecolor(color)
#
#     # Adding extra features
#     plt.xlabel("Sequence Length")
#     plt.ylabel("Number of Sequences")
#     plt.legend(legend)
#     plt.title('Sequence Length Distribution')
#
#     # Show plot
#     plt.show()


def main():

    AnalyticsAndGraph(Sequence)
    # histogram()
