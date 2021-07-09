# Data Analysis
import numpy as np
import matplotlib as plt

def AnalyticsAndGraph(Sequence):

    count_aminos = {}
    length_seqs = []
    for i, seq in enumerate(Sequence):
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
