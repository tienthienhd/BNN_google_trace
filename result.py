import pandas as pd
import matplotlib.pyplot as plt

idx = range(9, 10)
for i in idx:
    path = 'log/results_mlp/{}_predict.csv'.format(i)

    df = pd.read_csv(path)
    df.plot()
    plt.xlabel(i)
    plt.legend()
    plt.show()
    # plt.clf()