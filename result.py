import pandas as pd
import matplotlib.pyplot as plt

idx = range(0,2)
for i in idx:
    path = 'results/mlp/{}_predict.csv'.format(i)
    print(path)

    df = pd.read_csv(path)
    df.plot()
    plt.xlabel(i)
    plt.legend()
    plt.show()
    # plt.clf()