import pandas as pd
import matplotlib.pyplot as plt

idx = 8
path = './log/resutls_mlp/{}_predict.csv'.format(idx)

df = pd.read_csv(path)
df.plot()
plt.show()