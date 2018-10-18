import pandas as pd
import matplotlib.pyplot as plt

i=1

path = 'results/mlp/{}_predict.csv'.format(i)
print(path)

df = pd.read_csv(path)
df.plot()
plt.xlabel(i)
plt.legend()
plt.show()
# plt.clf()