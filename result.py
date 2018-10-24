import pandas as pd
import matplotlib.pyplot as plt

# i=1
#
# path = 'results/mlp/{}_predict.csv'.format(i)
# print(path)
#
# df = pd.read_csv(path)
# df.plot()
# plt.xlabel(i)
# plt.legend()
# plt.show()
# # plt.clf()

path_result_mae = 'results/ed_mul_24_10/mae_rmse_log.csv'

df_mae = pd.read_csv(path_result_mae, header=None, names=['index', 'mae', 'rmse'], index_col=0)
df_mae = df_mae.sort_values('rmse')
df_mae = df_mae.iloc[:10]
print(df_mae.head(10))

df_mae.to_csv('results/ed_21_10/top_10_rmse.csv')