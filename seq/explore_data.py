import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 34)
sample = pd.read_csv('sample.csv')
sample = sample.set_index('timestamp')
sample_describe = sample.describe()
sample_correlation = sample.corr()

# plot correlation
sample_correlation = sample_correlation.dropna(axis=[0, 1], how='all')
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 200
plt.matshow(sample_correlation)
plt.xticks(range(sample_correlation.shape[1]), sample_correlation.columns, fontsize=12, rotation=45)
plt.yticks(range(sample_correlation.shape[1]), sample_correlation.columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# drop all zero columns
sample = sample.loc[:, (sample!=0).any(axis = 0)]

# sample.shape
# (18417, 34)
# 18,417 individual events
# 18,386 timestamps
# classes:
# 0    10101
# 1     8316

class_0 = sample[sample['class'] == 0]
class_1 = sample[sample['class'] == 1]

for col in class_1.columns:
    class_0[col].plot(style='.')
    plt.show()
    class_1[col].plot(style='.')
    plt.show()

res = pd.read_csv('res.csv')
# res.shape
# (861383, 2)
