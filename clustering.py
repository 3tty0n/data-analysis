# coding=utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from util import columns

df = pd.read_csv('data/csv/input.csv', encoding='utf-8')
df_name = df['name']
del df['name']
lst = map(lambda index: df[columns[index]].tolist(), range(0, len(columns) - 1))
numpy_df = np.array(list(lst), np.int32)
numpy_df = numpy_df.T
pred = KMeans(n_clusters=4).fit_predict(numpy_df)
df['cluster_id'] = pred

clusterinfo = pd.DataFrame()
for i in range(4):
    clusterinfo['cluster' + str(i)] = df[df['cluster_id'] == i].mean()

clusterinfo = clusterinfo.drop('cluster_id')
plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
plot.set_xticklabels(plot.xaxis.get_majorticklabels(), rotation=0)
plt.savefig('data/picture/clustering.png')
plt.show()
