# coding=utf-8
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


"""
0: 人口総数,
1: "15歳未満人口",
2: "15〜64歳人口",
3: "65歳以上人口",
4: 出生数,
5: 死亡数,
6: 転入者数,
7: 転出者数,
8: 昼間人口,
9: "高齢単身世帯数",
10: 婚姻件数,
11: 離婚件数,
12: 政府統計の総合窓口より2012年の人口状況(市町村名)
"""

df = pd.read_csv('data/csv/input.input_csv', header=None, skiprows=1, encoding='utf-8')
del(df[0])
del(df[4])
del(df[5])
del(df[6])
del(df[11])
numpy_df = np.array([df[1].tolist(),
                     df[2].tolist(),
                     df[3].tolist(),
                     df[7].tolist(),
                     df[8].tolist(),
                     df[9].tolist(),
                     df[10].tolist()], np.int32)
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
