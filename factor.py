import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.decomposition import PCA
from sklearn import svm

df = pd.read_csv('csv/input2.csv', header=-1, skiprows=1, encoding='utf-8')
del(df[1])
del(df[12])
pca = PCA(n_components=2, whiten=False)
pca.fit(df)
pca_point = pca.transform(df)

clf = svm.LinearSVC()

for i, point in enumerate(pca_point):
    plt.scatter(*point, marker=",")

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('data/factor.png')
plt.show()
