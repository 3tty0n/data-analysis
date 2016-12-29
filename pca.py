import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('csv/input_log.csv', header=-1, skiprows=1)
del(df[0])
for i in range(4, 12):
    del(df[i])
pca = PCA(n_components=2)
pca.fit(df)
pca = pca.transform(df)
np_array = np.array(pca)
np_array = np_array.T
plt.scatter(np_array[0], np_array[1])
plt.show()
