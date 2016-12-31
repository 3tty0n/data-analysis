import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition as dec


def cluster_v1():
    df = pd.read_csv('input_csv/input.csv')
    del(df['name'])
    X = df
    y = df['daytime']
    X_bis = dec.RandomizedPCA().fit_transform(X)
    plt.scatter(X_bis[:, 0], X_bis[:, 1], c=df['total'], s=30, cmap=plt.cm.rainbow)
    plt.show()


def cluster_v2():
    df = pd.read_csv('input_csv/input.csv', header=None, skiprows=1, encoding='utf-8')
    labels = df[1]
    df.drop(1)
    del df[12]
    pca_2 = dec.PCA(2)
    plot_colums = pca_2.fit_transform(df)
    plt.scatter(x=plot_colums[:,0], y=plot_colums[:,1], c=labels)
    plt.show()


if __name__ == '__main__':
    cluster_v2()
