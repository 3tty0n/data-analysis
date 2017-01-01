import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn import svm


def main():
    fp = FontProperties(fname='/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc', size=8)
    factor(fp=fp, is_save=True)


def read_csv(is_log):
    if is_log is True:
        df = pd.read_csv('data/csv/input_log.csv', encoding='utf-8')
    else:
        df = pd.read_csv('data/csv/input.csv', encoding='utf-8')
    return df


def factor(is_log=True, is_save=False, fp=None):
    assert fp

    df = read_csv(is_log)

    df_index = df['name']
    df = df.drop('name', axis=1)
    pca = PCA(n_components=2, whiten=False)
    pca_point = pca.fit_transform(df)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    def annotate(txt):
        ax.annotate(txt, (pca_point[i, 0], pca_point[i, 1]), fontproperties=fp)

    def scatter(c):
        plt.scatter(pca_point[i, 0], pca_point[i, 1], c=c)

    for i, txt in enumerate(df_index):
        if '村' in txt:
            annotate(txt)
            scatter('r')
        elif '区' in txt:
            annotate(txt)
            scatter('y')
        else:
            scatter('b')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if is_save is True and is_log is True:
        plt.savefig('data/picture/factor_log.png')
    elif is_save is True and is_log is False:
        plt.savefig('data/picture/factor.png')
    plt.show()

if __name__ == '__main__':
    main()
