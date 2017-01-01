import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn import svm


def main():
    fp = FontProperties(fname='/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc', size=8)
    factor(fp=fp)


def factor(is_log=True, is_tmp=False, fp=None):
    assert fp

    if is_log is True:
        df = pd.read_csv('data/csv/input_log.csv', encoding='utf-8')
    else:
        df = pd.read_csv('data/csv/input.csv', encoding='utf-8')
    df_index = df['name']
    df = df.drop('name', axis=1)
    pca = PCA(n_components=2, whiten=False)
    pca_point = pca.fit_transform(df)

    clf = svm.LinearSVC()

    # for i in range(len(df_index)):
    #     plt.scatter(pca_point[i, 0], pca_point[i, 1])

    fig, ax = plt.subplots()
    ax.scatter(pca_point[:, 0], pca_point[:, 1])
    for i, txt in enumerate(df_index):
        if '村' in txt:
            ax.annotate(txt, (pca_point[i, 0], pca_point[i, 1]), fontproperties=fp)
        else:
            ax.annotate('', (pca_point[i, 0], pca_point[i, 1]), fontproperties=fp)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if is_tmp is True:
        if is_log is True: plt.savefig('data/picture/factor_log_tmp.png')
        else: plt.savefig('data/picture/factor_tmp.png')
    else:
        if is_log is True: plt.savefig('data/picture/factor_log.png')
        else: plt.savefig('data/picture/factor.png')
    plt.show()

if __name__ == '__main__':
    main()
