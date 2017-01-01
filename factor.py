import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn import svm


def read_csv(is_log):
    """
    csvを読み込む
    
    :param is_log: log(1+x)とスケール変換されたファイルを読み込むフラグ
    :return:
    """
    if is_log is True:
        df = pd.read_csv('data/csv/input_log.csv', encoding='utf-8')
    else:
        df = pd.read_csv('data/csv/input.csv', encoding='utf-8')
    return df


def calc_pca(df):
    """
    主成分分析を行う。

    :param df:
    :return:
    """
    df = df.drop('name', axis=1)

    pca = PCA(n_components=2, whiten=False)
    pca_point = pca.fit_transform(df)

    return pca, pca_point


def pca_components(is_log=True):
    """
    因子負荷量、寄与率、累積寄与率を計算しプロットする。

    因子負荷量 : • 各変数の各主成分への影響⼒ → 各主成分の意味の推定
    寄与率 : 各主成分の重要性
    累積寄与率 : 主成分の寄与率を⾜し合わせたもの
                選択した複数の主成分によって説明できるデータの割合を表す

    :param is_log:
    :return:
    :see: http://i.cla.kobe-u.ac.jp/murao/class/2015-SeminarB3/05_Python_de_PCA.pdf
    """
    df = read_csv(is_log)
    pca, pca_point = calc_pca(df)
    components = pca.components_
    ratio = np.cumsum(pca.explained_variance_ratio_)
    print(components)
    print(ratio)


def factor(is_log=True, is_save=False, fp=None):
    """
    主成分分析を行った結果をプロットする。

    :param is_log:
    :param is_save:
    :param fp:
    :return:
    """
    assert fp

    df = read_csv(is_log)
    df_index = df['name']
    pca, pca_point = calc_pca(df)

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


def main():
    fp = FontProperties(fname='/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc', size=8)
    factor(fp=fp, is_save=False)
    pca_components()


if __name__ == '__main__':
    main()
