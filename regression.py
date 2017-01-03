import numpy as np
import pandas as pd
import seaborn as sns

from util import columns, write_file
from scipy import linalg as LA # 重回帰分析
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #3Dplot


def regression_middle_away(is_log, df):
    df = df[np.isfinite(df['15to64'])]
    df = df[np.isfinite(df['out-migrant'])]
    pop_middle = df['15to64']
    num_away = df['out-migrant']

    plt.plot(pop_middle, num_away, 'o')
    plt.title('regression analysis')
    plt.xlabel('population of middle age')
    plt.ylabel('number of away from the city')

    model = pd.ols(y=num_away, x=pop_middle, intercept=True)
    print(model)
    plt.plot(model.x['x'], model.y_fitted, 'g-')
    if is_log is True:
        with open('data/text/regression_ma_log_model.txt', 'w') as f:
            f.write(str(model))
        plt.savefig('data/picture/regression_ma_log.png')
    else:
        with open('data/text/regression_ma_model.txt', 'w') as f:
            f.write(str(model))
        plt.savefig('data/picture/regression_ma.png')
    plt.show()


def regression_old_divorce(is_log, df):
    divorce = df['divorce']
    old = df['over64']

    plt.plot(old, divorce, 'o')
    plt.title('regression analysis')
    plt.xlabel('population of old people')
    plt.ylabel('divorce')

    model = pd.ols(y=divorce, x=old, intercept=True)
    print(model)
    plt.plot(model.x['x'], model.y_fitted, 'g-')
    if is_log is True:
        write_file('data/text/regression_od_log.txt', model)
        plt.savefig('data/picture/regression_od_log.png')
    else:
        write_file('data/text/regression_od.txt', model)
        plt.savefig('data/picture/regression_od.png')
    plt.show()


def calc_relation(is_log=True, df=None):
    from pandas.tools.plotting import scatter_matrix  # 散布図行列を求める
    assert df
    df.drop('name', axis=1)
    df_res = pd.DataFrame(np.array(df), columns=columns)
    scatter_matrix(df_res)
    if is_log is True:
        plt.savefig('data/picture/scatter_log.png')
    else:
        plt.savefig('data/picture/scatter.png')
    plt.show()


def main():
    df = pd.read_csv('data/csv/input.csv', encoding='utf-8')
    # calc_relation(df=df)
    regression_old_divorce(False, df)
    # regression_middle_away(True, df)


if __name__ == '__main__':
    main()
