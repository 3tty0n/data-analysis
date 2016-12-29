import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg as LA # 重回帰分析
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #3Dplot


def regression_middle_away(df):
    df = df[np.isfinite(df[2])]
    df = df[np.isfinite(df[7])]
    pop_middle = df[2]
    num_away = df[7]

    plt.plot(pop_middle, num_away, 'o')
    plt.title('regression analysis')
    plt.xlabel('population of middle age')
    plt.ylabel('number of away from the city')

    model = pd.ols(y=num_away, x=pop_middle, intercept=True)
    print(model)
    plt.plot(model.x['x'], model.y_fitted, 'g-')
    plt.savefig('data/regression.png')
    plt.show()


def regression_old_divorce(df):
    divorce = df[11]
    old = df[4]

    plt.plot(old, divorce, 'o')
    plt.title('regression analysis')
    plt.xlabel('population of old people')
    plt.ylabel('divorce')

    model = pd.ols(y=divorce, x=old, intercept=True)
    print(model)
    plt.plot(model.x['x'], model.y_fitted, 'g-')
    plt.savefig('data/regression_old_divorce.png')
    plt.show()


def calc_relation(df):
    from pandas.tools.plotting import scatter_matrix  # 散布図行列を求める
    plt.figure()
    plt.savefig('data/scatter.png')
    scatter_matrix(df)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('csv/input.csv', header=-1, skiprows=1, encoding='utf-8')
    # regression_middle_away(df)
    # regression_old_divorce(df)
    calc_relation(df)


