import csv
import numpy as np
import pandas as pd


def to_log_pandas():
    df = pd.read_csv('data/csv/input.csv', header=None, skiprows=1, encoding='utf-8')
    df = df.drop(12, axis=1)

    df_arr = np.array(df).T
    res_arr = []
    for arr in df_arr:
        res_arr.append(list(map(lambda x: np.log(1 + x), arr)))

    columns = [
        'total', 'under15', '15to64', 'over64', 'birth', 'death',
        'transferee', 'out-migrant', 'daytime', 'elder', 'marriage', 'divorce']

    df_res = pd.DataFrame(np.array(res_arr).T, columns=columns)
    print(df_res)
    df_res.to_csv('data/csv/input_log.csv', index=None, index_label=None)


if __name__ == '__main__':
    to_log_pandas()
