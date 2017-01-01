import numpy as np
import pandas as pd
from columns import columns


def to_log_pandas():
    df = pd.read_csv('data/csv/input.csv')
    df_name = df['name']
    df_dropped = df.drop('name', axis=1)

    df_arr = np.array(df_dropped).T
    res_arr = []
    for arr in df_arr:
        res_arr.append(list(map(lambda x: np.log(1 + x), arr)))

    df_res = pd.DataFrame(np.array(res_arr).T)
    df_res['name'] = df_name
    df_res.columns = columns
    print(df_res)
    df_res.to_csv('data/csv/input_log.csv', index=None, index_label=None, encoding='utf-8')


if __name__ == '__main__':
    to_log_pandas()
