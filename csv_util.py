import input_csv
import numpy as np
import pandas as pd


def to_log_pandas():
    df = pd.read_csv('input_csv/input.csv', header=-1, skiprows=1)
    del(df[12])

    df_arr = np.array(df).T
    res_arr = []
    for arr in df_arr:
        res_arr.append(list(map(lambda x: np.log(1 + x), arr)))

    df_res = pd.DataFrame(np.array(res_arr).T)
    df_res.to_csv('input_csv/input_log.input_csv')


def to_log_csv():
    csv_reader = csv.reader(open('input_csv/input.csv'))
