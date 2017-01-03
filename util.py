import numpy as np
import pandas as pd

columns = [
    'total', 'under15', '15to64', 'over64', 'birth', 'death',
    'transferee', 'out-migrant', 'daytime', 'elder', 'marriage', 'divorce', 'name'
]


def to_log_pandas():
    df = pd.read_csv('data/csv/input.csv')
    df_name = df['name']
    df_dropped = df.drop('name', axis=1)

    df_arr = np.array(df_dropped).T
    res_arr = list(map(lambda x: np.log(1 + x), map(lambda arr: arr, df_arr)))

    df_res = pd.DataFrame(np.array(res_arr).T)
    df_res['name'] = df_name
    df_res.columns = columns
    print(df_res)
    df_res.to_csv('data/csv/input_log.csv', index=None, index_label=None, encoding='utf-8')


def write_file(path, text):
    f = open(path, 'w')
    f.write(str(text))
    f.close()

if __name__ == '__main__':
    to_log_pandas()
