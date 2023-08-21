import pandas as pd
import os
from datetime import datetime

DIR = 'data'

def df_to_csv(df, fname='default.csv'):
    if fname==None or fname=='':
        raise AttributeError(f'{fname} not provided or empty')
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    if '.csv' not in fname:
        fname += '.csv'
    df.reset_index(drop=True)
    df.to_csv(os.path.join(DIR, fname), index=False)


def csv_to_df(fname):
    fpath = os.path.join(DIR, fname)
    if fname==None or not os.path.exists(fpath):
        raise AttributeError(f'{fname} or directory {DIR} does not exist')
    return pd.read_csv(fpath)

# def df_to_quotes(df):
#     return [stock_indicators.Quote(datetime.fromtimestamp(t),o,h,l,c,v) for o,h,l,c,v,t in zip(df['o'],df['h'],df['l'],df['c'],df['v'],df['t'])]