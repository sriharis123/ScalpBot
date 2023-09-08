import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go

DIR = 'data'

# Collection variables
SEC_TO_MILLI = 1000
UNIX_DAY = 86400000
UNIX_HOUR = 3600000
# top 15 by market cap
INSTRUMENTS = ['BTC_USDT','ETH_USDT','XRP_USDT','DOGE_USDT','ADA_USDT','SOL_USDT','MATIC_USDT','LTC_USDT','DOT_USDT','SHIB_USDT','DAI_USDT']
REQ_COUNTS = {"1m":288, "5m":288, "15m":96, "30m":48, "1h":24}
SEC_PER_IDX = {'1m':60,'5m':300,'15m':900,'30m':1800,'1h':2900}

# Environment variables
CAPITAL=50
PMAXBUY=0.15
WINDOW=8
TAKERFEE=0.00075

# Agent variables
STATE_DIM=8*WINDOW
ACTION_DIM=3
LR_LAMBDA = lambda epoch: 0.95 ** (epoch//50)

def df_to_csv(df, fname='default.csv', directory=DIR):
    if fname==None or fname=='':
        raise AttributeError(f'{fname} not provided or empty')
    if not os.path.exists(directory):
        os.mkdir(directory)
    if '.csv' not in fname:
        fname += '.csv'
    df.reset_index(drop=True)
    df.to_csv(os.path.join(directory, fname), index=False)


def csv_to_df(fname):
    fpath = os.path.join(DIR, fname)
    if fname==None or not os.path.exists(fpath):
        raise AttributeError(f'{fname} or directory {DIR} does not exist')
    return pd.read_csv(fpath)

def visualize_go(df, markers=[]):
    data = []
    if len(markers):
        markers = np.array(markers)
        idx = np.where(markers!=0)[0]
        data.append(go.Scatter(x=df['t'].iloc[idx],
                    y=df['c'].iloc[idx],
                    mode="markers",
                    marker_color=np.select([df['t'].iloc[idx].mask(markers[idx]!=1,0)>0, df['t'].iloc[idx].mask(markers[idx]!=2,0)>0], ["orange", "purple"], "rgba(0,0,0,0)"),
                    marker_size=8))

    data.append(go.Candlestick(x=df['t'],
                    open=df['o'],
                    high=df['h'],
                    low=df['l'],
                    close=df['c']))
                    
    fig = go.Figure(data)
    fig.show()