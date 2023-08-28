import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

df = pd.read_csv("data/DOT_USDT_dur_597_end_1692576000000_ts_1m.csv")

fig = go.Figure(data=[go.Candlestick(x=df['t'],
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'])])

fig.show()