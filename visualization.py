import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

df = pd.read_csv("data/ADA_USDT_dur_35_end_1691625600000_ts_5m")

fig = go.Figure(data=[go.Candlestick(x=df['t'],
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'])])

fig.show()