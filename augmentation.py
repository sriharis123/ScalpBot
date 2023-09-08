import pandas as pd
import numpy as np
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import ema_indicator, MACD
from ta.momentum import rsi
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pmdarima import arima
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import LogEndogTransformer
import util
import plotly.graph_objects as go

class CandlestickData:

    def __init__(self, fname):
        self.df = util.csv_to_df(fname)
        self.name = fname
        self.close_np = np.empty((0))
        self.indicators = {'ema': set(), 'macd': set(), 'rsi': set(), 'bb': set(), 'pct': set(), 'lr': set()}
        for ind in self.indicators:
            for col in list(self.df.columns.values):
                if ind in col.lower():
                    self.indicators[ind].add(col)
        print('currently existing indicators:',self.indicators)

    def add_EMA(self, ema_window=12, ind='c'):
        name = f'ema_{ema_window}_{ind}'
        if name in self.indicators['ema']:
            print(f'{name} already exists in {self.name}')
            return
        ema = ema_indicator(self.df[ind], window=ema_window, fillna=True)
        self.indicators['ema'].add(name)
        self.df.loc[:,name] = ema
        
    def add_RSI(self, rsi_window=14, ind='c'):
        name = f'rsi_{rsi_window}_{ind}'
        if name in self.indicators['rsi']:
            print(f'{name} already exists in {self.name}')
            return
        rser = rsi(self.df[ind], window=rsi_window, fillna=True) / 100.0
        self.indicators['rsi'].add(name)
        self.df.loc[:,name] = rser

    # should this be normalized? currently standard scale
    def add_MACD(self, window_fast=12, window_slow=26, window_sign=9, ind='c', mom=True, sig=True, dif=True):
        if (not mom or mom and f'MACD_{window_fast}_{window_slow}_{ind}' in self.indicators['macd']) \
                and (not sig or sig and f'MACD_{window_fast}_{window_slow}_{ind}_sign' in self.indicators['macd']) \
                and (not dif or dif and f'MACD_{window_fast}_{window_slow}_{ind}_diff' in self.indicators['macd']) :
            print(f'MACD_{window_fast}_{window_slow}_{ind} already exists in {self.name}')
            return
        macd = MACD(self.df[ind], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign, fillna=True)
        scaler = StandardScaler()
        if mom:
            name = f'MACD_{window_fast}_{window_slow}_{ind}'
            mser = macd.macd()
            mser = scaler.fit_transform(mser.to_numpy().reshape(-1,1))
            self.df.loc[:,name] = mser
            self.indicators['macd'].add(name)
        if sig:
            name = f'MACD_{window_fast}_{window_slow}_{ind}_sign'
            signal = macd.macd_signal()
            signal = scaler.fit_transform(signal.to_numpy().reshape(-1,1))
            self.df.loc[:,name] = signal
            self.indicators['macd'].add(name)
        if dif:
            name = f'MACD_diff_{window_fast}_{window_slow}_{ind}_diff'
            diff = macd.macd_diff()
            diff = scaler.fit_transform(diff.to_numpy().reshape(-1,1))
            self.df.loc[:,name] = diff
            self.indicators['macd'].add(name)

    """
    ind: indicator to use (default: close price)
    pb: apply percent calculation on cur price vs bands. useful for normalization.
    """
    def add_BB(self, window=20, std=2, ind='c', mov=True, hb=True, lb=True, wb=False, pb=False):
        bbname = f'BB_{window}_{std}_{ind}'
        if (not mov or mov and f'{bbname}_mavg' in self.indicators['bb']) \
                and (not hb or hb and f'{bbname}_hband' in self.indicators['bb']) \
                and (not lb or lb and f'{bbname}_lband' in self.indicators['bb']) \
                and (not wb or wb and f'{bbname}_bbiwband' in self.indicators['bb']) \
                and (not pb or pb and f'{bbname}_bbipband' in self.indicators['bb']) :
            print(f'{bbname} already exists in {self.name}')
            return
        bb = BollingerBands(self.df[ind], window=window, window_dev=std, fillna=True)
        if mov:
            bbser = bb.bollinger_mavg()
            name = f'{bbname}_{bbser.name}'
            self.df.loc[:,name] = bbser
            self.indicators['bb'].add(name)
        if hb:
            hband = bb.bollinger_hband()
            name = f'{bbname}_{hband.name}'
            self.df.loc[:,name] = hband
            self.indicators['bb'].add(name)
        if lb:
            lband = bb.bollinger_lband()
            name = f'{bbname}_{lband.name}'
            self.df.loc[:,name] = lband
            self.indicators['bb'].add(name)
        if wb:
            wband = bb.bollinger_wband()
            name = f'{bbname}_{wband.name}'
            self.df.loc[:,name] = wband
            self.indicators['bb'].add(name)
        if pb: # standard scaling applied, so that val is not only between 0 and 1.
            pband = bb.bollinger_pband()#.clip(0,1)
            scaler = StandardScaler()
            name = f'{bbname}_{pband.name}'
            self.df.loc[:,name] = scaler.fit_transform(pband.to_numpy().reshape(-1,1))
            self.indicators['bb'].add(name)           

    def add_pct_change(self, cols=[], period=1):
        self.validate_cols(cols, True)
        for c in cols:
            self.df[f'{c}_pct_{period}'] = self.df[c].pct_change(periods=period)
            self.indicators['pct'].add(f'{c}_pct_{period}')

    # https://gregorygundersen.com/blog/2022/02/06/log-returns/: log returns follow a lognormal distribution
    def add_log_return(self, cols=[], period=1):
        self.validate_cols(cols, True)
        for c in cols:
            self.df[f'{c}_lr_{period}'] = np.log(self.df[c]) - np.log(self.df[c].shift(period))
            self.indicators['lr'].add(f'{c}_lr_{period}')

    def add_future_lr(self, cols=[], periods=[1]):
        self.validate_cols(cols, True)
        periods.sort()
        for c in cols:
            frames = pd.DataFrame()
            future_name = f'flr_{c}_{len(periods)}'
            for p in periods:
                frames[p] = np.log(self.df[c].shift(-p)) - np.log(self.df[c])
            self.df[future_name] = frames[frames.columns[::-1]].ewm(span=len(periods), axis=1, adjust=False).mean()[0]
        

    def clip(self, cols=[], low=-1, high=1, divide=1):
        self.validate_cols(cols, True)
        for c in cols:
            self.df[c] = (self.df[c] / divide).clip(low, high) / 2 + 0.5

    # Doesn't really work. Prediction is too linear
    def forecast_arima(self, predict_idx, col='c', return_gt=False):
        self.validate_cols([col])
        if self.close_np.shape == (0,):
            self.close_np = self.df['c'].to_numpy()
        train = self.close_np[predict_idx - (util.N_TRAIN + util.N_TEST):predict_idx - util.N_TEST]
        test = self.close_np[predict_idx - util.N_TEST:predict_idx]
        pipe = Pipeline([('log', LogEndogTransformer()), ('arima', arima.ARIMA(order=(10,0,20)))])

        pipe.fit(train)
        preds = pipe.predict(util.N_PREDICT + util.N_TEST, return_conf_int=False)
        
        if return_gt:
            return self.close_np[predict_idx-(util.N_TRAIN + util.N_TEST):predict_idx + util.N_PREDICT], preds
        return preds[util.N_TEST:]

    def validate_cols(self, cols=[], check_empty=False):
        if type(cols) != list or (check_empty and len(cols) == 0):
            raise AttributeError('cols must be a non-empty list of strings')
        for c in cols:
            if c not in list(self.df.columns.values):
                raise AttributeError(f'col {c} in cols does not exist')
        return True

    def remove(self, ind, cols=[]):
        self.validate_cols(cols, False)
        if ind in self.indicators and (cols==[] or cols==None):
            cols = self.indicators[ind]
        self.df.drop(columns=cols, inplace=True)

    def remove_ema(self, cols=[]):
        self.remove('ema', cols)

    def remove_macd(self, cols=[]):
        self.remove('macd', cols)

    def remove_rsi(self, cols=[]):
        self.remove('rsi', cols)

    def remove_bb(self, cols=[]):
        self.remove('bb', cols)

    def remove_pct_change(self, cols=[]):
        self.remove('pct', cols)
    
    def remove_log_return(self, cols=[]):
        self.remove('lr', cols)

    def dropna(self):
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def to_numpy(self):
        return self.df.to_numpy()

    def write_to_file(self, name=None, directory=None):
        self.df.reset_index(drop=True, inplace=True)
        name = self.name if name==None else name
        directory = util.DIR if directory==None else directory
        util.df_to_csv(self.df, name, directory)
    


def testema():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_EMA(20)
    print(ada_usdt.df.iloc[:20])
    ada_usdt.add_EMA(20)
    ada_usdt.write_to_file()
    ada_usdt.remove_ema()
    print(ada_usdt.df.iloc[:20])
    ada_usdt.write_to_file()

def testmacd():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_MACD()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    ada_usdt.remove_macd()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    
def testrsi():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_RSI()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    ada_usdt.remove_rsi()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    
def testbb():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_BB()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    ada_usdt.add_BB()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.add_BB(wb=True)
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.remove_bb()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()

def removespec():
    ada_usdt = CandlestickData("XRP_USDT_dur_35_end_1691625600000_ts_1m.csv")
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.remove('', cols=['Unnamed: 0'])
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()

def testinvalid():
    ada_usdt = CandlestickData("ADA_USDT_ur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_BB()
    print(ada_usdt.df.iloc[2000:2020])

def testpctlr():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m.csv")
    ada_usdt.add_pct_change(cols=['v'], period=1)
    ada_usdt.add_log_return(cols=['c'], period=1)
    ada_usdt.add_log_return(cols=['c'], period=5)
    print(ada_usdt.df.iloc[0:40])
    ada_usdt.write_to_file()
    ada_usdt.remove_pct_change()
    ada_usdt.remove_log_return()
    print(ada_usdt.df.iloc[0:40])
    ada_usdt.write_to_file()

# test feature engineering
def testmulti():
    dot_usdt = CandlestickData("DOT_USDT_dur_35_end_1691625600000_ts_1m.csv")
    
    dot_usdt.add_RSI()
    dot_usdt.add_log_return(cols=['c'], period=1)                           # log return of the close
    dot_usdt.add_log_return(cols=['c'], period=5)                           # log return of the close. period 5
    dot_usdt.add_BB(mov=False, hb=False, lb=False, pb=True)                 # percentB of the bollinger band on close
    dot_usdt.add_BB(ind='c_lr_5', mov=False, hb=False, lb=False, pb=True)   # bollinger bands surrounding log returns
    dot_usdt.add_MACD(mom=False, sig=False)                                 # macd indicator. only shows diff
    # dot_usdt.clip(cols=['BB_20_2_c_bbipband', 'BB_20_2_c_lr_1_bbipband'], divide=2)
    # dot_usdt.clip(cols=['MACD_diff_12_26_c_diff'], divide=3)
    dot_usdt.dropna()

    # dot_usdt.remove_bb()
    # dot_usdt.remove_macd()
    # dot_usdt.remove_log_return()
    # dot_usdt.remove_rsi()

    print(dot_usdt.df.iloc[0:40])
    util.visualize_go(dot_usdt.df.iloc[0:40])

def testfore():
    dot_usdt = CandlestickData("DOT_USDT_dur_35_end_1691625600000_ts_1m.csv")
    trainset, forecast = dot_usdt.forecast_arima(1000, return_gt=True)
    print(trainset.shape, forecast.shape)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(util.N_PREDICT + util.N_TEST + util.N_TRAIN),
        y=trainset,
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(start=util.N_TRAIN, stop=util.N_TRAIN + util.N_TEST + util.N_PREDICT),
        y=forecast,
        connectgaps=True
    ))
    fig.show()

def test_logreturns_ema():
    dot_usdt = CandlestickData("DOT_USDT_dur_35_end_1691625600000_ts_1m.csv")

    dot_usdt.add_EMA(ema_window=20)
    dot_usdt.add_future_lr(cols=['ema_20_c'], periods=np.arange(200))

    df = dot_usdt.df.iloc[1000:3000]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['ema_20_c'],
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['flr_ema_20_c_200'] + df['ema_20_c'],
        connectgaps=True
    ))
    fig.show()



# removespec()
# testmulti()
# testema()
# testfore()
test_logreturns_ema()