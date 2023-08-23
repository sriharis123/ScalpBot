import pandas as pd
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import ema_indicator, MACD
from ta.momentum import rsi
import util

class CandlestickData:

    def __init__(self, fname):
        self.df = util.csv_to_df(fname)
        self.name = fname
        self.indicators = {'ema': set(), 'macd': set(), 'rsi': set(), 'bb': set()}
        for ind in self.indicators:
            for col in list(self.df.columns.values):
                if ind in col.lower():
                    self.indicators[ind].add(col)
        print('currently existing indicators:',self.indicators)

    def add_EMA(self, ema_window=12):
        if f'ema_{ema_window}' in self.indicators['ema']:
            print(f'ema_{ema_window} already exists in {self.name}')
            return
        ema = ema_indicator(self.df['c'], window=ema_window, fillna=True)
        self.indicators['ema'].add(ema.name)
        self.df.loc[:,ema.name] = ema
        
    def add_RSI(self, rsi_window=14):
        if f'rsi_{rsi_window}' in self.indicators['rsi']:
            print(f'rsi_{rsi_window} already exists in {self.name}')
            return
        rser = rsi(self.df['c'], window=rsi_window, fillna=True)
        self.indicators['rsi'].add(rser.name)
        self.df.loc[:,rser.name] = rser

    def add_MACD(self, window_fast=12, window_slow=26, window_sign=9, mom=True, sig=True, dif=True):
        if (not mom or mom and f'MACD_{window_fast}_{window_slow}' in self.indicators['macd']) \
                and (not sig or sig and f'MACD_sign_{window_fast}_{window_slow}' in self.indicators['macd']) \
                and (not dif or dif and f'MACD_diff_{window_fast}_{window_slow}' in self.indicators['macd']) :
            print(f'MACD_{window_fast}_{window_slow} already exists in {self.name}')
            return
        macd = MACD(self.df['c'], window_fast=window_fast, window_slow=window_slow, window_sign=window_sign, fillna=True)
        if mom:
            mser = macd.macd()
            self.df.loc[:,mser.name] = mser
            self.indicators['macd'].add(mser.name)
        if sig:
            signal = macd.macd_signal()
            self.df.loc[:,signal.name] = signal
            self.indicators['macd'].add(signal.name)
        if dif:
            diff = macd.macd_diff()
            self.df.loc[:,diff.name] = diff
            self.indicators['macd'].add(diff.name)

    def add_BB(self, window=20, std=2, mov=True, hb=True, lb=True, wb=False, pb=False):
        bbname = f'BB_{window}_{std}'
        if (not mov or mov and f'{bbname}_mavg' in self.indicators['bb']) \
                and (not hb or hb and f'{bbname}_hband' in self.indicators['bb']) \
                and (not lb or lb and f'{bbname}_lband' in self.indicators['bb']) \
                and (not wb or wb and f'{bbname}_bbiwband' in self.indicators['bb']) \
                and (not pb or pb and f'{bbname}_bbipband' in self.indicators['bb']) :
            print(f'{bbname} already exists in {self.name}')
            return
        bb = BollingerBands(self.df['c'], window=window, window_dev=std, fillna=True)
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
        if pb:
            pband = bb.bollinger_pband()
            name = f'{bbname}_{pband.name}'
            self.df.loc[:,name] = pband
            self.indicators['bb'].add(name)

    def validate_cols(self, cols=[]):
        if type(cols) != list:
            raise AttributeError('cols must be a list of strings')
        for c in cols:
            if c not in list(self.df.columns.values):
                raise AttributeError(f'col {c} in cols does not exist')
        return True

    def remove(self, ind, cols=[]):
        self.validate_cols(cols)
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

    def set_pct_change(self, cols=[], period=1):
        self.validate_cols(cols)
        for c in cols:
            self.df[c] = self.df[c].pct_change(periods=period)

    def to_numpy(self):
        return self.df.to_numpy()

    def write_to_file(self, name=None, directory=None):
        self.df.reset_index(drop=True)
        name = self.name if name==None else name
        directory = util.DIR if directory==None else directory
        util.df_to_csv(self.df, name, directory)
    


def testema():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m")
    ada_usdt.add_EMA(20)
    print(ada_usdt.df.iloc[:20])
    ada_usdt.add_EMA(20)
    ada_usdt.write_to_file()
    ada_usdt.remove_ema()
    print(ada_usdt.df.iloc[:20])
    ada_usdt.write_to_file()

def testmacd():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m")
    ada_usdt.add_MACD()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    ada_usdt.remove_macd()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    
def testrsi():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m")
    ada_usdt.add_RSI()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    ada_usdt.remove_rsi()
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()
    
def testbb():
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m")
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
    ada_usdt = CandlestickData("ADA_USDT_dur_35_end_1691625600000_ts_1m")
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.remove('', cols=['mavg','hband','lband'])
    print(ada_usdt.df.iloc[2000:2020])
    ada_usdt.write_to_file()

def testinvalid():
    ada_usdt = CandlestickData("ADA_USDT_ur_35_end_1691625600000_ts_1m")
    ada_usdt.add_BB()
    print(ada_usdt.df.iloc[2000:2020])


    
testema()