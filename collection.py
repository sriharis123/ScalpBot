import requests

import time
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
import util

SEC_TO_MILLI = 1000
UNIX_DAY = 86400000
VERBOSE = False
# top 15 by market cap
INSTRUMENTS = ['BTC_USDT','ETH_USDT','XRP_USDT','DOGE_USDT','ADA_USDT','SOL_USDT','MATIC_USDT','LTC_USDT','DOT_USDT','SHIB_USDT','DAI_USDT']

public_url = "https://api.crypto.com/exchange/v1/public"

def get_instrument_candlestick_by_day(instrument_name = "BTC_USDT", num_days = 1, timestep = "5m", end_time = None):
    # if end_ts not specified, back from nearest day
    if end_time == None:
        end_time = int(time.time())*SEC_TO_MILLI
        end_time -= end_time%UNIX_DAY
    counts = {"1m":288, "5m":288, "15m":96, "30m":48, "1h":24}
    results = []
    print('REQUEST: start_ts', int(end_time - UNIX_DAY * num_days), 'end_ts', end_time, 'timestep', timestep)
    # max count request data size is 300
    for i in tqdm(range(num_days-1, -1, -1)):
        start_ts = int(end_time - UNIX_DAY * (i + 1))
        end_ts = int(end_time - UNIX_DAY * i)
        if timestep == "1m":
            for j in range(5):
                results.append(requests.get("/".join([public_url, "get-candlestick"]), params={"instrument_name":instrument_name, "count":counts[timestep], 
                                "timeframe":timestep, "start_ts":str(start_ts+int(UNIX_DAY/5*j)), "end_ts":str(start_ts+int(UNIX_DAY/5*(j+1)))}))
                if results[-1].status_code == 500:
                    print("ERROR:", results[-1].text())
                    return
                if VERBOSE:
                    print(results[-1].json()['result']['data'][0], results[-1].json()['result']['data'][-1], flush=True)
                    print(len(results[-1].json()['result']['data']),flush=True)
        else:
            results.append(requests.get("/".join([public_url, "get-candlestick"]), params={"instrument_name":instrument_name, "count":counts[timestep], 
                            "timeframe":timestep, "start_ts":str(start_ts), "end_ts":str(end_ts)}))
            if results[-1].status_code == 500:
                print("ERROR:", results[-1].text())
                return
            if VERBOSE:
                print(results[-1].json()['result']['data'][0], results[-1].json()['result']['data'][-1])
                print(len(results[-1].json()['result']['data']))
    return results, num_days, end_time

def get_instrument_candlestick_by_ts(instrument_name = "BTC_USDT", start_ts = 0, end_ts = UNIX_DAY, timestep = "5m"):
    unix_time = time.time()*SEC_TO_MILLI
    results = []
    # implement
    return results

# response: requests.Response list
def response_to_csv(response, dir='data', filename=None):
    if len(response) == 0:
        return
    if filename==None:
        filename = response[0].json()['result']['instrument_name']
    data = []
    for r in response:
        data.extend(r.json()['result']['data'])
    df = pd.DataFrame(data)
    if '.csv' not in filename:
        filename += '.csv'
    util.df_to_csv(df, filename)
    return filename


if __name__=="__main__":
    durations = [35]
    timesteps = ['1m', '5m', '30m']
    for instr in INSTRUMENTS:
        for d in durations:
            for t in timesteps:
                r, n, e = get_instrument_candlestick_by_day(instr, d, t)
                fname = '_'.join([instr, 'dur', str(n), 'end', str(e), 'ts', t])
                response_to_csv(r, filename=fname)