# Trading environment inspired by Harsha Andey https://medium.com/coinmonks/deep-reinforcement-learning-for-trading-cryptocurrencies-5b5502b1ece1

import numpy as np
import pandas as pd
import util
from sklearn import preprocessing
from augmentation import CandlestickData
from collections import Iterable

class SimpleTradingEnvironment:
    """
    Trading Environment for trading a single asset, one buy/sell at a time.
    The Agent interacts with the environment class through the step() function.
    State Space: {o, h, l, c, v%, rsi, [macd], [bb]}[i-window+1:i+1]
    Action Space: {-1: Sell, 0: Do Nothing, 1: Buy}
    """

    """
    asset_data: list of CandlestickData. if len(asset_data) > 1, highly recommend normalize=True.
    usdt_capital: existing capital, in dollars
    usdt_pbuy: fraction of capital to use for buy
    store_flag: store SARSA info
    """
    def __init__(self, asset_data: Iterable[CandlestickData], normalize=True, hours_per_rollout=100, ts='1m',
                 window=util.WINDOW, usdt_capital=util.CAPITAL, usdt_pbuy=util.PMAXBUY, store_flag=1,
                 running_thresh=0.1, usdt_capital_threshold=0.3):

        if type(asset_data) != list or len(asset_data) == 0 or type(asset_data[0]) != pd.Series:
            raise AttributeError('asset_data expected to be a list of pd.Series')

        self.initial_capital = usdt_capital
        self.usdt_capital = usdt_capital
        self.usdt_pbuy = usdt_pbuy
        self.running_thresh = running_thresh
        self.usdt_capital_threshold = usdt_capital_threshold
        self.asset_data = []

        for a in asset_data:
            a.dropna(inplace=True)
            a.set_pct_change(cols=['v'])
            a.remove(cols=['t'])
            self.asset_data.append(a.to_numpy())

        # asset_data[#assets][N][D]

        # if normalize:
        #     self.asset_data = [preprocessing.StandardScaler().fit_transform(X)]
        # self.scaler = self.asset_data.scaler

        self.current_asset = None
        self.asset_amount_held = 0

        self.timestep = 0
        self.window = window
        self.rollout_length = util.UNIX_HOUR/util.SEC_TO_MILLI*hours_per_rollout/util.SEC_PER_IDX[ts]
        self.end = 0

        self.state = None
        self.action = 0
        self.reward = 0
        self.next_state = None
        # self.next_action = 0

        self.done = False

        self.store_flag = store_flag
        if self.store_flag == 1:
            self.store = {"action_store": [],
                          "reward_store": [],
                          "running_capital": [],
                          "port_ret": []}

    def reset(self):
        self.usdt_capital = self.initial_capital
        self.asset_amount_held = 0

        self.current_asset = self.asset_data[np.random.randint(0, self.asset_data.shape()[0])]
        self.timestep = np.random.randint(0, self.asset_data[0])
        self.end = self.timestep + self.rollout_length

        self.state = self.get_state(self.timestep)
        self.action = 0
        self.reward = 0
        self.next_state = None
        # self.next_action = 0
        
        self.done = False

        if self.store_flag == 1:
            self.store = {"action_store": [],
                          "reward_store": [],
                          "running_capital": [],
                          "port_ret": []}

        return self.state

    def step(self, action):
        self.current_act = action
        self.current_price = self.asset_data.frame.iloc[self.timestep,:]['Adj Close']
        self.current_reward = self.calculate_reward()
        self.prev_act = self.current_act
        self.timestep += 1
        self.next_return, self.current_state = self.get_state(self.timestep)
        self.done = self.check_terminal()

        if self.done:
            reward_offset = 0
            ret = (self.store['running_capital'][-1] /
                   self.store['running_capital'][-0]) - 1
            if self.timestep < self.terminal_idx:
                reward_offset += -1 * \
                    max(0.5, 1 - self.timestep/self.terminal_idx)
            if self.store_flag:
                reward_offset += 10 * ret
            self.current_reward += reward_offset

        if self.store_flag:
            self.store["action_store"].append(self.current_act)
            self.store["reward_store"].append(self.current_reward)
            self.store["running_capital"].append(self.capital)
            info = self.store
        else:
            info = None

        return self.current_state, self.current_reward, self.done, info

    def calculate_reward(self):
        investment = self.running_capital * self.capital_frac
        reward_offset = 0

        # Buy Action
        if self.current_act == 1:
            if self.running_capital > self.initial_cap * self.running_thresh:
                self.running_capital -= investment
                asset_units = investment/self.current_price
                self.asset_amount_held += asset_units
                self.current_price *= (1 - self.trans_cost)

        # Sell Action
        elif self.current_act == -1:
            if self.asset_amount_held > 0:
                self.running_capital += self.asset_amount_held * \
                    self.current_price * (1 - self.trans_cost)
                self.asset_amount_held = 0

        # Do Nothing
        elif self.current_act == 0:
            if self.prev_act == 0:
                reward_offset += -0.1
            pass

        # Reward to give
        prev_cap = self.capital
        self.capital = self.running_capital + \
            (self.asset_amount_held) * self.current_price
        reward = 100*(self.next_return) * self.current_act - \
            np.abs(self.current_act - self.prev_act) * self.trans_cost
        if self.store_flag == 1:
            self.store['port_ret'].append((self.capital - prev_cap)/prev_cap)

        if reward < 0:
            # To make the Agent more risk averse towards negative returns.
            reward *= NEG_MUL
        reward += reward_offset

        return reward

    def check_terminal(self):
        if self.timestep == self.terminal_idx:
            return True
        elif self.capital <= self.initial_cap * self.usdt_capital_threshold:
            return True
        else:
            return False

    def get_state(self, idx):
        # state = self.asset_data[idx][1:]
        # state = self.scaler.transform(state.reshape(1, -1))

        ## [state[close, macd, rsi, bb], total return %, existing capital %, unrealized return %, previous action]
        # state = np.concatenate([state, [[self.capital/self.initial_cap,
        #                                  self.running_capital/self.capital,
        #                                  self.asset_amount_held * self.current_price/self.initial_cap,
        #                                  self.prev_act]]], axis=-1)

        # next_ret = self.asset_data[idx][0]
        return state
