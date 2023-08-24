# Trading environment inspired by Harsha Andey https://medium.com/coinmonks/deep-reinforcement-learning-for-trading-cryptocurrencies-5b5502b1ece1

import numpy as np
import pandas as pd
import util
from sklearn import preprocessing
from augmentation import CandlestickData

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
    def __init__(self, asset_data: list, hours_per_rollout=100, ts='1m',
                 window=util.WINDOW, usdt_capital=util.CAPITAL, usdt_pbuy=util.PMAXBUY, store_flag=1,
                 running_thresh=0.1, usdt_capital_threshold=0.3):

        if type(asset_data) != list or len(asset_data) == 0 or type(asset_data[0]) != CandlestickData:
            raise AttributeError('asset_data expected to be a list of CandlestickData')

        # the initial capital to start every simulation
        self.initial_capital = usdt_capital
        # the current amount of usdt maintained
        self.usdt_capital = usdt_capital
        # the current value of all assets maintained
        self.total_capital = usdt_capital
        # fraction of capital to use for buy
        self.usdt_pbuy = usdt_pbuy
        # minimum fraction of initial capital allowable
        self.running_thresh = running_thresh
        # minimum fraction of initial capital left to execute a trade
        self.usdt_capital_threshold = usdt_capital_threshold

        for a in asset_data:
            a.add_RSI()                                                      # measure of volatility
            a.add_log_return(cols=['c'], period=1)                           # log return of the close
            a.add_log_return(cols=['c'], period=5)                           # log return of the close. period 5
            a.add_BB(mov=False, hb=False, lb=False, pb=True)                 # percentB of the bollinger band on close
            a.add_BB(ind='c_lr_1', mov=False, hb=False, lb=False, pb=True)   # bollinger bands surrounding log returns
            a.add_MACD(mom=False, sig=False)                                 # macd indicator. only shows diff
            a.dropna()

        self.asset_data = asset_data

        # asset_data[#assets][N][D]

        self.asset_num = 0
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
        self.cum_rew = 0
        self.price = 0

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

        self.asset_num = np.random.randint(0, len(self.asset_data))
        self.current_asset = self.asset_data[self.asset_num]
        self.timestep = np.random.randint(0,len(self.current_asset.df.index)-self.rollout_length)
        self.end = self.timestep + self.rollout_length

        self.state = None
        self.action = 0
        self.reward = 0
        self.next_state = self.get_state(self.timestep)
        self.cum_rew = 0
        self.price = self.current_asset.df['c'].iloc[self.timestep]
        # self.next_state = None
        # self.next_action = 0
        
        self.done = False

        if self.store_flag == 1:
            self.store = {"action_store": [],
                          "reward_store": [],
                          "running_capital": [],
                          "port_ret": []}

        return self.next_state

    def step(self, action):
        self.price = self.current_asset.df['c'].iloc[self.timestep]

        self.state = self.next_state
        self.action = action

        self.compute_reward()

        self.timestep += 1
        self.next_state = self.get_state(self.timestep)

        self.done = self.check_terminal()

        if self.done:
            reward_offset = 0
            # ret = (self.store['running_capital'][-1] /
            #        self.store['running_capital'][-0]) - 1
            if self.timestep < self.terminal_idx: # ran out of money :(
                reward_offset += -1 * \
                    max(0.5, 1 - self.timestep/self.terminal_idx)
            # if self.store_flag:
            #     reward_offset += 10 * ret
            self.reward += reward_offset

        if self.store_flag:
            self.store["action_store"].append(self.action)
            self.store["reward_store"].append(self.reward)
            self.store["running_capital"].append(self.usdt_capital)
            info = self.store
        else:
            info = None

        return self.next_state, self.reward, self.done

    def compute_reward(self):
        # reward calculation
        self.reward = 0.0
        if self.action == 1:
            if self.usdt_capital > self.initial_capital * self.usdt_capital_threshold: # if agent has more than 30% money left
                investment = self.usdt_capital * self.usdt_pbuy
                self.usdt_capital -= investment
                self.asset_amount_held += investment*(1-util.TAKERFEE)/self.price # agent gains assets equal to investment usdt
                self.reward = 0.0 # abs(self.action) * state[-2]
            else:
                self.reward = -0.1
        elif self.action == -1:
            if self.asset_amount_held > 0:
                self.usdt_capital += self.asset_amount_held*(1-util.TAKERFEE)*self.price
                self.asset_amount_held = 0
                self.reward = (np.log(self.total_capital)-np.log(self.initial_capital))
            else:
                reward = -0.1
        if self.reward < 0:
            self.reward *= 2 # double negative rewards
        self.cum_rew += self.reward

    def check_terminal(self):
        if self.timestep == self.end:
            return True
        elif self.usdt_capital <= self.initial_capital * self.running_thresh:
            return True
        else:
            return False

    def get_state(self, idx):

        self.total_capital = self.usdt_capital + self.asset_amount_held * self.price # capital + asset_held * closing_price

        state = np.concatenate([self.current_asset.df.iloc[idx][6:], np.array([ \
                np.log(self.total_capital)-np.log(self.initial_capital), \
                np.log(self.usdt_capital)-np.log(self.initial_capital) # , self.total_capital, self.usdt_capital, self.asset_amount_held, self.price
                ])])[:,np.newaxis]

        return state


def testenv():
    s = SimpleTradingEnvironment(asset_data=[CandlestickData('DOT_USDT_dur_35_end_1691625600000_ts_1m.csv')])
    s.reset()
    print(s.step(1))
    print(s.total_capital)
    print(s.price)
    print(s.step(0))
    for i in range(4000):
        s.step(0)
    print(s.step(0))
    print(s.price)
    print(s.step(-1))
    print(s.total_capital)
    print(s.next_state)
    print(s.cum_rew)

testenv()