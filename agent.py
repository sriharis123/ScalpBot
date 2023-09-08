import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from environment import SimpleTradingEnvironment
from augmentation import CandlestickData
import datetime
import util
from tqdm import tqdm
import pickle
from operator import itemgetter

class Actor(nn.Module):  # q
    def __init__(self, input_dim=util.STATE_DIM, output_dim=util.ACTION_DIM, lr=0.0001):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=util.LR_LAMBDA)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action_probs = self.l3(x)
        return action_probs

class Critic(nn.Module):  # v
    def __init__(self, input_dim=util.STATE_DIM, output_dim=1, lr=0.0001):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=util.LR_LAMBDA)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        state_value = self.l3(x)
        return state_value

class Policy(nn.Module):
    def __init__(self, input_dim=util.STATE_DIM, output_dim=util.ACTION_DIM, lr=0.0001):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=util.LR_LAMBDA)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        policy = F.softmax(self.l3(x), dim=0)
        return policy

# experience replay buffer
class RingBuf:
    def __init__(self, size=6000):
        self.position=0
        self.size=size
        self.s=[None]*self.size
        self.a=[None]*self.size
        self.r=[None]*self.size
        self.sp=[None]*self.size
    
    def add(self, s, a, r, sp):
        self.s[self.position%self.size] = s
        self.a[self.position%self.size] = a
        self.r[self.position%self.size] = r
        self.sp[self.position%self.size] = sp
        self.position += 1
    
    def get(self):
        # idx = np.random.choice(range(self.position), min(self.position, amt), replace=False)
        idx = np.random.randint(0, min(self.position, self.size))
        return (self.s[idx], self.a[idx], self.r[idx], self.sp[idx])

class Agent:
    def __init__(self, epochs=200, lr=0.0000008, gamma=0.8, vis=False):
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.vis = vis

    def train(self):
        nnQ = Actor(lr=self.lr)
        nnV = Critic(lr=self.lr)
        nnP = Policy(lr=self.lr)

        assets = ['DOT_USDT_dur_597_end_1692576000000_ts_1m.csv']
        env = SimpleTradingEnvironment(asset_data=[CandlestickData(assetname) for assetname in assets])
        D = [RingBuf(9000), RingBuf(9000), RingBuf(9000)]

        for i in range(1,self.epochs+1):
            tau = []
            done = False
            next_state, candles = env.reset()
            state = torch.Tensor(np.array(next_state)).permute((2,1,0)).flatten()
            freq = np.array([0,0,0])
            while not done:
                action = int(torch.multinomial(nnP.forward(state), 1).int())
                next_state, reward, done = env.step(action)
                freq[action] += 1
                next_state = torch.Tensor(np.array(env.next_state)).permute((2,1,0)).flatten()
                D[action].add(state, action, reward, next_state)
                tau.append((state, action, reward, next_state))
                state = next_state

            cpl = torch.zeros((1))
            cql = torch.zeros((1))
            cvl = torch.zeros((1))
            if i % 5 == 0 and self.vis:
                util.visualize_go(candles, [t[1] for t in tau])

            # update policy
            for s,a,r,sp in tqdm(tau):
                # if a==0 and np.random.rand() > min(1, sum(freq[1:])/freq[0]):
                #     continue
                prob_a = nnP.forward(s)[a]
                advantage = nnQ.forward(s)[a] - nnV.forward(s) # rew + gamma*q(s,a) - v(s)
                loss = -(advantage * torch.log(prob_a)) / len(tau)
                cpl += loss

                nnP.optimizer.zero_grad()
                loss.backward()
                nnP.optimizer.step()
                # nnP.scheduler.step()

            # update q&v values
            # for s,a,r,sp in tqdm(tau):
            for j in tqdm(range(len(tau))):
                s, a, r, sp = D[j%3].get()
                # if a==0 and np.random.rand() > min(1, sum(freq[1:])/freq[0]):
                #     continue
                utilityQ = r + self.gamma * nnQ.forward(sp).max()
                qs = nnQ.forward(s)
                targetQ = qs.clone()
                targetQ[a] = utilityQ
                loss = nnQ.loss(qs, targetQ) / len(tau)
                cql += loss

                nnQ.optimizer.zero_grad()
                loss.backward()
                nnQ.optimizer.step()
                # nnQ.scheduler.step()

                targetV = r + self.gamma * nnV.forward(sp)
                vs = nnV.forward(s)
                loss = nnV.loss(vs, targetV) / len(tau)
                cvl += loss

                nnV.optimizer.zero_grad()
                loss.backward()
                nnV.optimizer.step()
                # nnV.scheduler.step()
                
            if i % 1 == 0:
                print(f'============= EPOCH {i} =============')
                print(f'act freq:  {freq}')
                print(f'cum rew:   {env.cum_rew}')
                print(f'total:     {env.total_capital}')
                print(f'usdt:      {env.usdt_capital}')
                print(f'asset:     {env.asset_amount_held}')
                print(f'cpl:       {str(cpl)}')
                print(f'cql:       {str(cql)}')
                print(f'cvl:       {str(cvl)}')
                print()
            
        torch.save(nnP.state_dict(), f'models/P_buf.pt')
        torch.save(nnQ.state_dict(), f'models/Q_buf.pt')
        torch.save(nnV.state_dict(), f'models/V_buf.pt')

def play(verbose=True, visualize=True):
    nnP = Policy()
    nnP.load_state_dict(torch.load("models/P.pt"))
    assets = ['DOT_USDT_dur_597_end_1692576000000_ts_1m.csv']
    env = SimpleTradingEnvironment(asset_data=[CandlestickData(assetname) for assetname in assets])
    
    print(f"====== REPLAY ======")
    print(f'asset_usdt:   {env.price}')
    
    actions = []
    done = False
    next_state, candles = env.reset()
    state = torch.Tensor(np.array(next_state)).permute((2,1,0)).flatten()
    freq = np.array([0,0,0])
    while not done:
        action = int(torch.multinomial(nnP.forward(state), 1).int())
        next_state, reward, done = env.step(action)
        freq[action] += 1
        next_state = torch.Tensor(np.array(env.next_state)).permute((2,1,0)).flatten()
        actions.append(action)
        state = next_state

        if verbose and action==1:
            print(f"======BUY {freq[action]}======")
            print(f'asset_usdt:   {env.price}')
            print(f'state:   {state}')
            print(f'reward:  {reward}')
            print(f'cum rew: {env.cum_rew}')
            print(f'total:   {env.total_capital}')
            print(f'usdt:    {env.usdt_capital}')
            print(f'asset:   {env.asset_amount_held}')
        elif verbose and action==2:
            print(f"======SELL {freq[action]}======")
            print(f'asset_usdt:   {env.price}')
            print(f'state:   {state}')
            print(f'reward:  {reward}')
            print(f'cum rew: {env.cum_rew}')
            print(f'total:   {env.total_capital}')
            print(f'usdt:    {env.usdt_capital}')
            print(f'asset:   {env.asset_amount_held}')
        if verbose and action:
            input("press enter to continue")
    print(f'final total:  {env.total_capital}')
    print(f'freq:         {freq}')
    if visualize:
        util.visualize_go(candles, [a for a in actions])

if __name__=='__main__':
    Agent(epochs=300, vis=True).train()
    # play(False, True)

