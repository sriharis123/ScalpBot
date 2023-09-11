import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
import augmentation
from plotly import graph_objects as go
import dash
import pickle
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'device: {device}')

class Attention(nn.Module):
    def __init__(self, n_dims=32):
        super(Attention, self).__init__()
        self.weights = nn.Linear(n_dims, n_dims)
        self.act = nn.Tanh()
    
    def forward(self, x):
        return self.act(self.weights(x))

class FutureLR(nn.Module):
    def __init__(self, n_dims=32, lr=0.01):
        super(FutureLR, self).__init__()
        # attention layer
        self.attention = Attention(n_dims)
        self.fc1 = nn.Linear(n_dims, 128)
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 1)
        self.drop = nn.Dropout(p=0.8)
        self.act = nn.Tanh()
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        # x = self.attention(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x

class TSDataset(Dataset):
    def __init__(self, X, Y):
        self.data = torch.FloatTensor(X).to(device)
        self.targets = torch.FloatTensor(Y).to(device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def collect_data(files = [], output_size = 10, sublen=32):
    candles = []
    for f in files:
        c = augmentation.CandlestickData(fname=f)
        c.add_EMA(20)
        c.add_future_lr(cols=['ema_20_c'], periods=np.arange(200))
        c.dropna()
        candles.append(c)
    X = np.empty((output_size, sublen))
    Y = np.empty((output_size, 1))
    for i in range(output_size):
        data = np.random.randint(len(files))
        idx = np.random.randint(len(candles[data].df['ema_20_c']) - sublen)
        X[i] = minmax_scale(candles[data].df['ema_20_c'].iloc[idx:idx+sublen].to_numpy())
        Y[i] = candles[data].df['flr_ema_20_c_200'].iloc[idx+sublen]
    return X, Y

def plotloss(lf='', losses=[]):
    names = ['train', 'test']
    if len(losses)==0:
        with open(lf, 'rb') as l:
            losses = pickle.load(l)
    fig = go.Figure(data=[go.Scatter(y=loss,name=names[i]) for i,loss in enumerate(losses)])
    fig.show()


def train(data, epochs, n_dims, folds=5, lr=0.001):
    kfold = KFold(n_splits=folds, shuffle=True)
    results = []

    for f, (train_idx, test_idx) in enumerate(kfold.split(data)):
        train = torch.utils.data.DataLoader(
            data, batch_size=8, sampler=SubsetRandomSampler(train_idx))
        test = torch.utils.data.DataLoader(
            data, batch_size=8, sampler=SubsetRandomSampler(test_idx))
        nn = FutureLR(n_dims=n_dims, lr=lr).to(device)

        losses = [[],[]]

        print(f'=============== FOLD {f} ===============')

        for e in range(1,epochs+1):
            trainloss = torch.Tensor([0]).to(device)
            testloss = torch.Tensor([0]).to(device)
            # training
            for i, (x, y) in enumerate(train):
                nn.optimizer.zero_grad()
                yhat = nn(x)
                loss = nn.loss(yhat, y)
                loss.backward()
                nn.optimizer.step()
                trainloss += loss
            # validation
            with torch.no_grad():
                for i, (x, y) in enumerate(test):
                    yhat = nn(x)
                    testloss += nn.loss(yhat, y)
            # print
            if not e % 25:
                print(f'EPOCH {e}:\t Training Loss: {trainloss/len(train)}; Testing Loss: {testloss/len(test)}')
                torch.save(nn.state_dict(), f'models\predict\inter\predict_fold_{f}_{e}.pt')
            # append
            losses[0].append(trainloss.item()/len(train))
            losses[1].append(testloss.item()/len(test))
        # save
        torch.save(nn.state_dict(), f'models\predict\predict_fold_{f}.pt')
        with open(f'models\predict\losses_{f}.txt', 'wb') as lf:
            pickle.dump(losses, lf)
        # plot
        plotloss(losses=losses)
        with torch.no_grad():
            testloss = torch.Tensor([0]).to(device)
            for i, (x, y) in enumerate(test):
                yhat = nn(x)
                testloss += nn.loss(yhat, y)
            results.append(testloss.item()/len(test))
            print(f'Loss for fold {f}: {testloss.item()/len(test)}')
    return results
                    
def test(data, model):
    for i, (x, y) in enumerate(data):
        print(model(x), y)
    
if __name__=="__main__":
    X, Y = collect_data(['DOGE_USDT_dur_597_end_1694044800000_ts_1m.csv', 
        'DOGE_USDT_dur_597_end_1694044800000_ts_1m.csv', 
        'SHIB_USDT_dur_597_end_1694131200000_ts_1m.csv'], 32000, 64)
    # X, Y = collect_data(['DOGE_USDT_dur_35_end_1691625600000_ts_1m.csv'], 8, 64)
    data = TSDataset(X, Y)
    print(train(data, 200, 64))

    # nn = FutureLR(64)
    # nn.load_state_dict(torch.load('models/predict/predict_fold_1.pt'))
    # X, Y = collect_data(['DOGE_USDT_dur_35_end_1691625600000_ts_1m.csv'], 8, 64)
    # data = TSDataset(X, Y)
    # test(data, nn)
    
    # plotloss(lf='models\predict\save\losses_3.txt')