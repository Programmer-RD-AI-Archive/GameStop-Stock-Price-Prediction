import torch,torchvision
from torch.nn import *
device = 'cuda'
class Model(Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.hidden = hidden
        self.lstm1 = LSTMCell(1,self.hidden).to(device)
        self.lstm2 = LSTMCell(self.hidden,self.hidden).to(device)
        self.linear = Linear(self.hidden,1).to(device)

    def forward(self,X,future=0):
        outputs = []
        batch_size = X.size(0)
        h_t1 = torch.zeros(batch_size,self.hidden,dtype=torch.float32)
        c_t1 = torch.zeros(batch_size,self.hidden,dtype=torch.float32)
        h_t2 = torch.zeros(batch_size,self.hidden,dtype=torch.float32)
        c_t2 = torch.zeros(batch_size,self.hidden,dtype=torch.float32)
        h_t1 = h_t1.to(device)
        c_t1 = c_t1.to(device)
        h_t2 = h_t2.to(device)
        c_t2 = c_t2.to(device)
        for X_batch in X.split(1,dim=1):
            X_batch = X_batch.to(device)
            h_t1,c_t1 = self.lstm1(X_batch,(h_t1,c_t1))
            h_t1 = h_t1.to(device)
            c_t1 = c_t1.to(device)
            h_t2,c_t2 = self.lstm2(h_t1,(h_t2,c_t2))
            h_t2 = h_t2.to(device)
            c_t2 = c_t2.to(device)
            pred = self.linear(h_t2)
            outputs.append(pred)
        for i in range(future):
            h_t1,c_t1 = self.lstm1(X_batch,(h_t1,c_t1))
            h_t1 = h_t1.to(device)
            c_t1 = c_t1.to(device)
            h_t2,c_t2 = self.lstm2(h_t1,(h_t2,c_t2))
            h_t2 = h_t2.to(device)
            c_t2 = c_t2.to(device)
            pred = self.linear(h_t2)
            outputs.append(pred)
        outputs = torch.cat(outputs,dim=1)
        return outputs
