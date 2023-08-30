# %%
import numpy as np
import torch as t
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# %%
x,y = t.load('C:\\Users\\hp\\Desktop\\Mans1611\\University\\deep learning\\Project\\MNIST\\processed\\training.pt')
y

# %%
labels = F.one_hot(y)
labels.shape

# %%
class CTDataset(Dataset):
    def __init__(self,path):
        self.x , self.y = t.load(path)
        self.x = self.x / 256
        self.y = F.one_hot(self.y,num_classes=10).to(float)
    def __len__(self):
        return self.x.shape[0]  # return length.
    def __getitem__(self,index):
        return self.x[index], self.y[index]        

# %%
train_dataset = CTDataset('C:\\Users\\hp\\Desktop\\Mans1611\\University\\deep learning\\Project\\MNIST\\processed\\training.pt')
test_dataset = CTDataset('C:\\Users\\hp\\Desktop\\Mans1611\\University\\deep learning\\Project\\MNIST\\processed\\test.pt')

# %%
train_dl = DataLoader(train_dataset,batch_size = 5)

# %%
Loss = nn.CrossEntropyLoss()

# %%
class NeuralNetwork(nn.Module):
    def __init__(self): ## constructor 
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.relu = nn.LeakyReLU() 
    def forward (self,x):
        x = x.view(-1,28**2)  ## flatten the dataset 
        x = self.relu(self.Matrix1(x))
        x = self.relu(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    

# %%
f = NeuralNetwork()

# %%
def train_model (dl,f,n_epoches=20):
    opt = SGD(f.parameters(),lr=0.01)
    opt2 = Adam(f.parameters(),lr=0.01)
    loss = nn.CrossEntropyLoss()
    epoches = []
    losses = []
    
    for epoch in range(n_epoches):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x,y) in enumerate(dl):
            opt.zero_grad()
            loss_value = loss(f(x),y)
            loss_value.backward()
            opt.step()
            epoches.append(epoch+1/N)
            losses.append(loss_value.item())
    return np.array(epoches),np.array(losses)

# %%
epoch_data,loss_data = train_model(train_dl,f)

# %%
epoch_data_avg = epoch_data.reshape(20,-1).mean(axis=1)
loss_data_avg = loss_data.reshape(20,-1).mean(axis=1)
plt.plot(epoch_data_avg,loss_data_avg,'--x')
plt.ylabel('loss function')
plt.xlabel('time')

# %%
train_dataset[0][1]
x_sample = train_dataset[0][0]
y_hat = f(x_sample)
t.argmax(y_hat)

# %%
xs,ys = train_dataset[0:2000]
y_hat = f(xs).argmax(axis=1)

# %%
Loss(f(xs),ys)

# %%
fig,ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'The Predicted {y_hat[i]}')
fig.tight_layout()
plt.show()

# %%
xt,yt = test_dataset[:2000]
test_hat = f(xt).argmax(axis=1)


# %%
fig,ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xt[i])
    plt.title(f'The Predicted {test_hat[i]}')
fig.tight_layout()
plt.show()


