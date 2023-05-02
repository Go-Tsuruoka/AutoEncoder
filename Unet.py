import os
from pickletools import optimize
from turtle import forward, showturtle
import numpy as np
from pkg_resources import load_entry_point
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


#img = Image.opne("dataset")
#img_resize = img.resize((28,28))
#img.show()

class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        #self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform = target_transform
    def __len__(self):
        #return len(self.img_labels)
        return 0
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir)
        image = read_image(img_path)
        """
        label=self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label=self.target_transform(label)
        
        """
        return image







cuda = torch.cuda.is_available()#gpaが利用可能かどうか

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


epochs=100
batch_size=128
learning_late=1e-3

out_dir = "./autoencoder"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(512*512,28*28),
            nn.ReLU(),
            nn.Linear(28*28,128),#in 784,out 512 bias =T
            nn.ReLU(),
            #nn.Linear(512,128),
            #nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,2),
            
        )
        self.decoder=nn.Sequential(
            nn.Linear(2,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.ReLU(),
            nn.Linear(28*28,512*512),
            nn.Tanh()

        )
    def forward(self,x):
        y=self.encoder(x)
        z=self.decoder(y)+x
        return z
model=Autoencoder()
if cuda:
    model.cuda()

#model=Autoencoder().to(device)
print(model)

#def tensor_to_img(x):




loss_fn=nn.MSELoss()#二乗誤差を基準に
#optimizer=torch.optim.SGD(model.parameters(),lr=learning_late)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_late)
losses=[]

"""
def train_loop(dataloader,model,loss_fn,optimizer):
    #size = len(dataloader.dataset)
    for batch,
"""
training_data = CustomImageDataset()
test_data =CustomImageDataset()
train_dataloader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)
for epoch in range(epochs):
    print(f"Epoch{epoch+1}\n----------------------")
    for data in train_dataloader:
        img=data
        x=img.view(img.size(0),-1)
        if cuda:
            x=Variable(x).cuda()
        else:
            x=Variable(x)
        output =model(x)
        loss=loss_fn(x,output)#誤差
        losses.appned(loss.data[0])
        optimizer.zero_grad()
        loss.backward()#逆伝搬
        optimizer.step()#パラメタ更新

    
plt.plot(losses)
plt.show()