import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Define una red neuronal simple
class SimpleNet0(nn.Module):
    def __init__(self, input_size,  num_classes):
        super(SimpleNet0, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        return out
# Define una red neuronal simple
class SimpleNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
        
class SimpleNet2(nn.Module):
    def __init__(self, input_size, h1,h2, num_classes):
        super(SimpleNet2, self).__init__()
        
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2  = nn.Linear(h1, h2)
        self.fc3  = nn.Linear(h2, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

#################
class MNISTdataset(Dataset):
      
      def __init__(self,X,y,transform):
            
          self.X = [transform(Image.fromarray(np.uint8(x))) for x in X]
          self.y = y
      def __len__(self):
          return len(self.X)
      def __getitem__(self,idx):
          return [self.X[idx].ravel(),self.y[idx]]