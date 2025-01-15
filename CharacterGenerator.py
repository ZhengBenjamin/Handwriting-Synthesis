import numpy as np
import random
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CharGenModel(nn.Module):
  
  def __init__(self, x, y, device):
    super(CharGenModel, self).__init__()
    
    self.device = device
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Flatten for scaling
    x_flat = x.reshape(-1, 280)
    y_flat = y.reshape(-1, 280)
    
    scaler_x.fit(x_flat)
    scaler_y.fit(y_flat)
    
    self.scaler_x = scaler_x
    self.scaler_y = scaler_y # Save for reverse scaling during prediction 
    
    # Transform back to org shape 
    x_scaled = scaler_x.transform(x_flat).reshape(x.shape)
    y_scaled = scaler_y.transform(y_flat).reshape(y.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # Convert to tensors
    self.x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    self.y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    self.x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    self.y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    self.rnn = nn.LSTM(
      input_size=70,
      hidden_size=512,
      num_layers=1,
      batch_first=True
    )
    
    self.fc = nn.Linear(512, 70)
    
  def forward(self, x):
    rnn_out, _ = self.rnn(x)
    return self.fc(rnn_out)
  
  def predict(self, x):
    with torch.no_grad():
      x = torch.tensor(x, dtype=torch.float32).to(self.device)
      y = self.forward(x)
      y = y.cpu().numpy()[0]
      
      noise = np.random.normal(loc=0.0, scale=0.2, size=y.shape)
      randomized_output = y + noise
      
      return randomized_output