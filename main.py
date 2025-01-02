import torch
import time
import numpy as np
import csv
import os

from DataGen import MakeVectors
from DataProcessing import DataProcessing
from CharacterGenerator import CharGenModel

def train(x, y, device):
  epochs = 250000
  model = CharGenModel(x, y, device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  loss_fn = torch.nn.MSELoss()
  
  start_time = time.time()  
  
  for epoch in range(epochs):
    
    optimizer.zero_grad()
    y_pred = model.forward(model.x_train)
    train_loss = loss_fn(y_pred, model.y_train)
    train_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
      y_test_pred = model.forward(model.x_test)
      test_loss = loss_fn(y_test_pred, model.y_test)
    
    if epoch % 1000 == 0:
      elapsed_time = time.time() - start_time
      print(f"Epoch {epoch}: Train loss {train_loss}, Test loss {test_loss}, Epoch/Sec {epoch / elapsed_time}")
      
  torch.save(model.state_dict(), "model.pth")
  
def generate_character(model, char, device):
  model.eval()
  
  x = np.zeros(shape=(1, 4, 320))
  x[0][0][0] = ord(char)
  out = model.predict(x)

  os.makedirs("results", exist_ok=True)  
  DataProcessing.draw_vector(DataProcessing.convert_vectors_array(out), "results/test.png")
  
  for i in range(len(out)):
    for j in range(len(out[0])):
      out[i][j] = int(out[i][j])
      
  print(out)
  
if __name__ == "__main__":
  # MakeVectors(10, 20)
  DataProcessing.gen_training_data("training/vectors.csv")
  x, y = DataProcessing.gen_data_vectors(308)
  
  device = torch.device("mps")
  train(x, y, device)
  
  model = CharGenModel(x, y, device).to(device)
  model.load_state_dict(torch.load("model.pth", weights_only=False))
  generate_character(model, "B", device)