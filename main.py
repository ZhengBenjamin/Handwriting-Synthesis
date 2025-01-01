import torch
import numpy as np
import csv

from DataGen import MakeVectors
from DataProcessing import DataProcessing
from CharacterGenerator import CharGenModel

def train(x, y, device):
  epochs = 20000
  model = CharGenModel(x, y, device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  loss_fn = torch.nn.MSELoss()
  
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
      print(f"Epoch {epoch}: Train loss {train_loss}, Test loss {test_loss}")
      
  torch.save(model.state_dict(), "model.pth")
  
def generate_character(model, char, device):
  model.eval()
  
  x = np.zeros(shape=(1, 4, 160))
  out = model.predict(x)
  
  DataProcessing.draw_vector(DataProcessing.convert_vectors_array(out), "test.png")
  
  for i in range(len(out)):
    for j in range(len(out[0])):
      out[i][j] = int(out[i][j])
      
  print(out)
  
if __name__ == "__main__":
  # MakeVectors(10, 20)
  DataProcessing.gen_training_data("training/vectors.csv")
  x, y = DataProcessing.gen_data_vectors()
  print(y[0])
  
  device = torch.device("cpu")
  train(x, y, device)
  
  model = CharGenModel(x, y, device).to(device)
  model.load_state_dict(torch.load("model.pth", weights_only=False))
  generate_character(model, "A", device)