import torch
import time
import numpy as np
import csv
import sys
import os

from DataGen import MakeVectors
from DataProcessing import DataProcessing
from CharacterGenerator import CharGenModel

def train(x, y, device, continue_training=False):
  epochs = 100000000
  learning_rate = 0.0005
  model = CharGenModel(x, y, device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Starting learning rate
  loss_fn = torch.nn.MSELoss()
  test_loss_check_frequency = 300
  start_epoch = 0
  
  start_time = time.time()  
  
  if continue_training:
    checkpoint = torch.load("model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
  
  for epoch in range(start_epoch, epochs):
    
    optimizer.zero_grad()
    y_pred = model.forward(model.x_train)
    train_loss = loss_fn(y_pred, model.y_train)
    train_loss.backward()
    optimizer.step()
    
    if epoch % test_loss_check_frequency == 0:
      with torch.no_grad():
        y_test_pred = model.forward(model.x_test)
        test_loss = loss_fn(y_test_pred, model.y_test)
      
      elapsed_time = time.time() - start_time
      print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Test loss {test_loss:.4f}, LR {learning_rate}, Epoch/Sec {epoch / elapsed_time:.2f}")
        
    if epoch % 10000 == 0:
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      }, "model.pth")
      
      print("Model saved at epoch {}".format(epoch))
      
      # intervals = [(48, 57) , (65, 90)] #, (97, 122)]
      # for interval in intervals:
      #   for i in range(interval[0], interval[1] + 1):
      #     generate_character(model, chr(i), device, epoch)
      
      generate_character(model, "s", device, epoch)
      DataProcessing.gen_output_images(model, "the quick brown fox jumps over the lazy dog")
      
  
def generate_character(model, char, device, epochs):
  model.eval()
  
  x = np.zeros(shape=(1, 4, 70))
  x[0][0][0] = ord(char)
  out = model.predict(x)

  os.makedirs("progress", exist_ok=True)  
  DataProcessing.draw_vector(DataProcessing.convert_vectors_array(out), f"progress/{char}{epochs}.png")
  
if __name__ == "__main__":
  # MakeVectors(10, 20)
  # DataProcessing.gen_training_data("straining/vectors.csv")
  x, y = DataProcessing.gen_data_vectors()
  
  device = torch.device("mps")
  # train(x, y, device)
  # train(x, y, device, continue_training=True)
  
  model = CharGenModel(x, y, device).to(device)
  checkpoint = torch.load("model.pth", weights_only=False)
  model.load_state_dict(checkpoint["model_state_dict"])
  
  intervals = [(48, 57), (97, 122)]
  
  DataProcessing.gen_output_images(model, "The quick brown fox jumps over the lazy dog")