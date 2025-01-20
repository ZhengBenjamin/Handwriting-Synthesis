import torch
import numpy as np
import random
import os

from DataGen import MakeVectors
from DataProcessing import DataProcessing
from CharacterGenerator import CharGenModel
import time

def train(x, y, device, continue_training=False):
  """ Trains the model using generated data, and periodically saves the model """
  epochs = 100000000
  learning_rate = 0.000003
  model = CharGenModel(x, y, device).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = torch.nn.MSELoss()
  test_loss_check_frequency = 100
  start_epoch = 0
  scaler = torch.amp.GradScaler('cuda')
  
  if continue_training:
    checkpoint = torch.load("model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
  
  start_time = time.time()
  
  for epoch in range(start_epoch, epochs):
    
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
      y_pred = model.forward(model.x_train)
      train_loss = loss_fn(y_pred, model.y_train)
    
    scaler.scale(train_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if epoch % test_loss_check_frequency == 0:
      with torch.no_grad():
        with torch.amp.autocast('cuda'):
          y_test_pred = model.forward(model.x_test)
          test_loss = loss_fn(y_test_pred, model.y_test)
      
      elapsed_time = time.time() - start_time
      epochs_per_sec = (epoch - start_epoch + 1) / elapsed_time
      print(f"Epoch {epoch}: Train loss {train_loss:.4f}, Test loss {test_loss:.4f}, LR {learning_rate}, Epochs/sec {epochs_per_sec:.2f}")
        
    if epoch % 10000 == 0:
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      }, "model.pth")
      
      print("Model saved at epoch {}".format(epoch))
      
def generate_character(model, char, epochs):
  """ Runs the model using input character and generates an image """
  
  model.eval() 
  
  x = np.zeros(shape=(1, 4, 70))
  x[0][0][0] = ord(char)
  out = model.predict(x)

  os.makedirs("progress", exist_ok=True)  
  DataProcessing.draw_vector(DataProcessing.convert_vectors_array(out), f"progress/{char}{epochs}.png")

def generate_string(model, string):
  """ Runs the model using input string and generates an image """
  lines = []
  line = []

  for word in string.split():
    if len(line) < 20:
      line.append(word)
    else:
      lines.append(" ".join(line))
      line = [word]

  lines.append(" ".join(line))

  DataProcessing.gen_output_images(model, string, 0)

""" Entry point for applicaiton """

# Data generation:
# MakeVectors(35)
x, y = DataProcessing.gen_data_vectors()

# Device selection: 
device = torch.device("cuda") # Change CPU/CUDA/MPS/ROCm depending on your hardware

# Train model from scratch:
train(x, y, device) 

# Continue training from last checkpoint, (uncomment previous line):
# train(x, y, device, continue_training=True)

# Infrence: 
model = CharGenModel(x, y, device).to(device)
checkpoint = torch.load("model.pth", weights_only=False) # Pretrained model only contains lowercase letters
model.load_state_dict(checkpoint["model_state_dict"])

# Input characters/sentence: 
string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" 
generate_string(model, string)