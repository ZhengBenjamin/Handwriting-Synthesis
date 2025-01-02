import os
import csv
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw

class DataProcessing: 
  
  def convert_vectors_array(data):
    """Converts output vectors to list of coordnates"""
    converted = []

    for i in range(len(data)):
      for j in range(0, len(data[i]), 2):
        converted.append([int(data[i][j]), int(data[i][j + 1])])
    
    print(converted)
    return converted
  
  def draw_vector(vector, output_file):
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)

    stroke_length = len(vector) // 4 # Max 4 strokes per character
    
    for i in range(4): # For each stroke
      stroke = vector[i * stroke_length : (i + 1) * stroke_length - 1]
      for j in range(len(stroke) - 1):
        if stroke == [0, 0]:
          continue
        draw.line([stroke[j][0], stroke[j][1], stroke[j + 1][0], stroke[j + 1][1]], fill="black", width=3)

    image.save(output_file)
  
  def gen_training_data(data): 
    output_folder = "training"
    input_data = "training/training_vectors.csv"
    output_data = "training/training_output.csv"
    
    os.makedirs(output_folder, exist_ok=True)
    
    with open(input_data, "w") as output:
      writer = csv.writer(output)
      
      with open(data, "r") as file:
        reader = csv.reader(file)
        for row in reader: # Each character
          writer.writerow([0, 1] + [0] * 1280)
          for i in range(1, 4): # Each stroke to write character
            writer.writerow([i, 1] + row[0:i * 320] + [0] * (1280 - i * 320))
    
    print("Training input data generated to {}".format(input_data))
    
    with open(output_data, "w") as output:
      writer = csv.writer(output)
      
      with open(data, "r") as file:
        reader = csv.reader(file)
        for row in reader:
          for i in range(4): 
            writer.writerow(row[i * 320 : (i + 1) * 320])
            
    print("Training output data generated to {}".format(output_data))
    
  def gen_data_vectors(batch_size=10):
    vectors = pd.read_csv("training/vectors.csv", header=None)
      
    input_vectors = np.zeros(shape=(batch_size, 4, 320))
    output_vectors = np.empty(shape=(batch_size, 4, 320))
    
    for i in range(0, len(vectors), batch_size):
      for j in range(batch_size):
        for k in range(4):
          for l in range(320):
            output_vectors[j][k][l] = vectors.iloc[i + j][k * 320 + l + 1]
        input_vectors[j][0][0] = vectors.iloc[i + j][0]
    
    return input_vectors, output_vectors