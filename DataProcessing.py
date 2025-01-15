import os
import csv
import random
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
    
    return converted
  
  def draw_vector(vector, output_file):
    """ Draws a vector to an image """
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
    """ Deprecated: Generates training data from vectors """
    output_folder = "training"
    input_data = "training/training_vectors.csv"
    output_data = "training/training_output.csv"
    
    os.makedirs(output_folder, exist_ok=True)
    
    with open(input_data, "w") as output:
      writer = csv.writer(output)
      
      with open(data, "r") as file:
        reader = csv.reader(file)
        for row in reader: # Each character
          writer.writerow([0, 1] + [0] * 320)
          for i in range(1, 4): # Each stroke to write character
            writer.writerow([i, 1] + row[0:i * 70] + [0] * (320 - i * 70))
    
    print("Training input data generated to {}".format(input_data))
    
    with open(output_data, "w") as output:
      writer = csv.writer(output)
      
      with open(data, "r") as file:
        reader = csv.reader(file)
        for row in reader:
          for i in range(4): 
            writer.writerow(row[i * 70 : (i + 1) * 70])
            
    print("Training output data generated to {}".format(output_data))
    
  def gen_data_vectors():
    """ Updated generation of training matricies from vectors csv """
    vectors = pd.read_csv("training/vectors.csv", header=None)
    batch_size = len(vectors)
      
    input_vectors = np.zeros(shape=(batch_size, 4, 70))
    output_vectors = np.empty(shape=(batch_size, 4, 70))
    
    for i in range(0, len(vectors), batch_size):
      for j in range(batch_size):
        for k in range(4):
          for l in range(70):
            output_vectors[j][k][l] = vectors.iloc[i + j][k * 70 + l + 1]
        input_vectors[j][0][0] = vectors.iloc[i + j][0]
    
    return input_vectors, output_vectors
  
  def gen_output_images(model, input: str):
    """ Runs model using input string and generates an image """
    words = []
    input = input + " "
    
    letters = []
    for letter in input.lower():
      if ord(letter) >= 97 and ord(letter) <= 122:
        x = np.zeros(shape=(1, 4, 70))
        x[0][0][0] = ord(letter)
        for i in range(random.randint(1, 4)):
          x[0][random.randint(0, 3)][random.randint(0, 69)] = random.randint(-1, 1)
        out = model.predict(x)
        letters.append(DataProcessing.convert_vectors_array(out))
      elif ord(letter) == 32:
        words.append(letters)
        letters = []
    
    offset = 0
    for word in words:
      for letter in word:
        start = min([coord[0] for coord in letter if coord[0] > 20])
        end = max([coord[0] for coord in letter if coord[0] < 240])
        
        for coord in letter:
          coord[0] += offset
        
        offset += max(end - start + random.randint(-5, 10), 30)
        
      offset += random.randint(50, 100)
      
    # Convert to strokes
    strokes = []
    
    for word in words:
      for letter in word:
        stroke_length = len(letter) // 4
        for i in range(4):
          stroke = letter[i * stroke_length : (i + 1) * stroke_length - 1]
          strokes.append(stroke)
          
    # Draw strokes
    img_width = max([x[0] for stroke in strokes for x in stroke]) + 100
    img_height = 256
    
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)
    
    for stroke in strokes:
      for j in range(len(stroke) - 1):
        if stroke == [0, 0]:
          continue
        if stroke[j][1] < 10:
          continue
        draw.line([stroke[j][0], stroke[j][1], stroke[j + 1][0], stroke[j + 1][1]], fill="black", width=4)
        
    image.save("output.png")