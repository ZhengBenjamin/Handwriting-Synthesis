import os 
import csv
import tkinter as tk
import pandas as pd
import numpy as np

from DataProcessing import DataProcessing
from PIL import Image, ImageDraw
from tkinter import ttk

class MakeVectors():
  
  def __init__(self, num_intermediates):
    self.num_intermediates = num_intermediates 
    self.output_folder = "training"
    self.output_file = "training/vectors.csv"
    
    os.makedirs(self.output_folder, exist_ok=True)

    self.data_vectors = []
    self.vector_character = []
    self.intervals = [(48, 57), (97, 122), (65, 90)] # ASCII values for all numbers, lowercase and uppercase letters
    
    for intervals in self.intervals:
      for i in range(intervals[0], intervals[1] + 1):
        for j in range(7): # 7 vectors per character
          self.make_vector(i, j)
    
    self.save_vectors() 
      
  def make_vector(self, char, i):
    """ Creates a vector for each character """
    window = tk.Tk()
    window.title(f"Make Vector {chr(char)}, {i + 1}/7")
    vectors = []
    print(f"Make Vector {chr(char)}, {i + 1}/10")
    
    # Get screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    # Calculate position x, y to center the window
    position_x = int(screen_width / 2 - 512 / 2)
    position_y = int(screen_height / 2 - 512 / 2)
    
    window.geometry(f"512x512+{position_x}+{position_y}")
    
    canvas = tk.Canvas(window, width=256, height=256, bg="white") 
    canvas.pack()
    
    image = Image.new("1", (256, 256), 1)
    draw = ImageDraw.Draw(image)
    # Draw grid lines
    canvas.create_line(0, 85, 256, 85, fill="gray", dash=(2, 2))  # Horizontal line
    canvas.create_line(0, 170, 256, 170, fill="gray", dash=(2, 2))  # Horizontal line
    canvas.create_line(85, 0, 85, 256, fill="gray", dash=(2, 2))  # Vertical line
    canvas.create_line(170, 0, 170, 256, fill="gray", dash=(2, 2))  # Vertical line
    
    self.last_x, self.last_y = None, None
    stroke_vector = []

    def on_hold(event):
      # Paint
      x1, y1 = (event.x - 1), (event.y - 1)
      x2, y2 = (event.x + 1), (event.y + 1)
      canvas.create_oval(x1, y1, x2, y2, fill="black", width=7)
      draw.line([x1, y1, x2, y2], fill="black", width=10)
      
      curr_x = int(event.x)
      curr_y = int(event.y)
      
      # Save vector details 
      if self.last_x == None:
        self.last_x = curr_x
        self.last_y = curr_y
      else:
        
        if self.last_x != curr_x and self.last_y != curr_y:
          stroke_vector.append([self.last_x, self.last_y, curr_x, curr_y])
          self.last_x = curr_x
          self.last_y = curr_y
      
    def on_release(event):
      stroke_vector.append([self.last_x, self.last_y, int(event.x), int(event.y)])
      self.last_x, self.last_y = None, None
      vectors.append(stroke_vector.copy())
      stroke_vector.clear()
      
    canvas.bind("<B1-Motion>", on_hold)
    canvas.bind("<ButtonRelease-1>", on_release)
    
    def compress_stroke_vector(vector, num_intermediates):
      # Flatten the stroke into a list of points
      points = []
      for segment in vector:
        points.append((segment[0], segment[1]))
        points.append((segment[2], segment[3]))

      # Remove duplicates and maintain order
      points = list(dict.fromkeys(points))

      # Compute total path length
      distances = [np.linalg.norm(np.subtract(points[i + 1], points[i])) for i in range(len(points) - 1)]
      total_length = sum(distances)

      # Determine evenly spaced distances
      if total_length == 0 or len(points) < 2:
        return [[0, 0] for _ in range(num_intermediates)]  # Handle edge cases
      segment_lengths = np.cumsum([0] + distances)
      target_distances = np.linspace(0, total_length, num_intermediates)

      # Interpolate points along the path
      compressed = []
      current_index = 0
      for target_distance in target_distances:
        while current_index < len(segment_lengths) - 1 and segment_lengths[current_index + 1] < target_distance:
          current_index += 1
        t = (target_distance - segment_lengths[current_index]) / (segment_lengths[current_index + 1] - segment_lengths[current_index])
        interpolated_x = int((1 - t) * points[current_index][0] + t * points[current_index + 1][0])
        interpolated_y = int((1 - t) * points[current_index][1] + t * points[current_index + 1][1])
        compressed.append([interpolated_x, interpolated_y])
      return compressed
    
    def save_image():
      window.destroy()
      
      compressed_vectors = [] 
      for stroke_vector in vectors:
        compressed_stroke_vector = compress_stroke_vector(stroke_vector, self.num_intermediates)
        
        while len(compressed_stroke_vector) < self.num_intermediates: # Approprate length for training
          compressed_stroke_vector.append([0, 0])
          
        compressed_vectors += compressed_stroke_vector
        
      while len(compressed_vectors) < 4 * self.num_intermediates: # 4 strokes per vector 
        compressed_vectors.append([0, 0])
      
      self.data_vectors.append(compressed_vectors)
      self.vector_character.append(char)
      
      DataProcessing.draw_vector(compressed_vectors, f"{self.output_folder}/{chr(char)}_{i}.png")

    save_button = ttk.Button(window, text="Save", command=save_image)
    save_button.pack()
    
    window.mainloop()
    
  def save_vectors(self):
    """ Save vectors to a CSV file """
    with open(self.output_file, "w") as file:
      writer = csv.writer(file)
      
      for i, character in enumerate(self.data_vectors):
        row = []
        for coordinate in character:
          row += coordinate
        writer.writerow([self.vector_character[i]] + row)