# Character Generator using Neural Networks

## Overview

This project leverages a deep learning approach to generate handwritten characters or words from input strings. The model is trained on vectorized character representations and produces visually accurate images using a Long Short-Term Memory (LSTM) network combined with a fully connected layer. This system is ideal for tasks involving handwriting synthesis or character recognition enhancement.

## Examples: 
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/2db5bc60-fcce-49e9-8668-da0f2c10c069" style="height: 80px;">
  <img src="https://github.com/user-attachments/assets/d46dd87f-92d8-4492-b2d3-e9ac9c18a0d8" style="height: 80px;">
  <img src="https://github.com/user-attachments/assets/7061f777-8c3e-44c9-8c78-eeb2fd3b0577" style="height: 80px;">
</div>

Handwriting this was trained on for reference: 
<img src="https://github.com/user-attachments/assets/39a7bc95-fb02-4894-8f36-5a7ab9c60495" style="height: 80px;">

Training Example:

<img src="https://github.com/user-attachments/assets/e2a5ed3c-a9cc-4c5d-9227-fdd339d6f427" style="height: 50px;">

## Implementation

The architecture consists of an LSTM network that processes vectorized representations of characters. The LSTM captures sequential dependencies in the strokes of handwritten characters. Outputs are then passed through a fully connected layer to predict the next strokes.
- **Data Generation:** Vectorized character strokes are generated and preprocessed.
- **Model Training:** The LSTM-based model is trained to minimize the Mean Squared Error (MSE) loss between predicted and actual vectors.
- **Character Rendering:** Predicted vectors are converted into images using the PIL library.

## How It Works

### 1. Data Preparation
- The `MakeVectors` module generates vectorized representations of handwritten characters.
- `DataProcessing`:
  - Scales input and output data using `MinMaxScaler`.
  - Splits data into training and testing sets.
  - Converts strokes into coordinate-based arrays for image rendering.

### 2. Model Architecture
The `CharGenModel` consists of:
- **Input Layer:** Accepts vectorized character strokes (shape: `[batch_size, 4, 70]`).
- **LSTM Layer:** Captures sequential stroke dependencies with 512 hidden units.
- **Fully Connected Layer:** Maps the LSTM outputs to a 70-dimensional vector for the next set of strokes.

### 3. Training
The model is trained using:
- **Optimizer:** Adam with a learning rate of 0.0005.
- **Loss Function:** Mean Squared Error (MSE).
- **Checkpointing:** Saves model state and optimizer state every 10,000 epochs for resuming training.

### 4. Character Generation
- Input a string of characters.
- Convert each character to vectors.
- Pass vectors through the trained model.
- Generate images by plotting vectors using PIL.

## Features
- **Pretrained Model:** A pretrained model is included, capable of generating lowercase English letters.
- **Custom Input:** Allows custom strings as input to generate corresponding handwriting.
- **Scalable Training:** Resume training from checkpoints to improve performance.
- **Noise Augmentation:** Adds noise to predictions for natural-looking variations in handwriting.

## Setup
- Generate training data with MakeVectors(). Adjust the resolution of your vectors if higher accuracy is needed. Note: Doing this requires you to change the resolution in other files and may result in higher training times.
- Select device based on your hardware (CUDA for Nvidia, ROCm for AMD, MPS for Apple Silicon GPU etc)
- Adjust training parameters in CharacterGeneration.py.
- Train your model.
- Once your model is trained, you can comment out the training lines and run generation using gen_output_images() 
