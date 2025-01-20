# Character Generator using Neural Networks

## Overview
![output0](https://github.com/user-attachments/assets/d985d885-9fec-4920-9ff9-b42545e54b4c)
![output0](https://github.com/user-attachments/assets/fe7dd73e-6fe9-481b-8ef3-049ec813062e)

Handwriting this was trained on for reference: 
![originalhandwriting](https://github.com/user-attachments/assets/8813c301-fea7-4b41-a990-03ea09c6a5a2)

Training Example:

<div><img src="https://github.com/user-attachments/assets/e2a5ed3c-a9cc-4c5d-9227-fdd339d6f427" style="height: 50px;"></div>

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
