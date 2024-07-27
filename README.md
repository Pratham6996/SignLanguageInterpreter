# Sign Language Interpreter

This project aims to develop a Sign Language Interpreter using computer vision and machine learning techniques. It processes sign language alphabet data to train a model that can interpret signs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Dataset Creation](#dataset-creation)
  - [Model Training](#model-training)
  - [Model Inference](#model-inference)
- [Project Structure](#project-structure)
- [Code Explanation](#code-explanation)
  - [collect_imgs.py](#collect_imgspy)
  - [create_dataset.py](#create_datasetpy)
  - [train_classifier.py](#train_classifierpy)
  - [inference_classifier.py](#inference_classifierpy)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Sign Language Interpreter project involves:
- Collecting sign language alphabet images.
- Processing these images to extract hand coordinates.
- Creating a dataset and training a RandomForestClassifier model.
- Testing the trained model.

The main goal is to facilitate communication for the deaf and hard-of-hearing community by interpreting sign language gestures into readable text.

## Features

- **Data Collection**: Capture images of sign language alphabets.
- **Dataset Creation**: Convert images to hand coordinates arrays.
- **Model Training**: Train a RandomForestClassifier model.
- **Model Inference**: Test the trained model to interpret sign language.

## Requirements

The project requires the following libraries:
- `cv2`
- `numpy`
- `mediapipe`
- `pickle`
- `matplotlib`
- `sklearn`
  - `train_test_split`
  - `accuracy_score`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Pratham6996/SignLanguageInterpreter.git
   cd SignLanguageInterpreter
