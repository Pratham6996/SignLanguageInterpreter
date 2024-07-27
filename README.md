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
- [Contributing](#contributing)
- [License](#license)

## Overview

The Sign Language Interpreter project involves:
- Collecting sign language alphabet images.
- Processing these images to extract hand coordinates.
- Creating a dataset and training a RandomForestClassifier model.
- Testing the trained model.

## Features

- Collect sign language alphabet data.
- Create a dataset with hand coordinates.
- Train a machine learning model.
- Test the model to interpret sign language.

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
