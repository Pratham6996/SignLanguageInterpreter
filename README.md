# Sign Language Interpreter

This project aims to develop a Sign Language Interpreter using computer vision and machine learning techniques. It processes sign language alphabet data to train a model that can interpret signs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
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
  ```sh
  pip install opencv-python numpy mediapipe matplotlib scikit-learn
```
```
## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Pratham6996/SignLanguageInterpreter.git
   cd SignLanguageInterpreter

## Usage

1. Use collect_imgs.py to capture sign language alphabet images
```sh
python collect_imgs.py
```
2.Convert the collected images into a dataset of hand coordinates using create_dataset.py
```sh
python create_dataset.py
```
3. Train the model using the created dataset with train_classifier.py
```sh
python train_classifier.py
```
4. Test the trained model using inference_classifier.py
```sh
python inference_classifier.py
```

## Contributing 

Contributions are welcome! Please follow these steps to contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.
