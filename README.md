# Medical Image Classification with PyTorch

## Project Overview
This project implements a deep learning solution for medical image classification using the Medical MNIST dataset. The model uses a modified ResNet architecture to classify medical images into 6 different categories.

## Dataset
The Medical MNIST dataset consists of 58,954 grayscale medical images across 6 classes:
- AbdomenCT
- BreastMRI
- ChestCT
- CXR
- Hand
- HeadCT

## Requirements
```
Python 3.8+
PyTorch 1.9+
torchvision
numpy
matplotlib
seaborn
scikit-learn
PIL
```

To install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
medical_classification/
│
├── data/
│   └── medical_mnist/
│       ├── raw/
│       ├── train/
│       ├── val/
│       └── test/
│
├── prepare.py
├── train.py
├── requirements.txt
└── README.md
```

## Dataset Preparation
1. Download the Medical MNIST dataset
2. Place the raw images in `data/medical_mnist/raw/` with the following structure:
```
raw/
├── AbdomenCT/
├── BreastMRI/
├── ChestCT/
├── CXR/
├── Hand/
└── HeadCT/
```

3. Run the preparation script:
```bash
python data-prepare.py
```

This will:
- Split the dataset into train/validation/test sets
- Apply necessary preprocessing
- Save the processed data in respective directories

## Training the Model
To train the model:
```bash
python train.py
```

Training parameters can be modified in train.py:
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 10
- Device: CPU/GPU

## Model Architecture
The model uses a modified ResNet18 architecture:
- Modified first layer for grayscale input
- Custom fully connected layers
- Dropout for regularization
- 6-way classification output

## Output
The training process will generate:
- Training metrics plot (training_metrics.png)
- Confusion matrix (confusion_matrix.png)
- Model checkpoints
- Results summary (results.txt)

All outputs will be saved in a timestamped directory: `medical_training_YYYYMMDD_HHMMSS/`
