# Prostatectomy and Biopsy Specimen Segmentation

This project implements a segmentation model for prostatectomy and biopsy specimens using deep learning techniques. The model leverages attention mechanisms and dilated convolutions to enhance feature extraction and improve segmentation accuracy.

## Project Structure

```
prostatectomy-biopsy-segmentation
├── src
│   ├── models
│   │   ├── abg.py               # Squeeze and excitation block for attention mechanism
│   │   ├── abm.py               # Combines spatial and channel attention
│   │   ├── dbn.py               # Implements dilated convolutions for feature enhancement
│   │   ├── afn.py               # Applies attention and up-sampling to encoder features
│   │   ├── encoder.py           # Encoder using pre-trained ResNet with DbN layers
│   │   ├── decoder.py           # Decoder part of the segmentation model
│   │   └── segmentation_model.py # Combines encoder and decoder for segmentation
│   ├── dataset
│   │   └── segmentation_dataset.py # Handles loading images and masks
│   ├── train.py                 # Training loop for the segmentation model
│   ├── evaluate.py              # Evaluation logic for the segmentation model
│   ├── visualize.py             # Functions for visualizing segmentation results
│   └── utils.py                 # Utility functions for data preprocessing and metrics
├── requirements.txt              # Lists project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Files and directories to ignore in version control
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd prostatectomy-biopsy-segmentation
   ```

2. **Install dependencies:**
   It is recommended to create a virtual environment before installing the dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your dataset:**
   Ensure your dataset is organized with images and corresponding masks.

2. **Training the model:**
   You can train the model by running the following command:
   ```bash
   python src/train.py
   ```

3. **Evaluating the model:**
   To evaluate the trained model, use:
   ```bash
   python src/evaluate.py
   ```

4. **Visualizing results:**
   To visualize the segmentation results, run:
   ```bash
   python src/visualize.py
   ```

## Model Architecture

The segmentation model consists of an encoder-decoder architecture with attention mechanisms. The encoder utilizes a pre-trained ResNet model, enhanced with dilated convolutions for better feature extraction. The decoder employs attention and up-sampling techniques to produce high-resolution segmentation maps.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.