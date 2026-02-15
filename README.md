# SM-GAT: Statistically Modulated Graph Neural Network for Encrypted Traffic Classification

This repository contains the official PyTorch implementation of the paper "Bridging Micro-Packet and Macro-Session: A Statistically Modulated Graph Neural Network for Encrypted Traffic Classification", submitted to KDD 2026.

![](C:\Users\90408\Desktop\实验结果\framework.png)

## Abstract

Encrypted traffic classification is fundamental to network security but struggles with temporal distribution drift (model aging). Existing methods often treat traffic as rigid sequences or static vectors, which are fragile when micro-level side-channel measurements fluctuate.

SM-GAT is a statistically modulated session-graph framework that bridges micro-level packet instability and macro-level session stability. It features:

1.  Drift-Resilient Node Encoding: Combines Gaussian soft-quantization with session-conditioned FiLM modulation to stabilize packet features.
2.  Relation-Aware Graph Reasoning: A session-centric interaction graph that treats timing as contextual edge attributes rather than rigid positions.

## Repository Structure

```text
.
├── checkpoints/          # Directory for saving model weights (created automatically)
├── data/                 # Directory for processed datasets
├── models/               # Neural network architecture definitions
│   ├── graph_encoder.py  # Macro-scale Graph Attention Network 
│   └── node_encoder.py   # Micro-scale Node Encoder 
├── scripts/              # Data processing and loading utilities
│   ├── dataset.py        # PyTorch Geometric Dataset loader
│   └── preprocess.py     # Data preprocessing script 
├── requirements.txt      # Python dependencies
├── test.py               # Evaluation script
└── train.py              # Main training loop
```

## Requirements

The code is implemented in Python 3.8+ and PyTorch. To install the necessary dependencies:

Bash

```
pip install -r requirements.txt
```

*Key Dependencies:*

- `torch` >= 1.12.0
- `torch_geometric`
- `numpy`
- `scikit-learn`
- `tqdm`

##  Usage

### 1. Data Preparation

The model expects raw traffic data in JSON format, organized by application. The directory structure for raw input should be:

Plaintext

```
raw_data/
├── AppName_1/
│   ├── flow_1.json
│   └── ...
├── AppName_2/
│   └── ...
```

Run the preprocessing script to convert raw flows into the dynamic graph format (JSONL). This script handles burst segmentation and dynamic thresholding.

Bash

```
# Basic usage
python preprocess.py --input ./path_to_raw_data --output ./data/processed

# With dynamic thresholding enabled (Recommended for drift resilience)
python preprocess.py --input ./path_to_raw_data --output ./data/processed --dynamic
```

This will generate `train.jsonl`, `valid.jsonl`, and `test.jsonl` in the output directory.

### 2. Training

To train the SM-GAT model, run the training script. The script automatically handles class balancing via memory-level oversampling.

*Note: Hyperparameters (batch size, learning rate, dimensions) can be configured in the `CONFIG` dictionary within `train.py`.*

Bash

```
python train.py
```

The model checkpoints will be saved in the `./checkpoints/` directory. The best model based on validation accuracy is saved as `best_model.pth`.

### 3. Evaluation

To evaluate the model on the test set (or OOD drift datasets), run:

Bash

```
python test.py
```

Ensure the `TEST_SETS` dictionary in `test.py` points to the correct location of your processed test files. By default, it looks for:

- `./data/processed/test_indistribution.jsonl`
- `./data/processed/test_drift_ood.jsonl`

## Model Architecture Details

The implementation corresponds directly to the sections in the paper:

- Node Encoder (`node_encoder.py`):
  - *Gaussian Soft-Quantization:* Handles numerical jitter in packet sizes (Eq. 1-3 in paper).
  - *FiLM Modulation:* Injects global session statistics (Duration, Bytes, Count) to calibrate local features (Eq. 5 in paper).
- Graph Encoder (`graph_encoder.py`):
  - *Dynamic Graph Construction:* Logic resides in `preprocess.py`.
  - *Time Encoding:* Sinusoidal encoding for edge time intervals (Eq. 8 in paper).
  - *Relation-Aware Attention:* GATv2-style message passing conditioned on edge types (Intra-burst, Seq, Anchor) (Eq. 9-11 in paper).

