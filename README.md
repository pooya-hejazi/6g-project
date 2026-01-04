# Robust User-Centric Clustering in 6G Cell-Free MIMO

This repository contains the official implementation of the paper: **"Robust User-Centric Clustering in 6G Cell-Free MIMO: A Joint Deep Learning Framework"**.

## Abstract
We propose a joint deep learning framework that integrates:
1.  **Hybrid LS-CNN Estimator**: Reconstructs channels from sparse pilots.
2.  **Robust Resource Allocation**: Optimizes user clustering under CSI uncertainty.
3.  **Mimic MLP**: Distills the optimization solver into a real-time neural network.

## Project Structure
* `main.py`: The complete pipeline (Data Gen -> Estimation -> Optimization -> Mimic Learning).
* `config.yaml`: Configuration parameters for SNR, antennas, and training.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/pooya-hejazi/6g-project.git](https://github.com/pooya-hejazi/6g-project.git)
    cd robust-6g-clustering
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) Setup DeepMIMO:
    * Download the **O1_60** scenario from [DeepMIMO.net](https://deepmimo.net/).
    * Extract it into a folder named `DeepMIMO_Dataset`.
    * *Note: If DeepMIMO is not found, the code automatically falls back to a synthetic ray-based channel generator for reproducibility.*

## Usage

Run the full pipeline:
```bash
python main.py --config config.yaml
