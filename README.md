# Photoacoustic Image Reconstruction with Attention UNet

This repository contains the implementation of an **encoder-decoder architecture with an attention mechanism** for **photoacoustic image reconstruction**. The project includes both a baseline UNet model and an enhanced Attention UNet model for comparison.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Usage](#usage)
   - [Generate Simulated Data](#generate-simulated-data)
   - [Train the Models](#train-the-models)
   - [Evaluate the Models](#evaluate-the-models)


---

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

---

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Generate Simulated Data
To generate the simulated data required for training and evaluation, run:
```bash
python generate_simulated_data.py --num_samples 5000
```

### Train the Models
You can train both the **Attention UNet** and the **baseline UNet** models using the following commands:

- **Train the Attention UNet model**:
  ```bash
  python train.py --model_name attention_unet --num_epochs 200
  ```

- **Train the baseline UNet model**:
  ```bash
  python train.py --model_name unet --num_epochs 200
  ```

### Evaluate the Models
After training, evaluate the performance of the models using the test script:

- **Evaluate the Attention UNet model**:
  ```bash
  python test.py --model_name attention_unet
  ```

- **Evaluate the baseline UNet model**:
  ```bash
  python test.py --model_name unet
  ```
