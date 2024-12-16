
# GRID: Visual Layout Generation - A Novel Framework for Unified Visual Generation Tasks
The official implementation of work "GRID: Visual Layout Generation".

[![arXiv](https://img.shields.io/badge/arXiv-[paper_id]-b31b1b.svg)](https://arxiv.org/abs/[paper_id])
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview
GRID introduces a novel paradigm that reframes visual generation tasks as grid layout problems. Built upon FLUX.1 architecture, our framework transforms temporal sequences into grid layouts, enabling image generation models to process visual sequences holistically. This approach achieves remarkable efficiency and versatility across diverse visual generation tasks.

![image](https://github.com/user-attachments/assets/e9f42567-5d73-4ba2-9479-740dd1155171)

### Key Features
- **Efficient Inference**: up to 35× faster inference speeds compared to specialized models
- **Resource Efficient**: Requires <1/1000 of computational resources  
- **Versatile Applications**: Supports Text-to-Video, Image-to-Video, Multi-view Generation, and more
- **Preserved Capabilities**: Maintains strong image generation performance while expanding functionality


## Results
![GIF](https://github.com/user-attachments/assets/c5949f2b-9aeb-426e-abf6-7d95ed06e8be)

Due to upload limits of github, we compress our size from 1024×1024 to 256×256, to see full size of each please refer to:

[vid1](https://github.com/user-attachments/assets/360827f6-a141-4e6c-9a03-67ef0a4ae0e5) [vid2](https://github.com/user-attachments/assets/6f68b74c-7851-4c4e-82d4-730e1e4e50f9) [vid3](https://github.com/user-attachments/assets/08b97df2-aca8-49a7-b0e2-3310ffcb5251) [vid4](https://github.com/user-attachments/assets/46c2c2f4-5522-402c-96ea-f7fb40ec1331) [vid5](https://github.com/user-attachments/assets/9cc29fd9-02d2-4775-80aa-d3aee2308477) [vid6](https://github.com/user-attachments/assets/8e5ab2e6-35d8-4a0d-a3fc-3758756febd2) [vid7](https://github.com/user-attachments/assets/18a56da8-04d3-4fe4-9919-0e1a48dd0dd0)          


![result](https://github.com/user-attachments/assets/df01f2da-1b73-44ab-a473-bfcd999a0c9b)

From left to right: input cat video, and edited results of fox, tiger, and red panda transformations.







## Installation

### Requirements
- Python >= 3.10
- NVIDIA GPU with 24GB+ VRAM
- CUDA 11.6+
- PyTorch >= 1.12


```bash
git clone https://github.com/[username]/GRID.git
cd GRID
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies
```bash
torch>=1.12
transformers>=4.25.1
diffusers>=0.16.1
accelerate>=0.18.0
opencv-python>=4.7.0
numpy>=1.24.3
pillow>=9.5.0
tqdm>=4.65.0
```

## Data Preparation Steps:

### 1. Initial Directory Structure:
```bash
source/
├── train/
│   ├── sequence1/
│   │   └── frame_{1..n}.jpg  # Sequential frames 
│   └── sequence2/
│       └── frame_{1..n}.jpg
└── val/
    └── ...
```
### 2. Grid Layout Generation:
```bash
python tools/concat.py \\
    --input_dir source/train \\
    --output_dir vidgrid \\
    --grid_rows 4 \\
    --grid_cols 6 \\
    --frames_per_grid 24
```
Data Structure:
```bash
vidgrid/
├── vid1.jpg  # 4x6 grid containing 24 frames
└── vid2.jpg  # Each .jpg is a complete sequence
```


### 3. Caption Generation:

```bash
mkdir -p models

# Download GLM-4V-9B weights
# Option 1: From ModelScope
wget https://modelscope.cn/models/ZhipuAI/glm-4v-9b/resolve/main/pytorch_model.bin -O models/glm-4v-9b.bin

# Option 2: From Hugging Face
wget https://huggingface.co/THUDM/glm-4v-9b/resolve/main/pytorch_model.bin -O models/glm-4v-9b.bin

# Option 3: From WiseModel
wget https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B/resolve/main/pytorch_model.bin -O models/glm-4v-9b.bin

python tools/caption_glm.py
```

Final Training Data Structure:
```bash
vidgrid/
├── vid1.jpg  # Grid image
├── vid1.txt  # Corresponding caption
├── vid2.jpg
└── vid2.txt
```
## Training

### FLUX.1-based Training Setup

GRID utilizes FLUX.1 architecture for training. You'll need:
- GPU with minimum 24GB VRAM
- FLUX.1-dev model access and license

#### Setup Steps:
Accept the model license at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), then follow the official setup guide in black-forest-labs/flux repository for deployment and model weights download.

### Training Configuration
1. Copy example config:

cp config/train_lora_4d.yaml config/your_config.yaml

2. Edit configuration parameters
3. Start training:

python run.py config/your_config.yaml

Training can be interrupted safely (except during checkpoint saving) and will resume from the last checkpoint.

## Inference


## Results


## Applications
- Text-to-Video Generation
- Image-to-Video Synthesis
- Multi-view Image Generation
- Video Style Transfer
- Editing

## Benchmarks
[Performance comparison charts]

## TODO
- [x] Release the paper
- [x] Release the training codes
- [ ] Update the demo and project page
- [ ] Release the model weights

## Change Log

#### 2024-12-17
- Release the paper

## Citation


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
