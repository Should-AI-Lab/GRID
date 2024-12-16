# GRID: Visual Layout Generation - A Novel Framework for Unified Visual Generation Tasks
The official implementation of work "GRID: Visual Layout Generation".

[![arXiv](https://img.shields.io/badge/arXiv-[paper_id]-b31b1b.svg)](https://arxiv.org/abs/[paper_id])
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Results

![demo_img_1](https://github.com/user-attachments/assets/92f33c7b-36e2-43db-b45c-a00c4fb53414)
![demo_img_2](https://github.com/user-attachments/assets/c401d0b2-ca73-4fe9-8754-66ac768a40dd)
![demo_img_3](https://github.com/user-attachments/assets/f1b5469b-41f3-4955-8352-52996cc68076)
![demo_img_4](https://github.com/user-attachments/assets/8c2565e8-ceb8-4080-804e-4e1b18221e1c)
![demo_img_5](https://github.com/user-attachments/assets/6ca23dcf-89b3-48af-9ebd-31d6d30505dd)
![demo_img_6](https://github.com/user-attachments/assets/c93aacf6-608d-4f6f-aec8-f47c3e32a107)
![demo_img_7](https://github.com/user-attachments/assets/5bea45ad-4354-4c11-a876-adbbd08a6d87)


## Overview
GRID introduces a novel paradigm that reframes visual generation tasks as grid layout problems. Built upon FLUX.1 architecture, our framework transforms temporal sequences into grid layouts, enabling image generation models to process visual sequences holistically. This approach achieves remarkable efficiency and versatility across diverse visual generation tasks.

### Key Features
- **Efficient Inference**: up to 35× faster inference speeds compared to specialized models
- **Resource Efficient**: Requires <1/1000 of computational resources  
- **Versatile Applications**: Supports Text-to-Video, Image-to-Video, Multi-view Generation, and more
- **Preserved Capabilities**: Maintains strong image generation performance while expanding functionality

![image](https://github.com/user-attachments/assets/e9f42567-5d73-4ba2-9479-740dd1155171)
Figure 1: Architectural overview of GRID framework, demonstrating the transformation of temporal sequences into grid layouts for efficient visual generation.

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
### Data Structure:
```bash
vidgrid/
├── vid1.jpg  # 4x6 grid containing 24 frames
└── vid2.jpg  # Each .jpg is a complete sequence
```
### 3. Caption Generation:
```bash
python tools/caption_glm.py
```
### Final Training Data Structure:
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
1. Accept the model license at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Create .env file in project root
3. Add your HuggingFace READ token: HF_TOKEN=your_token_here

### Training Configuration
1. Copy example config:

cp config/examples/train_lora_grid_24gb.yaml config/your_config.yaml

2. Edit configuration parameters
3. Start training:

python run.py config/your_config.yaml

Training can be interrupted safely (except during checkpoint saving) and will resume from the last checkpoint.

## Inference
[Coming Soon]

## Results
[Showcase of various visual generation results]

## Applications
- Text-to-Video Generation
- Image-to-Video Synthesis
- Multi-view Image Generation
- Video Style Transfer
- Temporal Consistency Enhancement

## Benchmarks
[Performance comparison charts]

## TODO
- [x] Release the paper
- [ ] Release the training codes
- [ ] Update the project page
- [ ] Release the model weights

## Change Log

#### 2024-12-17
- Release the paper

## Citation
[Coming Soon]

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
