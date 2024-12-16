# GRID: Visual Layout Generation - A Novel Framework for Visual Generation Tasks

[![arXiv](https://img.shields.io/badge/arXiv-[paper_id]-b31b1b.svg)](https://arxiv.org/abs/[paper_id])
[![Project Page][]](https://[project_page])
[![License][]](https://opensource.org/licenses/Apache-2.0)

[Add a banner image or GIF showcasing key results]

## Overview
GRID introduces a novel paradigm that reframes visual generation tasks as grid layout problems. By transforming temporal sequences into grid layouts, our framework enables image generation models to process visual sequences holistically, achieving remarkable efficiency and versatility across diverse visual generation tasks.

### Key Features
- **Efficient Inference**: 6-35× faster inference speeds compared to specialized models
- **Resource Efficient**: Requires <1/1000 of computational resources  
- **Versatile Applications**: Supports Text-to-Video, Image-to-Video, Multi-view Generation, and more
- **Preserved Capabilities**: Maintains strong image generation performance while expanding functionality

[Add an architecture diagram]

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
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt



Training
Dataset Preparation
Data Structure

data/
├── train/
│   ├── video1/
│   │   ├── frames/
│   │   └── metadata.json
│   └── video2/
│       ├── frames/
│       └── metadata.json
└── val/



## FLUX.1 Training

### Requirements
You currently need a GPU with **at least 24GB of VRAM** to train FLUX.1. If you are using it as your GPU to control 
your monitors, you probably need to set the flag `low_vram: true` in the config file under `model:`. This will quantize
the model on CPU and should allow it to train with monitors attached. Users have gotten it to work on Windows with WSL,
but there are some reports of a bug when running on windows natively. 
I have only tested on linux for now. This is still extremely experimental
and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all. 

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.

1. Sign into HF and accept the model access here [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Make a file named `.env` in the root on this folder
3. [Get a READ key from huggingface](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so `HF_TOKEN=your_key_here`


### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 

## TODO
- [X] Release the paper
- [ ] Release the training codes
- [ ] Update the project page
- [ ] Release the weights

---

## Change Log

#### 2024-12-17
 - Release the paper ()



