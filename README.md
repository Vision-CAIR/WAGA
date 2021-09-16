# Wölfflin Affective Generative Analysis

This repo contains the code for implementation of StyleCAN1 and StyleCAN2 models proposed in the paper Wölfflin Affective Generative Analysis for Visual Art published in International Conference on Computational Creativity (ICCC) 2021.
The implementations for StyleGAN1 and StyleGAN2 models are taken from [rosinality/style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch) and [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) respectively.

## Requirements

- PyTorch 1.3.1
- CUDA 10.1/10.2
  
## Usage

First download the wikiart dataset from [here](https://www.dropbox.com/s/ssw0fdcdld50o1g/wikiartimages.zip/).

Extract the data in the root directory.

### StyleCAN1
To run StyleCAN1.

```bash
cd StyleCAN1
bash stylecan1.sh
```

### StyleCAN2
To run StyleCAN2.

```bash
cd StyleCAN2
bash stylecan2.sh
```

## Sample Results

### StyleCAN1

![StyleCAN1 sample]("./samples/StyleCAN1.png")

### StyleCAN2

![StyleCAN2 sample]("./samples/StyleCAN2.png")


