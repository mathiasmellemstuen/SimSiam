# Implementation of SimSiam
A simple PyTorch implementation of the SimSiam algorithm from [Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf). In addition to the implementation of the SimSiam algorithm, this repository contains a simple experiment which compares the training results from SimSiam with and without the stop-gradient operation in the algorithm. This comparison was also performed in [Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf).

## Dataset
The dataset used to generate the results in the report was the tiny-imagenet-200 dataset from [ImageNet](https://image-net.org/).

## Prerequisite
- PyTorch
- Matplotlib
- Numpy