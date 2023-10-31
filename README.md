# cloud-training-workshop


## Data preparation

To get started with this workshop, you can refer to the [`odsc_workshop_data.ipynb`](https://github.com/Lightning-AI/cloud-training-workshop/blob/main/odsc_workshop_data.ipynb) notebook. This notebook is divided into two key sections:

1. **Optimizing ImageNet with Lightning:** The first part of this notebook provides a step-by-step walkthrough of the process required to use Lightning to optimize ImageNet.

2. **Training ResNet Efficiently:** The second part utilizes the optimized ImageNet to train ResNet models swiftly and efficiently, allowing you to leverage the benefits of the ImageNet optimization for your model training.


## Training

In the second part of this workshop, we use [`odsc_workshop_train.py`](https://github.com/Lightning-AI/cloud-training-workshop/blob/main/odsc_workshop_train.py) to train ResNet18 on a subset of the data using the Lightning trainer. This part illustrates how simple it is to implement various training optimizations using Lightning, such as multi-GPU training and different precision techniques.
