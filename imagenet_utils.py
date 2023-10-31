import os
import json
import numpy as np
import torchvision.transforms as T
import os
from PIL import Image

def load_imagenet_class_names_to_index_map():
    # download ImageNet class mapping
    os.system("curl https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json -o ~/imagenet_class_index.json")
    
    # Find the associate class for each filepath 
    with open(os.path.join(os.path.expanduser('~'), "imagenet_class_index.json"), "r") as f:
        data = json.load(f)

    return {v[0]: int(k) for k, v in data.items()}

def get_class_index_from_filepath(filepath):
    class_name = filepath.split("/")[-2]
    return int(class_names_to_index_map[class_name])

def shuffle(x):
    return np.random.permutation(x).tolist()


def to_rgb(img):
    if img.shape[0] == 1:
        img = img.repeat((3, 1, 1))
    if img.shape[0] == 4:
        img = img[:3]
    return img


def load_imagenet_val_class_names():
    os.system("curl https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt -o ~/imagenet_val_labels.txt")

    with open(os.path.join(os.path.expanduser('~'), "imagenet_val_labels.txt"), "r") as f:
        return f.read().split("\n")

class_names_to_index_map = load_imagenet_class_names_to_index_map()