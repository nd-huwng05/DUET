import os
import json
from PIL import Image
from torch.utils.data import Dataset
from joblib import Parallel, delayed

from utils.logger import Logger

def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)

class DatasetTemplate(Dataset):
    def __init__(self, config, transform=None, mode='train'):
        super(DatasetTemplate, self).__init__()
        self.config = config
        self.root = self.config.DATASET.ROOT
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(self.root, "data.json")) as f:
            data_dict = json.load(f)

        if mode == "train":
            train_list = data_dict["train"]["0"]
            self.slices += parallel_load(os.path.join(self.root, "images"), train_list, self.config.DATASET.IMAGE_SIZE)
            self.labels += (len(train_list)) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_list]
        else:
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, self.config.DATASET.IMAGE_SIZE)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img = self.slices[idx]
        label = self.labels[idx]
        img = self.transform(img)
        img_id = self.img_id[idx]
        return img, label, img_id