import json
import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    _repr_indent = 4

    def __init__(self, filename: str, 
                 template_file: str,
                 image_folder:str=None, 
                 seed:int=42):
        
        self.filename = filename
        self.image_folder = image_folder
        self.rng = np.random.default_rng(seed)
        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                self.data.append(line)
        
        with open(template_file, 'r') as f:
            self.templates = json.load(f)

    def get_raw_item(self, index: int):
        return json.loads(self.data[index])

    def get_image(self, image_path: str):
        if self.image_folder is not None:
            image_path = os.path.join(self.image_folder, image_path)
        try:
            image = Image.open(image_path).convert('RGB')
        except IOError:
            raise FileNotFoundError(f"The image file path is incorrect : {image_path}")
        return image

    def get_template(self):
        return self.rng.choice(self.templates)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
            f"ann file: {self.filename}"
        ]
        if self.image_folder is not None:
            body.append(f"image folder: {self.image_folder}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    # noinspection PyMethodMayBeStatic
    def extra_repr(self) -> str:
        return ""