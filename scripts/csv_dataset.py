from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

from seamseg.data import ISSTestDataset

class ISSTestCSVDatatset(data.Dataset):
    def __init__(self, input_csv_path, transform):
        super(data.Dataset, self).__init__()
        self.transform = transform
        self._images = pd.read_csv(input_csv_path, index_col=0)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        #return [img_desc.size for img_desc in self._images.itertuples()]
        return [(640,640)] * len(self._images)

    def __len__(self):
        return len(self._images)

    def __get_item__(self, item):
        # Load image
        img_info = self._images.iloc[item]
        with Image.open(img_info.save_loc]) as img_raw:
            size = (img_raw.size[1], img_raw.size[0])
            img = self.transform(img_raw.convert(mode="RGB"))

        return {
            "img": img,
            "idx": img_info.id,
            "size": size,
            "abs_path": img_info.save_loc
        }
