"""Data loader modules.

Classes
---------
CustomData(torch.utils.data.Dataset)

"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import copy


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for torch data.
    """

    def __init__(self, dict_data):
        self.input = np.moveaxis(
            dict_data["x"], -1, 1
        )  # pytorch expects sample, channel, vertical, horizontal
        self.input_unit = dict_data["temp_target"]
        self.input_co2 = dict_data["emissions_left"]
        self.target = dict_data["y"]

    def __len__(self):
        return len(self.target)

    def __getitem__(
        self, idx
    ):  # reshapes data to be in the correct format for the model
        # Retrieve the input and output data for the given index
        input = self.input[idx, ...]
        input_unit = self.input_unit[idx]
        input_co2 = self.input_co2[idx]
        target = self.target[idx]

        return (
            [
                torch.tensor(input, dtype=torch.float32),
                torch.tensor(input_unit, dtype=torch.float32),
                torch.tensor(input_co2, dtype=torch.float32),
            ],
            torch.tensor(target, dtype=torch.float32),
        )
