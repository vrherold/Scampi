import json
import torch
from pathlib import Path
from json import JSONDecodeError

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

dtype_cmapping = {
    'float16': torch.complex32,
    'float32': torch.complex64,
    'float64': torch.complex128,
}


class RecoParams:
    def __init__(self, params: dict = None):
        self.params = params

    def from_json(self, file_path: Path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            if data is None:
                raise ValueError("Loaded data is None.")
            self.params = data
        except JSONDecodeError:
            raise ValueError("Something went wrong while loading the JSON file. File may not be valid JSON format.")

    def to_json(self, file_path: Path):
        if self.params is None:
            raise ValueError("No data to save !")
        with open(file_path, "w") as file:
            json.dump(self.params, file, indent=4)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __str__(self):
        max_len = max(len(key) for key in self.params.keys())
        tmp = "\n"
        for key, value in self.params.items():
            space_needed = max_len - len(key)
            tabs = ' ' * space_needed + " "
            tmp += f'{key} {tabs} : {value}  \n'
        return tmp
