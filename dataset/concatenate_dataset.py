from typing import List
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    _repr_indent = 4
    def __init__(self, _datasets: List[Dataset]):
        
        self.concat_dataset = TorchConcatDataset(_datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index: int):
        return self.concat_dataset[index]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
        ]
        for i, ds in enumerate(self.concat_dataset.datasets):
            body.append(f"Subset {i + 1}/{len(self.concat_dataset.datasets)}")
            body += ds.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)