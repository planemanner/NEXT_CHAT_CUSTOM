import torch
from torch import nn
from torch.nn import functional as F


class MMProjector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super(MMProjector, self).__init__()
        """
        This project is used to align the vision and text features in terms of dimension.
        """
        modules = [nn.Linear(input_dim, out_dim),
                   nn.GELU(),
                   nn.Linear(out_dim, out_dim)]

        self.projector = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        return self.projector(x)


class LocEncoder(nn.Module):
    def __init__(self, out_dim):
        super(LocEncoder, self).__init__()
        
        self.layer_1 = nn.Linear(4, out_dim //2)
        self.layer_2 = nn.Linear(out_dim // 2, out_dim)
        self.relu = nn.ReLU()

    def forward(self, loc_info) -> torch.Tensor:
        """
        loc_info : box info
        """
        loc_enc = self.relu(self.layer_1(loc_info))
        return self.layer_2(loc_enc)
    

class LocDecoder(nn.Module):
    def __init__(self, input_dim):
        super(LocDecoder, self).__init__()
        
        self.layer_1 = nn.Linear(input_dim, input_dim //2)
        self.layer_2 = nn.Linear(input_dim // 2, 4)
        self.relu = nn.ReLU()

    def forward(self, loc_enc: torch.Tensor) -> torch.Tensor:
        """
        loc_enc : encoded location information
        """
        loc_dec = self.relu(self.layer_1(loc_enc))
        return self.layer_2(loc_dec)